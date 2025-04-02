from torch import nn
from modules.emb_position import *
from modules.datten import *
from modules.rmsa import *
from .nystrom_attention import NystromAttention
from modules.datten import DAttention
from timm.models.layers import DropPath
from modules.PVT import PyramidViT, PyramidViT_dl


def initialize_weights(module):
    """
    Initialize weights for convolutional, linear, and layer normalization layers.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.
    
    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features.
        out_features (int, optional): Number of output features.
        act_layer (nn.Module, optional): Activation function.
        drop (float, optional): Dropout rate.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransLayer(nn.Module):
    """
    Transformer Layer implementing different attention mechanisms.
    
    Args:
        norm_layer (nn.Module, optional): Normalization layer.
        dim (int): Feature dimension.
        head (int): Number of attention heads.
        drop_out (float): Dropout rate.
        drop_path (float): Drop path rate.
        ffn (bool): Whether to include feedforward network.
        ffn_act (str): Activation function for feedforward network.
        mlp_ratio (float): Ratio for feedforward network expansion.
        trans_dim (int): Transformer head dimension.
        attn (str): Attention type.
        n_region (int): Number of regions in attention.
        epeg (bool): Whether to use efficient positional encoding.
        region_size (int): Region size for attention.
        min_region_num (int): Minimum number of regions.
        min_region_ratio (float): Minimum region ratio.
        qkv_bias (bool): Whether to use bias in query, key, value layers.
        crmsa_k (int): Kernel size for Cross-Region Attention.
        epeg_k (int): Kernel size for efficient positional encoding.
    """
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8, drop_out=0.1, drop_path=0., ffn=False, 
                 ffn_act='gelu', mlp_ratio=4., trans_dim=64, attn='rmsa', n_region=8, epeg=False, 
                 region_size=0, min_region_num=0, min_region_ratio=0, qkv_bias=True, crmsa_k=3, 
                 epeg_k=15, **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()

        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=trans_dim,
                heads=head,
                num_landmarks=256,
                pinv_iterations=6,
                residual=True,
                dropout=drop_out
            )
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_out) if ffn else nn.Identity()

    def forward(self, x, need_attn=False):
        """
        Forward pass through the transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor.
            need_attn (bool, optional): Whether to return attention weights.
        
        Returns:
            torch.Tensor: Output tensor.
            (optional) torch.Tensor: Attention weights if need_attn is True.
        """
        x, attn = self.forward_trans(x, need_attn=need_attn)
        if need_attn:
            return x, attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        """
        Perform the forward pass with attention and optional feedforward network.
        """
        attn = None
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))
        x = x + self.drop_path(z)

        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class RRTEncoder(nn.Module):
    """
    Basic Re-embedded Regional Transformer

    Args:
        mlp_dim (int): Dimension of the feedforward network.
        pos_pos (int): Position embedding control.
        pos (str): Type of positional encoding ('none', 'ppeg', 'sincos', 'peg').
        peg_k (int): Kernel size for PEG.
        attn (str): Attention mechanism type (e.g., 'rmsa', 'crmsa').
        region_num (int): Number of regions in the Re-embedded Regional Transformer.
        drop_out (float): Dropout rate.
        n_layers (int): Number of layers in the encoder.
        n_heads (int): Number of attention heads.
        drop_path (float): Drop path rate for stochastic depth.
        ffn (bool): Whether to use a feedforward network.
        ffn_act (str): Activation function for the feedforward network.
        mlp_ratio (float): Ratio for MLP expansion.
        trans_dim (int): Dimension for transformer layers.
        epeg (bool): Whether to use enhanced positional encoding.
        epeg_k (int): Kernel size for enhanced positional encoding.
        region_size (int): Size of each region.
        min_region_num (int): Minimum number of regions.
        min_region_ratio (float): Minimum region ratio.
        qkv_bias (bool): Whether to add bias to the QKV projections.
        peg_bias (bool): Whether to add bias to the PEG.
        peg_1d (bool): Whether to use 1D convolution for PEG.
        cr_msa (bool): Whether to use CR-MSA (Cross-Region MSA).
        crmsa_k (int): Kernel size for CR-MSA.
        all_shortcut (bool): Whether to apply the shortcut connection after each layer.
        crmsa_mlp (bool): Whether to use an MLP layer after CR-MSA.
        crmsa_heads (int): Number of heads for CR-MSA.
        need_init (bool): Whether to initialize weights.
        **kwargs: Additional arguments for further customization.
    """
    def __init__(self, mlp_dim=512, pos_pos=0, pos='none', peg_k=7, attn='rmsa', region_num=8, drop_out=0.1, 
                 n_layers=2, n_heads=8, drop_path=0., ffn=False, ffn_act='gelu', mlp_ratio=4., trans_dim=64, 
                 epeg=True, epeg_k=15, region_size=0, min_region_num=0, min_region_ratio=0, qkv_bias=True, 
                 peg_bias=True, peg_1d=False, cr_msa=True, crmsa_k=3, all_shortcut=False, crmsa_mlp=False, 
                 crmsa_heads=8, need_init=False, **kwargs):
        super(RRTEncoder, self).__init__()
        
        # Set final dimension for the encoder
        self.final_dim = mlp_dim

        # Normalize output with LayerNorm
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        # Initialize encoder layers
        self.layers = []
        for i in range(n_layers - 1):
            self.layers += [TransLayer(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path, 
                                       ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, 
                                       attn=attn, n_region=region_num, epeg=epeg, region_size=region_size, 
                                       min_region_num=min_region_num, min_region_ratio=min_region_ratio, 
                                       qkv_bias=qkv_bias, epeg_k=epeg_k, **kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # Add Cross-Region Multi-Scale Attention (CR-MSA) if specified
        self.cr_msa = TransLayer(dim=mlp_dim, head=crmsa_heads, drop_out=drop_out, drop_path=drop_path, 
                                  ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, 
                                  attn='crmsa', qkv_bias=qkv_bias, crmsa_k=crmsa_k, crmsa_mlp=crmsa_mlp, **kwargs) if cr_msa else nn.Identity()

        # Initialize position embedding based on the type
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        # Set position embedding mode
        self.pos_pos = pos_pos

        # Initialize weights if needed
        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) or (B, C).
        
        Returns:
            Tensor: Output tensor after applying attention layers and normalization.
        """
        shape_len = 3
        # Adjust the shape for 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # Adjust the shape for 4D input (e.g., images)
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # Apply position embedding if required
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        # Apply Cross-Region MSA
        x = self.cr_msa(x)

        # Add shortcut connection if enabled
        if self.all_shortcut:
            x = x + x_shortcut

        # Normalize the output
        x = self.norm(x)

        # Adjust output shape based on the original input
        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1, 2)
            x = x.reshape(batch, C, int(num_patches**0.5), int(num_patches**0.5))
        return x


class COMPL(nn.Module):
    """
    Classification Model with Pyramid Vision Transformer and Re-embedded Regional Transformer.

    Args:
        input_dim (int): Input dimension of the patches.
        mlp_dim (int): Dimension of the MLP layers.
        act (str): Activation function to use ('tanh', 'relu', 'gelu').
        n_classes (int): Number of output classes for classification.
        multi_scale (int): Number of scales for multi-scale feature extraction.
        dropout (float): Dropout rate.
        pos_pos (int): Position embedding control.
        pos (str): Type of positional encoding ('none', 'ppeg', 'sincos', 'peg').
        peg_k (int): Kernel size for PEG.
        attn (str): Attention mechanism type (e.g., 'rmsa').
        pool (str): Pooling method ('attn' or 'avg').
        region_num (int): Number of regions in Re-embedded Regional Transformer.
        n_layers (int): Number of encoder layers.
        n_heads (int): Number of attention heads.
        drop_path (float): Drop path rate.
        da_act (str): Activation function for attention pooling.
        trans_dropout (float): Dropout rate in transformer layers.
        ffn (bool): Whether to use feedforward network.
        ffn_act (str): Activation function for feedforward network.
        mlp_ratio (float): MLP expansion ratio.
        da_gated (bool): Whether to use gated attention.
        da_bias (bool): Whether to add bias in attention pooling.
        da_dropout (bool): Dropout in attention pooling.
        trans_dim (int): Dimension of transformer layers.
        epeg (bool): Whether to use enhanced positional encoding.
        min_region_num (int): Minimum number of regions.
        qkv_bias (bool): Whether to add bias in QKV projections.
        num_patches (int): Number of patches.
        embed_weights (list): Weights for embedding each scale.
        scale_depths (list): Depths for pyramid transformer layers.
        scale_heads (int): Number of attention heads in pyramid.
        scale_mlp_dim (int): MLP dimension for pyramid transformer.
        emb_dropout (float): Dropout rate for embedding.
        ape (bool): Whether to use absolute positional embedding.
        **kwargs: Additional arguments for further customization.
    """
    def __init__(self, input_dim=1024, mlp_dim=256, act='tanh', n_classes=2, multi_scale=3, dropout=0.25, 
                 pos_pos=0, pos='none', peg_k=7, attn='rmsa', pool='attn', region_num=8, n_layers=2, 
                 n_heads=8, drop_path=0., da_act='relu', trans_dropout=0.1, ffn=False, ffn_act='gelu', 
                 mlp_ratio=4., da_gated=False, da_bias=False, da_dropout=False, trans_dim=64, epeg=True, 
                 min_region_num=0, qkv_bias=True, num_patches=21, embed_weights=[0.3333,0.3333,0.3333], 
                 scale_depths=[2, 2, 2, 2, 2], scale_heads=8, scale_mlp_dim=512, emb_dropout=0, ape=True, **kwargs):
        super(COMPL, self).__init__()

        # Add PyramidViT for multi-scale feature extraction
        if multi_scale == 3:
            self.pyramid_vit = PyramidViT(
                num_patches=num_patches, embed_weights=embed_weights, patch_dim=input_dim, dim=mlp_dim, 
                depths=[2, 2, 2, 2, 2], heads=scale_heads, mlp_dim=scale_mlp_dim, dropout=dropout, 
                emb_dropout=emb_dropout, ape=ape
            )
            self.patch_to_emb = [nn.Linear(mlp_dim, mlp_dim)]
        elif multi_scale == 2:
            self.pyramid_vit = PyramidViT_dl(
                num_patches=num_patches, embed_weights=embed_weights, patch_dim=input_dim, dim=mlp_dim, 
                depths=[2, 2, 2], heads=scale_heads, mlp_dim=scale_mlp_dim, dropout=dropout, 
                emb_dropout=emb_dropout, ape=ape
            )
            self.patch_to_emb = [nn.Linear(mlp_dim, mlp_dim)]
        else:
            self.pyramid_vit = None
            self.patch_to_emb = [nn.Linear(input_dim, mlp_dim)]
        
        # Downstream feature re-embedding
        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        # Initialize online encoder (RRTEncoder)
        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim, pos_pos=pos_pos, pos=pos, peg_k=peg_k, attn=attn, region_num=region_num, 
            n_layers=n_layers, n_heads=n_heads, drop_path=drop_path, drop_out=trans_dropout, ffn=ffn, 
            ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, epeg=epeg, min_region_num=min_region_num, 
            qkv_bias=qkv_bias, **kwargs)

        # Pooling function for attention
        self.pool_fn = DAttention(self.online_encoder.final_dim, da_act, gated=da_gated, bias=da_bias, 
                                  dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)

        # Prediction layer
        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, x, return_attn=False, no_norm=False):
        """
        Forward pass through the classification model.

        Args:
            x (Tensor): Input tensor (B, N, C), where B is batch size, N is number of patches, and C is the channel dimension.
            return_attn (bool): Whether to return the attention weights.
            no_norm (bool): Whether to skip normalization.
        
        Returns:
            Tensor: Logits for classification.
            (Optional) Tensor: Attention weights if return_attn=True.
        """
        # PyramidViT for multi-scale feature extraction
        if self.pyramid_vit is not None:
            x = self.pyramid_vit(x)  # Output: b x dim
        else:
            x = self.patch_to_emb(x)  # n*512

        x = self.dp(x)
        
        # Feature re-embedding using online encoder
        x = self.online_encoder(x)
        
        # Feature aggregation with attention pooling
        if return_attn:
            x, a = self.pool_fn(x, return_attn=True, no_norm=no_norm)
        else:
            x = self.pool_fn(x)

        # Final prediction
        logits = self.predictor(x)

        if return_attn:
            return logits, a
        else:
            return logits

