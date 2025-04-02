import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from position_embedding import positionalencoding2d
import math
from timm.models.layers import trunc_normal_

'''
modified from ViT-pytorch
The encoder includes FeedForward module and self-Attention module.
github: https://github.com/lucidrains/vit-pytorch
'''
class PreNorm(nn.Module):
    """
    Pre-normalization wrapper for a function.
    
    Args:
        dim (int): Dimension of input features.
        fn (nn.Module): Function to apply (e.g., attention or feed-forward network).
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    Feed-forward network consisting of two linear layers with GELU activation and dropout.
    
    Args:
        dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Basic self-attention module with multi-head attention.
    
    Args:
        dim (int): Input feature dimension.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # Compute query, key, and value from input
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute attention scores
        dots = torch.matmul(self.norm(q), self.norm(k).transpose(-1, -2)) * self.scale

        # Apply attention and dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Compute output from attention
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Rel_Attention(nn.Module):
    """
    Relative positional self-attention module with optional class token.
    
    Args:
        dim (int): Input feature dimension.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout probability for regularization.
        have_cls_token (bool): Whether to include a class token in attention.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., have_cls_token=True, **kwargs):
        super().__init__()

        self.have_cls_token = have_cls_token
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        num_rel_position = 2 * kwargs['window_size'] - 1

        # Create relative position bias indices
        h_range = torch.arange(kwargs['window_size']).cuda()
        w_range = torch.arange(kwargs['window_size']).cuda()
        grid_x, grid_y = torch.meshgrid(h_range, w_range)
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (kwargs['window_size'] - 1)
        self.bias_indices = (grid * torch.tensor([1, num_rel_position]).cuda()[:, None, None]).sum(dim=0)

        # Define relative positional encoding
        if kwargs['shared_pe']:
            self.rel_pe = nn.Embedding(num_rel_position**2, 1)
        else:
            self.rel_pe = nn.Embedding(num_rel_position**2, heads)

        trunc_normal_(self.rel_pe.weight, std=0.02)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim_head)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # Compute relative position bias
        rel_position_bias = self.rel_pe(self.bias_indices)
        rel_position_bias = rearrange(rel_position_bias, 'i j h -> () h i j')

        # Compute query, key, and value from input
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute attention scores with relative position bias
        dots = torch.matmul(self.norm(q), self.norm(k).transpose(-1, -2)) * self.scale
        if self.have_cls_token:
            dots[:, :, 1:, 1:] += rel_position_bias
        else:
            dots += rel_position_bias

        # Apply attention and dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Compute output from attention
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """
    Transformer module consisting of multiple layers of self-attention and feed-forward networks.
    
    Args:
        dim (int): Input feature dimension.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the hidden layer in the feed-forward network.
        dropout (float): Dropout probability for regularization.
        attn_type (str): Type of attention ('sa' for self-attention or 'rel_sa' for relative self-attention).
        shared_pe (bool): Whether to use shared positional encoding.
        window_size (int): Size of the attention window.
        have_cls_token (bool): Whether to include a class token in attention.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 attn_type='sa', shared_pe=None, window_size=None,
                 have_cls_token=True):
        super().__init__()
        self.layers = nn.ModuleList([])

        # Select attention layer type
        if attn_type == 'sa':
            attn_layer = Attention
        elif attn_type == 'rel_sa':
            attn_layer = Rel_Attention

        # Initialize layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn_layer(dim, heads=heads, dim_head=dim_head, 
                                        dropout=dropout,
                                        shared_pe=shared_pe,
                                        window_size=window_size,
                                        have_cls_token=have_cls_token)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # Apply each layer in the transformer
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PyramidViT(nn.Module):
    """Pyramid Vision Transformer (PyramidViT)
    
    Args:
        num_patches (int): The number of patches for the input image.
        embed_weights (tensor or None): Predefined embedding weights or None for learnable weights.
        patch_dim (int): The dimensionality of each patch.
        dim (int): The dimensionality of the transformer model.
        depths (list of int): The depths (number of layers) for each stage of the transformer.
        heads (int, optional): The number of attention heads. Defaults to 4.
        mlp_dim (int, optional): The dimension of the MLP. Defaults to 512.
        pool (str, optional): Pooling strategy ('cls' or 'mean'). Defaults to 'cls'.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 64.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        emb_dropout (float, optional): Dropout rate for embedding layers. Defaults to 0.0.
        ape (bool, optional): Whether to use absolute positional encodings. Defaults to True.
        attn_type (str, optional): Attention type ('rel_sa' or 'sa'). Defaults to 'rel_sa'.
        shared_pe (bool, optional): Whether to use shared positional encodings across stages. Defaults to True.
    """
    
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, heads=4,
                 mlp_dim=512, pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        # Verify that pooling type is either 'cls' or 'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        # Store patch dimension and model dimension
        self.patch_dim = patch_dim
        self.dim = dim
        self.ape = ape  # Flag for using absolute positional encodings
        self.embed_weights = embed_weights  # Embedding weights (either predefined or learnable)
        
        # Patch embedding layers for 20x20, 10x10, and 5x5 patches
        self.to_patch_embedding_20 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_10 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_5 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

        # Initialize positional encodings if 'ape' is True
        if ape:
            addition = 1 if pool == 'cls' else 0  # Adjust based on pooling type
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 16 + addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)  # Truncated normal initialization
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 4 + addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)
            self.pos_emb_5 = nn.Parameter(torch.zeros(1, 1 + addition, dim))
            trunc_normal_(self.pos_emb_5, std=0.02)

        # Initialize class tokens if pooling is 'cls'
        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_5 = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)
            trunc_normal_(self.cls_token_5, std=0.02)

        # Dropout for embedding layer
        self.dropout = nn.Dropout(emb_dropout)

        # Check depths length to ensure proper number of transformer stages
        assert len(depths) == 5
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 4, have_cls_token)
        self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_10 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 2, have_cls_token)
        self.transformer_10_to_5 = Transformer(dim, depths[3], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_5 = Transformer(dim, depths[4], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 1, have_cls_token)

        # If no embedding weights are provided, initialize learnable weights
        if embed_weights is None:
            print('learnable embedding weights')
            self.learned_weights = nn.Parameter(torch.Tensor(3, 1))
            nn.init.kaiming_uniform_(self.learned_weights, a=math.sqrt(5))
        self.ms_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for Pyramid Vision Transformer.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, num_patches, patch_dim).
            
        Returns:
            tensor: Output tensor after applying multiple transformer stages and pooling.
        """
        
        # Remove batch dimension
        x = x.squeeze(0)
        b, _, _ = x.shape  # Extract batch size
        
        # Split input into 3 groups of patches
        x_20 = x[:, :16, :]
        x_10 = x[:, 16:20, :]
        x_5 = x[:, 20:, :]

        # Apply patch embedding if patch dimension is not equal to model dimension
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)
            x_5 = self.to_patch_embedding_5(x_5)

        # Handle pooling strategy: 'cls' or 'mean'
        if self.pool == 'cls':
            # Add class token for 20x20 patches
            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b)
            x_20 = torch.cat((cls_token_20, x_20), dim=1)
            if self.ape:
                x_20 += self.pos_emb_20  # Add positional encoding
            x_20 = self.dropout(x_20)
            x_20 = self.transformer_20(x_20)  # Apply transformer to 20x20 patches
            x_20_cls_token = x_20[:, 0, :]
            x_20 = x_20[:, 1:, :]

            # Rearrange 20x20 patches and concatenate with 10x10 patches
            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)
            x_20_10 = torch.cat((x_10, x_20), dim=1)
            x_20_10 = self.transformer_20_to_10(x_20_10)
            x_10 = x_20_10[:, 0:1, :]

            # Add class token for 10x10 patches
            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)
            x_10 = torch.cat((cls_token_10, x_10), dim=1)
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)
            x_10 = self.transformer_10(x_10)  # Apply transformer to 10x10 patches
            x_10_cls_token = x_10[:, 0, :]
            x_10 = x_10[:, 1:, :]

            # Rearrange 10x10 patches and concatenate with 5x5 patches
            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=1, h2=2, w1=1, w2=2)
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)
            x_10_5 = torch.cat((x_5, x_10), dim=1)
            x_10_5 = self.transformer_10_to_5(x_10_5)
            x_5 = x_10_5[:, 0:1, :]

            # Add class token for 5x5 patches
            cls_token_5 = repeat(self.cls_token_5, '() n d -> b n d', b=b)
            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)
            x_5 = torch.cat((cls_token_5, x_5), dim=1)
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)
            x_5 = self.transformer_5(x_5)
            x_5_cls_token = x_5[:, 0, :]

        elif self.pool == 'mean':
            # Apply mean pooling for the 20x20 patches
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)
            x_20 = self.transformer_20(x_20)
            x_20_cls_token = x_20.mean(dim=1)

            # Reshape and combine 20x20 and 10x10 patches for further processing
            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)
            x_20_10 = torch.cat((x_10, x_20), dim=1)
            x_20_10 = self.transformer_10_to_10(x_20_10)
            x_10 = x_20_10[:, 0:1, :]

            # Apply mean pooling for the 10x10 patches
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)
            x_10 = self.transformer_10(x_10)
            x_10_cls_token = x_10.mean(dim=1)

            # Reshape and combine 10x10 and 5x5 patches for further processing
            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=1, h2=2, w1=1, w2=2)
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)
            x_10_5 = torch.cat((x_5, x_10), dim=1)
            x_10_5 = self.transformer_10_to_5(x_10_5)
            x_5 = x_10_5[:, 0:1, :]

            # Apply mean pooling for the 5x5 patches
            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)
            x_5 = self.transformer_5(x_5)
            x_5_cls_token = x_5.mean(dim=1)

        # Combine the results from different stages using either learned weights or provided weights
        if self.embed_weights is None:
            learned_weights = torch.softmax(self.learned_weights, dim=0)
            x = learned_weights[0] * x_5_cls_token + learned_weights[1] * x_10_cls_token + learned_weights[2] * x_20_cls_token
        else:
            x_stack = torch.stack((self.embed_weights[0] * x_5_cls_token, 
                                   self.embed_weights[1] * x_10_cls_token, 
                                   self.embed_weights[2] * x_20_cls_token))
            x = torch.sum(x_stack, dim=0)

        return x


class PyramidViT_dl(nn.Module):
    """
    Basic Self-Attention Module
    Args:
        num_patches: Number of patches in the input image
        embed_weights: Embedding weights (optional, learnable if None)
        patch_dim: Dimension of each patch
        dim: Dimension of the transformer
        depths: List of depths for each transformer stage
        heads: Number of attention heads for the transformer (default: 4)
        mlp_dim: Dimension of the MLP layer in the transformer (default: 512)
        pool: Pooling type ('cls' for class token pooling, 'mean' for mean pooling)
        dim_head: Dimension of each attention head (default: 64)
        dropout: Dropout rate (default: 0.)
        emb_dropout: Dropout rate for embeddings (default: 0.)
        ape: If True, adds learnable positional embeddings (default: True)
        attn_type: Type of attention used in the transformer ('rel_sa' for relative self-attention)
        shared_pe: If True, uses shared positional embeddings for all stages (default: True)
    """
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, heads=4,
                 mlp_dim=512, pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        # Set patch embedding dimensions and initial parameters
        self.patch_dim = patch_dim
        self.dim = dim
        self.ape = ape
        self.embed_weights = embed_weights
        self.to_patch_embedding_20 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_10 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

        # If positional embedding is required, initialize it
        if ape:
            addition = 1 if pool == 'cls' else 0
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 4 + addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 1 + addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)

        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)

        self.dropout = nn.Dropout(emb_dropout)

        # Initialize the transformers for each stage
        assert len(depths) == 3
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 2, have_cls_token)
        self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_10 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 1, have_cls_token)

        # If no external embedding weights, initialize learnable weights
        if embed_weights is None:
            print('learnable embedding weights')
            self.learned_weights = nn.Parameter(torch.Tensor(2, 1))
            nn.init.kaiming_uniform_(self.learned_weights, a=math.sqrt(5))
        self.ms_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Pyramid Vision Transformer.

        Args:
            x: Input tensor containing image patches

        Returns:
            x: Output tensor after applying the pyramid transformer stages
        """
        x = x.squeeze(0)
        b, _, _ = x.shape

        # Split input tensor into 2 parts based on patch sizes (e.g., 20x20 and 10x10)
        x_20 = x[:, :4, :]
        x_10 = x[:, 4:, :]

        # Embed the patches if necessary
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)

        if self.pool == 'cls':
            # Add class token and positional embeddings for the 20x20 patches
            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b)
            x_20 = torch.cat((cls_token_20, x_20), dim=1)
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)
            x_20 = self.transformer_20(x_20)
            x_20_cls_token = x_20[:, 0, :]
            x_20 = x_20[:, 1:, :]

            # Reshape and combine 20x20 and 10x10 patches for further processing
            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=1, h2=2, w1=1, w2=2)
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)
            x_20_10 = torch.cat((x_10, x_20), dim=1)
            x_20_10 = self.transformer_20_to_10(x_20_10)
            x_10 = x_20_10[:, 0:1, :]

            # Add class token and positional embeddings for the 10x10 patches
            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)
            x_10 = torch.cat((cls_token_10, x_10), dim=1)
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)
            x_10 = self.transformer_10(x_10)
            x_10_cls_token = x_10[:, 0, :]

        elif self.pool == 'mean':
            # Apply mean pooling for the 20x20 patches
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)
            x_20 = self.transformer_20(x_20)
            x_20_cls_token = x_20.mean(dim=1)

            # Reshape and combine 20x20 and 10x10 patches for further processing
            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)
            x_20_10 = torch.cat((x_10, x_20), dim=1)
            x_20_10 = self.transformer_10_to_10(x_20_10)
            x_10 = x_20_10[:, 0:1, :]

            # Apply mean pooling for the 10x10 patches
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)
            x_10 = self.transformer_10(x_10)
            x_10_cls_token = x_10.mean(dim=1)

        # Combine the results from different stages using either learned weights or provided weights
        if self.embed_weights is None:
            learned_weights = torch.softmax(self.learned_weights, dim=0)
            x = learned_weights[0] * x_10_cls_token + learned_weights[1] * x_20_cls_token
        else:
            x_stack = torch.stack((self.embed_weights[0] * x_10_cls_token, 
                                   self.embed_weights[1] * x_20_cls_token))
            x = torch.sum(x_stack, dim=0)

        return x
