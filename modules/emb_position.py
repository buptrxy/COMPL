import torch, einops 
from torch import nn 
import numpy as np
from timm.models.layers import trunc_normal_

class PositionEmbedding(nn.Module):
    """
    Basic Position Embedding Module
    
    Args:
        size (int): Maximum sequence length.
        dim (int, optional): Embedding dimension. Default is 512.
    """
    def __init__(self, size, dim=512):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size + 1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size + 1, dtype=torch.long).cuda()

    def forward(self, emb):
        """
        Forward pass for position embedding.
        
        Args:
            emb (Tensor): Input embedding tensor.
        
        Returns:
            Tensor: Output tensor with positional encoding applied.
        """
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n - self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids)  # [b, n, pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings

class PPEG(nn.Module):
    """
    Basic Position-aware Convolutional Encoding Module
    
    Args:
        dim (int, optional): Input feature dimension. Default is 512.
        k (int, optional): Kernel size. Default is 7.
        conv_1d (bool, optional): Whether to use 1D convolution. Default is False.
        bias (bool, optional): Whether to include bias. Default is True.
    """
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim, bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim, bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim, bias=bias)

    def forward(self, x):
        """
        Forward pass for PPEG.
        
        Args:
            x (Tensor): Input tensor of shape [B, N, C].
        
        Returns:
            Tensor: Processed tensor with position-aware encoding.
        """
        B, N, C = x.shape

        # Compute spatial dimensions
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:, :add_length, :]], dim=1)  # Padding

        # Ensure minimum size constraint
        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        # Reshape for convolution operations
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        
        # Remove padding
        if add_length > 0:
            x = x[:, :-add_length]
        return x

class PEG(nn.Module):
    """
    Position Encoding Generator (PEG) using depth-wise convolution.
    
    Args:
        dim (int): Number of input and output channels.
        k (int): Kernel size for the convolution.
        bias (bool): Whether to use bias in convolution.
        conv_1d (bool): If True, applies a 1D convolution instead.
    """
    def __init__(self, dim=512, k=7, bias=True, conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim, bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]], dim=1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]
        
        return x

class SINCOS(nn.Module):
    """
    Sinusoidal Positional Encoding Module.
    
    Args:
        embed_dim (int): Dimensionality of the embeddings.
    """
    def __init__(self, embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """Generates 1D sinusoidal position embeddings."""
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega

        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        """Generates 2D sinusoidal position embeddings from a grid."""
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """Computes the full 2D positional embedding grid."""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        B, H, W, C = x.shape
        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
        return x

class APE(nn.Module):
    """
    Absolute Positional Encoding (APE).
    
    Args:
        embed_dim (int): Dimensionality of the embeddings.
        num_patches (int): Number of patches in the input sequence.
    """
    def __init__(self, embed_dim=512, num_patches=64):
        super(APE, self).__init__()
        self.absolute_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

class RPE(nn.Module):
    """
    Relative Positional Encoding (RPE).
    
    Args:
        num_heads (int): Number of attention heads.
        region_size (tuple): Size of the attention region.
    """
    def __init__(self, num_heads=8, region_size=(8,8)):
        super(RPE, self).__init__()
        self.region_size = region_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * region_size[0] - 1) * (2 * region_size[1] - 1), num_heads))
        
        coords_h = torch.arange(region_size[0])
        coords_w = torch.arange(region_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += region_size[0] - 1
        relative_coords[:, :, 1] += region_size[1] - 1
        relative_coords[:, :, 0] *= 2 * region_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        print(relative_position_bias.size())

        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
