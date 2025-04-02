import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
import math

'''
modified from Swin Transformer
title: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
github: https://github.com/microsoft/Swin-Transformer
'''

class Mlp(nn.Module):
    """
    MLP (Multilayer Perceptron) Block
    Args:
        in_features (int): Input feature dimension
        hidden_features (int, optional): Hidden feature dimension. Defaults to None.
        out_features (int, optional): Output feature dimension. Defaults to None.
        act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
        drop (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # First fully connected layer
        self.act = act_layer()  # Activation layer (default GELU)
        self.fc2 = nn.Linear(hidden_features, out_features)  # Second fully connected layer
        self.drop = nn.Dropout(drop)  # Dropout layer

    def forward(self, x):
        x = self.fc1(x)  # Pass through first layer
        x = self.act(x)  # Apply activation function
        x = self.drop(x)  # Apply dropout
        x = self.fc2(x)  # Pass through second layer
        x = self.drop(x)  # Apply dropout again
        return x

def region_partition(x, region_size):
    """
    Partition the input tensor into smaller regions
    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C)
        region_size (int): Size of each region
    Returns:
        torch.Tensor: A tensor containing smaller regions with shape (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Reverse the partition operation and reconstruct the original input
    Args:
        regions (torch.Tensor): The partitioned regions with shape (num_regions*B, region_size, region_size, C)
        region_size (int): Size of each region
        H (int): Height of the original image
        W (int): Width of the original image
    Returns:
        torch.Tensor: The reconstructed tensor with shape (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class InnerAttention(nn.Module):
    """
    Inner attention mechanism used for self-attention within regions
    Args:
        dim (int): Input feature dimension
        head_dim (int, optional): Dimension of each attention head. Defaults to None.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to include bias in the qkv layers. Defaults to True.
        qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
        attn_drop (float, optional): Dropout probability for attention. Defaults to 0.
        proj_drop (float, optional): Dropout probability for projection. Defaults to 0.
        epeg (bool, optional): Whether to use an EPeg module. Defaults to True.
        epeg_k (int, optional): Kernel size for EPeg. Defaults to 15.
        epeg_2d (bool, optional): Whether to use 2D EPeg. Defaults to False.
        epeg_bias (bool, optional): Whether to use bias in EPeg. Defaults to True.
        epeg_type (str, optional): Type of EPeg ('attn' or 'value_bf'). Defaults to 'attn'.
    """
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., epeg=True, epeg_k=15, epeg_2d=False, epeg_bias=True, epeg_type='attn'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads  # Calculate head dimension if not provided
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5  # Scaling factor for attention

        # Define layers
        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)  # Linear layer for QKV projection
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout layer for attention
        self.proj = nn.Linear(head_dim * num_heads, dim)  # Projection layer
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout layer for projection

        # Initialize EPeg module if required
        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, epeg_k, padding=padding, groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k, padding=padding, groups=head_dim * num_heads, bias=epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding=(padding, 0), groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (epeg_k, 1), padding=(padding, 0), groups=head_dim * num_heads, bias=epeg_bias)
        else:
            self.pe = None  # No EPeg

        self.softmax = nn.Softmax(dim=-1)  # Softmax for attention scores

    def forward(self, x):
        """
        Forward pass through the attention mechanism
        Args:
            x (torch.Tensor): Input features with shape (num_regions*B, N, C)
        Returns:
            torch.Tensor: Output features after attention
        """
        B_, N, C = x.shape  # Extract batch size, sequence length, and feature dimension

        # Perform QKV projection and reshape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Extract query, key, and value

        q = q * self.scale  # Apply scaling to the query
        attn = (q @ k.transpose(-2, -1))  # Compute attention scores

        # Apply EPeg adjustment if required
        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn + pe  # Add EPeg to attention scores

        attn = self.softmax(attn)  # Apply softmax to attention scores
        attn = self.attn_drop(attn)  # Apply dropout to attention scores

        if self.pe is not None and self.epeg_type == 'value_bf':
            # Adjust the value tensor with EPeg
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Apply attention to value tensor
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        # Apply EPeg adjustment to the output if needed
        if self.pe is not None and self.epeg_type == 'value_af':
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)  # Projection to output dimension
        x = self.proj_drop(x)  # Apply dropout after projection

        return x

    def extra_repr(self) -> str:
        """Return string representation of the module parameters for debugging.

        Returns:
            str: String representation of module parameters.
        """
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """Calculate the number of floating point operations (FLOPs) for the forward pass.

        Args:
            N (int): The length of tokens in each region.

        Returns:
            int: The estimated number of FLOPs.
        """
        flops = 0
        # FLOPs for QKV projections
        flops += N * self.dim * 3 * self.dim
        # FLOPs for attention calculation (q @ k.transpose)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # FLOPs for attention output (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # FLOPs for the final projection
        flops += N * self.dim * self.dim
        return flops

class RegionAttntion(nn.Module):
    """
    A class for region-based attention mechanism in vision models, including native and Nystrom attention types.
    
    Args:
        dim (int): The dimensionality of the input features.
        head_dim (int, optional): The dimension of each attention head. If not provided, calculated from dim.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        region_size (int, optional): The size of each region for partitioning the input. If 0, no region partitioning occurs.
        qkv_bias (bool): Whether to add a bias term to the Q, K, and V projections.
        qk_scale (float, optional): Scaling factor for the dot product of queries and keys.
        drop (float): Dropout rate applied to the attention output.
        attn_drop (float): Dropout rate applied to the attention weights.
        region_num (int): The number of regions to partition the input into.
        epeg (bool): Whether to apply the Enhanced Positional Encoding (EPEG).
        min_region_num (int): The minimum number of regions required for processing.
        min_region_ratio (float): The minimum ratio of regions for processing.
        region_attn (str): The type of region attention ('native' for standard attention or 'ntrans' for Nystrom attention).
    """
    
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., region_attn='native', **kawrgs):
        super().__init__()
        # Initialize the parameters for the region attention mechanism
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        # Choose the type of region attention mechanism
        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

    def padding(self, x):
        """
        Pads the input tensor to ensure it fits evenly into regions. If necessary, adds extra padding to ensure compatibility with region-based attention.

        Args:
            x (Tensor): The input tensor to be padded.
        
        Returns:
            Tuple: A tuple containing the padded tensor, height, width, added length, region number, and region size.
        """
        B, L, C = x.shape
        
        # Calculate region dimensions based on input length and region size or region number
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L
        
        # If excessive padding, adjust the region dimensions or abandon region attention for ablation
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        
        # Apply padding if necessary
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)
        
        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        """
        The forward pass for region-based attention. This involves padding, partitioning the input into regions,
        applying attention to each region, and then merging the results.

        Args:
            x (Tensor): The input tensor to be processed.
            return_attn (bool, optional): Whether to return the attention weights. Defaults to False.
        
        Returns:
            Tensor: The processed tensor after applying region-based attention.
        """
        B, L, C = x.shape
        
        # Apply padding to the input
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # Partition the input into regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # Apply region multi-head self-attention
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # Merge the regions after attention
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        # Reverse the partitioning to reconstruct the input tensor
        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        # Remove any added padding
        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttntion(nn.Module):
    """
    Cross-Region Self-Attention Module

    Args:
        dim (int): The dimensionality of input feature vectors.
        head_dim (int, optional): The dimensionality of each attention head. Defaults to None.
        num_heads (int): The number of attention heads.
        region_size (int, optional): The size of each region. Defaults to 0.
        qkv_bias (bool): Whether to add a bias term to the Q, K, V projections.
        qk_scale (float, optional): Scaling factor for the QK attention mechanism. Defaults to None.
        drop (float): Dropout rate for the attention weights.
        attn_drop (float): Dropout rate for the attention output.
        region_num (int): The number of regions to partition the input into.
        epeg (bool): Whether to use the EPEG method for attention.
        min_region_num (int): The minimum number of regions to be considered.
        min_region_ratio (float): The minimum ratio of regions.
        crmsa_k (int): The number of attention tokens.
        crmsa_mlp (bool): Whether to use MLP for CR-MSA (Cross-Region Multi-Scale Attention).
        region_attn (str): Type of attention used for region-based processing. Defaults to 'native'.
        **kawrgs: Additional keyword arguments for custom configurations.
    """
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., crmsa_k=3, crmsa_mlp=False, region_attn='native', **kawrgs):
        super().__init__()

        # Initialize attributes
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        # Define the attention mechanism (InnerAttention by default)
        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)

        # Define additional components for CR-MSA if needed
        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k, bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            # Use learnable parameters if MLP is not used
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        """
        Pads the input tensor to ensure the region partitioning is valid.

        Args:
            x (torch.Tensor): Input tensor to be padded.

        Returns:
            torch.Tensor: Padded tensor.
            int: Height of the padded input.
            int: Width of the padded input.
            int: Number of padding elements added.
            int: Number of regions after padding.
            int: Size of each region.
        """
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # Check if padding is too large and adjust the region size accordingly
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        """
        Performs the forward pass for Cross-Region Attention.

        Args:
            x (torch.Tensor): Input tensor.
            return_attn (bool): Whether to return the attention weights along with the output.

        Returns:
            torch.Tensor: The output tensor after applying cross-region attention.
        """
        B, L, C = x.shape

        # Pad the input tensor
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # Partition the input into regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # Apply Cross-Region Multi-Scale Attention (CR-MSA)
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)  # W*B, sW, region_size*region_size
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1, 2)  # nW*B, sW, region_size*region_size

        # Compute attention weights
        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        # Normalize the attention weights
        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        # Apply attention to the regions
        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0, 1)  # sW, nW, C

        if return_attn:
            attn_regions, _attn = self.attn(attn_regions, return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0, 1)  # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0, 1)  # nW, sW, C

        # Apply dispatch weights to the attention regions
        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)  # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)  # nW, region_size*region_size, C

        # Merge the attention regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        # Remove padding if added
        if add_length > 0:
            x = x[:, :-add_length]

        return x

