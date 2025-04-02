from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce


def exists(val):
    """Check if a value is not None."""
    return val is not None

def moore_penrose_iter_pinv(x, iters=6):
    """
    Compute the Moore-Penrose pseudoinverse using an iterative approach.

    Args:
        x (Tensor): Input tensor.
        iters (int, optional): Number of iterations for approximation. Defaults to 6.

    Returns:
        Tensor: Pseudoinverse of the input matrix.
    """
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

"""Basic Self-Attention Module"""

class NystromAttention(nn.Module):
    """
    Nystrom-Based Self-Attention Module.

    Args:
        dim (int): Input feature dimension.
        dim_head (int, optional): Dimension per attention head. Defaults to 64.
        heads (int, optional): Number of attention heads. Defaults to 8.
        num_landmarks (int, optional): Number of landmark points for low-rank approximation. Defaults to 256.
        pinv_iterations (int, optional): Number of iterations for Moore-Penrose pseudoinverse. Defaults to 6.
        residual (bool, optional): Whether to use a residual connection. Defaults to True.
        residual_conv_kernel (int, optional): Kernel size for residual depth-wise convolution. Defaults to 33.
        eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-8.
        dropout (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        """
        Forward pass of the Nystrom Attention module.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, dim).
            mask (Tensor, optional): Boolean mask tensor of shape (batch, seq_len). Defaults to None.
            return_attn (bool, optional): Whether to return attention maps. Defaults to False.

        Returns:
            Tensor: Output tensor after self-attention.
        """
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # Pad input sequence to be evenly divisible by the number of landmarks
        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # Compute query, key, and value
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Apply mask to queries, keys, and values
        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # Compute landmarks by sum reduction
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # Compute masked mean if a mask exists
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        # Compute attention similarity matrices
        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # Apply masking to attention matrices
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            attn1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # Apply softmax and compute approximate attention
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # Apply depth-wise convolution residual connection
        if self.residual:
            out += self.res_conv(v)

        # Merge and reshape output
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        # Return attention matrix if required
        if return_attn:
            attn1 = attn1[:, :, 0].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)
            return out, attn1[:, :, 0, -n+1:]

        return out
