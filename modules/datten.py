import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Basic Self-Attention Module
    
    Args:
        input_dim (int): Dimension of input features.
        act (str): Activation function type ('relu', 'gelu', or 'tanh').
        bias (bool): Whether to use bias in linear layers.
        dropout (bool): Whether to apply dropout.
    """
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        # Define the attention mechanism
        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        # Apply activation function based on the specified type
        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        # Apply dropout if enabled
        if dropout:
            self.attention += [nn.Dropout(0.25)]

        # Final linear transformation
        self.attention += [nn.Linear(self.D, self.K, bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        """
        Forward pass for self-attention.
        
        Args:
            x (Tensor): Input tensor of shape (N, L).
            no_norm (bool): If True, return raw attention scores before softmax.
        
        Returns:
            Tensor: Transformed feature representation.
            Tensor: Attention scores.
        """
        A = self.attention(x)  # Compute attention scores
        A = torch.transpose(A, -1, -2)  # Transpose to KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # Apply softmax along the last dimension
        x = torch.matmul(A, x)  # Compute weighted sum of input features
        
        if no_norm:
            return x, A_ori  # Return original attention scores
        else:
            return x, A  # Return normalized attention scores

class AttentionGated(nn.Module):
    """
    Gated Self-Attention Module
    
    Args:
        input_dim (int): Dimension of input features.
        act (str): Activation function type ('relu', 'gelu', or 'tanh').
        bias (bool): Whether to use bias in linear layers.
        dropout (bool): Whether to apply dropout.
    """
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        # Define the first attention pathway
        self.attention_a = [nn.Linear(self.L, self.D, bias=bias)]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        # Define the second attention pathway with sigmoid activation
        self.attention_b = [nn.Linear(self.L, self.D, bias=bias), nn.Sigmoid()]

        # Apply dropout if enabled
        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        # Final linear transformation after gated attention
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False):
        """
        Forward pass for gated self-attention.
        
        Args:
            x (Tensor): Input tensor of shape (N, L).
            no_norm (bool): If True, return raw attention scores before softmax.
        
        Returns:
            Tensor: Transformed feature representation.
            Tensor: Attention scores.
        """
        a = self.attention_a(x)  # Compute attention scores from pathway a
        b = self.attention_b(x)  # Compute gating weights from pathway b
        A = a.mul(b)  # Apply element-wise multiplication (gated attention)
        A = self.attention_c(A)  # Compute final attention scores

        A = torch.transpose(A, -1, -2)  # Transpose to KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # Apply softmax along the last dimension
        x = torch.matmul(A, x)  # Compute weighted sum of input features

        if no_norm:
            return x, A_ori  # Return original attention scores
        else:
            return x, A  # Return normalized attention scores

class DAttention(nn.Module):
    """
    Dynamic Attention Module that supports both standard and gated self-attention.
    
    Args:
        input_dim (int): Dimension of input features.
        act (str): Activation function type ('relu', 'gelu', or 'tanh').
        gated (bool): Whether to use gated attention.
        bias (bool): Whether to use bias in linear layers.
        dropout (bool): Whether to apply dropout.
    """
    def __init__(self, input_dim=512, act='relu', gated=False, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        # Choose between standard attention and gated attention
        if gated:
            self.attention = AttentionGated(input_dim, act, bias, dropout)
        else:
            self.attention = Attention(input_dim, act, bias, dropout)

    def forward(self, x, return_attn=False, no_norm=False, **kwargs):
        """
        Forward pass for dynamic attention.
        
        Args:
            x (Tensor): Input tensor of shape (N, L).
            return_attn (bool): If True, return attention scores.
            no_norm (bool): If True, return raw attention scores before softmax.
        
        Returns:
            Tensor: Transformed feature representation.
            Tensor (optional): Attention scores if return_attn is True.
        """
        x, attn = self.attention(x, no_norm)  # Compute attention-weighted feature

        if return_attn:
            return x.squeeze(1), attn.squeeze(1)  # Return attention scores if requested
        else:   
            return x.squeeze(1)  # Return only the feature output
