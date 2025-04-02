import math
import torch

def positionalencoding1d(d_model, length, ratio=1):
    """
    Basic 1D Positional Encoding 
    Args:
        d_model (int): The dimension of the model.
        length (int): The length of the positions.
        ratio (float, optional): A scaling factor for the positional encoding. Default is 1.
    
    Returns:
        torch.Tensor: A position matrix of size (length + 1) x d_model.
    """
    # Ensure that the model dimension is even for the sin/cos positional encoding
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    
    # Initialize position matrix with zeros
    pe = torch.zeros(length + 1, d_model)
    
    # Generate position indices
    position = torch.arange(0, length + 1).unsqueeze(1)
    
    # Calculate the division term for the encoding
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)) * ratio
    
    # Apply sine to even indices and cosine to odd indices in the model dimension
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width, ratio=1):
    """
    Basic 2D Positional Encoding 
    Args:
        d_model (int): The dimension of the model.
        height (int): The height of the positions.
        width (int): The width of the positions.
        ratio (float, optional): A scaling factor for the positional encoding. Default is 1.
    
    Returns:
        torch.Tensor: A position matrix of size (height * width + 1) x d_model.
    """
    # Ensure that the model dimension is divisible by 4 for 2D positional encoding
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    
    # Initialize position matrix with zeros
    pe = torch.zeros(height * width + 1, d_model)

    # Half of the model dimension for each axis
    d_model = int(d_model / 2)
    
    # Generate 1D positional encodings for height and width
    height_pe = positionalencoding1d(d_model, height, ratio)
    width_pe = positionalencoding1d(d_model, width, ratio)

    # Set the initial position (0 index) for height and width encodings
    pe[0, :d_model] = height_pe[0]
    pe[0, d_model:] = width_pe[0]

    # Populate the position matrix with 2D positional encodings
    for i in range(height):
        for j in range(width):
            pe[i * width + j + 1, :d_model] = height_pe[i + 1]
            pe[i * width + j + 1, d_model:] = width_pe[j + 1]

    return pe
