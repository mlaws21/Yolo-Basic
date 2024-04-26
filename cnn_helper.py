import torch
from torch.nn import Unfold, Parameter, Module, init, Sequential

def create_kernel_row_matrix(kernels):
    """
    Creates a kernel-row matrix (as described in the notes on
    "Computing Convolutions"). See the unit tests for example input
    and output.
    
    """
    num_kernels = kernels.size(dim=0)
    return torch.reshape(kernels, (num_kernels, -1))
    


def create_window_column_matrix(images, window_width, stride):
    """
    Creates a window-column matrix (as described in the notes on
    "Computing Convolutions"). See the unit tests for example input
    and output.
    
    """
    unfold = Unfold(kernel_size=(window_width, window_width), stride=stride)
    unfolded = unfold(images)
    return torch.cat([x for x in unfolded], dim=1)


def pad(images, padding):
    """
    Adds padding to a tensor of images.
    
    The tensor is assumed to have shape (B, C, W, W), where B is the batch
    size, C is the number of input channels, and W is the width of each
    (square) image.
    
    Padding is the number of zeroes we want to add as a border to each
    image in the tensor.
    
    """
    
    num_images, num_channels, height, width = images.size()
    top = torch.zeros(num_images, num_channels, padding, width)
    side = torch.zeros(num_images, num_channels, height + 2*padding, padding)
    bpad = torch.cat([images, top], dim=-2)
    tpad = torch.cat([top, bpad], dim=-2)
    lpad = torch.cat([side, tpad], dim=-1)
    rpad = torch.cat([lpad, side], dim=-1)
    
    return rpad



def convolve(kernels, images, stride, padding):
    """
    Convolves a kernel tensor with an image tensor, as described in the
    notes on "Computing Convolutions." See the unit tests for example input
    and output.
    
    """
    num_kernels, _, height, width = kernels.size()
    # num_images, num_channels, img_height, img_width = images.size()
    krm = create_kernel_row_matrix(kernels)

    padded_images = pad(images, padding)
    num_images, _, img_height, img_width = padded_images.size()

    
    wcm = create_window_column_matrix(padded_images, width, stride)
    flattened = torch.mm(krm, wcm)
    
    
    conv_height = (img_height - height) // stride + 1
    conv_width = (img_width - width) // stride + 1
    
    reshaped = torch.reshape(flattened, (num_kernels, num_images, conv_height, conv_width))
    transposed = torch.transpose(reshaped, 0, 1)
    return transposed
