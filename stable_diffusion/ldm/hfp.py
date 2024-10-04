import numpy as np
import cv2
import torch.fft
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def rgb_to_grayscale(image):
    if image.dim() == 4 and image.shape[1] == 3:  # Check if it's a batch of images
        # Use the Y' component of the Y'CbCr color space as weights for RGB channels to convert to grayscale
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    elif image.dim() == 3 and image.shape[0] == 3:  # Single image, not a batch
        r, g, b = image[0, :, :], image[1, :, :], image[2, :, :]
        grayscale = 0.299 * r + 0.587 * g + 0.114 * b
        grayscale = grayscale.unsqueeze(0)  # Add the batch dimension
    else:
        raise ValueError('Input tensor must have 3 or 4 dimensions')
    return grayscale


def high_pass_filter(batch_size, size, cutoff):
    crow, ccol = size[0] // 2, size[1] // 2
    rows = torch.arange(size[0]).unsqueeze(-1).repeat(1, size[1]) - crow
    cols = torch.arange(size[1]).repeat(size[0], 1) - ccol
    distance = torch.sqrt(rows**2 + cols**2 + 1e-8)
    mask = (distance > cutoff).float().unsqueeze(0).repeat(batch_size, 1, 1)
    return mask


def extract_high_freq_component(images_tensor, cutoff=30):
    '''
    image_tensor: torch.Tensor, Value range: [0, 1].
    Expecting images_tensor to be of shape (batch_size, channel, height, width)
    '''
    images_tensor = rgb_to_grayscale(images_tensor)
    
    # Apply DFT
    dft = torch.fft.fft2(images_tensor)
    dft_shift = torch.fft.fftshift(dft)
    
    # Create high-pass filter
    mask = high_pass_filter(images_tensor.shape[0], images_tensor[0].shape, cutoff)
    
    # Apply high-pass filter
    filtered_dft = dft_shift * mask.to(images_tensor.device)
    
    # Apply IDFT
    idft_shift = torch.fft.ifftshift(filtered_dft)
    img_back = torch.fft.ifft2(idft_shift)
    img_back = torch.abs(img_back)
    
    return img_back / img_back.max()


def sobel_operator(images_tensor):
    '''
    images_tensor: torch.Tensor, Value range: [0, 1].
    Expecting images_tensor to be of shape (batch_size, channel, height, width)
    '''
    images_tensor = rgb_to_grayscale(images_tensor).unsqueeze(1)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float32, device=images_tensor.device).view((1, 1, 3, 3))
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float32, device=images_tensor.device).view((1, 1, 3, 3))
    edge_x = F.conv2d(images_tensor, sobel_x, padding=1)
    edge_y = F.conv2d(images_tensor, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
    return edge / edge.max()