import torch
from torch.nn import Parameter
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from torchvision import io, transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from copy import deepcopy
from tqdm import tqdm
import time
import sys
import json
import os

def retrieve_image(filename, W=128):
    """
    Loads an image, resizes it to WxW pixels, and then converts it into a
    Torch tensor of shape (3, W, W). The "3" dimension corresponds to 
    the blue, green, and red channels of the input image.

    """
    try:
        img = io.read_image(filename, io.ImageReadMode.RGB)
        resize = transforms.Resize((W, W))
        img = resize(img)
        # img = (resize(img).float() - 127) / 256
        # make grayscale
        grayscale = transforms.Grayscale(num_output_channels=1)
        img = grayscale.forward(img)
        return img
    except:
        raise Exception(f"The following image is missing or corrupted: {filename}")
    
def show_image(img):
    """Shows an image, represented as a torch tensor of shape (3, W, W)."""
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2)
    imshow(img)

if __name__ == "__main__":
    img = retrieve_image("./imagenette2/train/n01440764/ILSVRC2012_val_00002138.JPEG")
    print(img)
    show_image(img)