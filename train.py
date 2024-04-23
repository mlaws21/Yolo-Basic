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
import random

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
    
def import_data(data_path, W=128):
    xtrain = []
    ytrain = []

    # lab = 0
    labs = sorted(os.listdir(data_path))
    if ".DS_Store" in labs: labs.remove(".DS_Store")

    for lab in labs:
        images = os.listdir(os.path.join(data_path, lab))
        if ".DS_Store" in images: images.remove(".DS_Store")

        random.Random(2).shuffle(images)
        for i in images:
            img_path = os.path.join(data_path, lab, i)
            try:
                img = io.read_image(img_path, io.ImageReadMode.RGB)
            except:
                raise Exception(f"The following image is missing or corrupted: {img_path}")
            resize = transforms.Resize((W, W))
            img = resize(img)
            grayscale = transforms.Grayscale(num_output_channels=1)
            img = grayscale.forward(img)
             
            xtrain.append(img)
            bounding_data = None
            # convert lab to a number (tensor), then cat and the bounding data
            y = torch.cat()
            ytrain.append(lab, bounding_data)

        # lab += 1
        print(lab, end=" ", flush=True)

    # return np.array(xtrain), np.array(ytrain)
    # DataLoader()
    return xtrain, ytrain


def show_image(img):
    """Shows an image, represented as a torch tensor of shape (3, W, W)."""
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2)
    imshow(img)



def main():
    # img = retrieve_image("./imagenette2/train/n01440764/ILSVRC2012_val_00002138.JPEG")
    # print(img)
    # show_image(img)
    import_data("imagenette2/train")
if __name__ == "__main__":
    main()