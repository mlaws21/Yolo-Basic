import torch
from datamanager import DataManager, DataPartition, retrieve_image

from specs import additional_yolo_specs, pretrain_specs
from model import build_yolo_net
import os
from random import randint
from display import display_image
import time
THRESHOLD = 0.7

def id(basepath: str, img_path: str, model, catagories: list[str], jupyter=True):
    """ Function to take in an image and identify all objects in the image

    Args:
        basepath (str): root folder for the image
        img_path (str): path to image from root
        model: YOLO-Basic model with params trained and loaded
        catagories (list[str]): list of potential catagories
        jupyter (bool, optional): Whether or not display_image is being called in a
        jupyter notebook. Defaults to True.

    Returns:
        PIL image if jupyter is set to true else None
    """
    
    id_to_col = {"Square": "green", "Star": "yellow", "X": "red", "O": "blue", "Pent":"orange"}
    try:
        img = retrieve_image(os.path.join(basepath, img_path), 112)
    except:
        print(f"ERROR: File {os.path.join(basepath, img_path)} Not Found")
        return
    img = img.unsqueeze(0)
    start = time.time()
    out = model(img)
    end = time.time()
    print("Compute Time:", end - start)
    print("Esimated fps:", 60 / (end - start))
    out = out.reshape((-1, 3, 3, 1 * 5 + 5))
    out = out.squeeze()
    
    bbs = []
    rcs = []
    colors = []
    labs = []
    # look through grid and check if any have a confidence over the threshold
    for row in range(3):
        for col in range(3):
            conf = out[row,col,5]
            if conf > THRESHOLD:
                
                print("Confidence:", conf.item()) 
                if torch.sum(out[row,col,:5]) == 0:
                    shape = "?"
                else:   
                    shape = catagories[(out[row,col,:5].argmax()).item()]
                print("Prediction:", shape)
                
                bbs.append(out[row,col, 6:])
                rcs.append((row, col))
                colors.append(id_to_col[shape])
                labs.append(shape)
            
    if jupyter:
        return display_image(os.path.join(basepath, img_path), bbs, rcs, colors, labs, jupyter=jupyter)
    else:
        display_image(os.path.join(basepath, img_path), bbs, rcs, colors, labs, jupyter=jupyter)
    
    
def id_ui(basepath: str, model: str, catagories: list[str]):
    """UI to run a series of tests

    Args:
        basepath (str): root folder for the image
        img_path (str): path to image from root
        catagories (list[str]): list of potential catagories
    """
    while True:
        im_path = input("Enter Image Path: ")
        id(basepath, im_path, model, catagories, jupyter=False)

def id_ui_test(basepath: str, model: str, catagories: list[str]):
    """UI to run a series of tests simplifed so you only add the letter
    that you want and it find a random image

    Args:
        basepath (str): root folder for the image
        img_path (str): path to image from root
        catagories (list[str]): list of potential catagories
    """
    while True:
        im_path = input("Enter Image Path: ")
        id(basepath, os.path.join(im_path, str(randint(0, 10)) + ".png"), model, catagories, jupyter=False) 


def main():
    data_config = 'DEMO/data.json'
    image_width = 112
    train_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    model = build_yolo_net(pretrain_specs, additional_yolo_specs)
    model.load_state_dict(torch.load("models/yolo.pt"))
    
    model.eval()
    model.requires_grad_(False)

    id_ui_test("DEMO/test/", model, manager.categories)
    
    
if __name__ == "__main__":
    main()