import torch
from datamanager import DataManager, DataPartition, retrieve_image
# from model import build_net
from specs import additional_yolo_specs, pretrain_specs
from model import build_yolo_net
import os
from random import randint
from display import display_image
import time
# from statistics import mean

def compute_test_accuracy(model, manager):
    model.eval()
    dev_accuracy = manager.evaluate(model, "test")
    return dev_accuracy
    # print(dev_accuracy)
    
def id(basepath, img_path, model, catagories, jupyter=True):
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
    out = out.reshape((-1, 3, 3, 1 * 5 + 5))
    # print(out.shape)
    out = out.squeeze()
    
    bbs = []
    rcs = []
    colors = []
    labs = []
    for row in range(3):
        for col in range(3):
            pred = out[row,col,:5]
            conf = out[row,col,5]
            # if conf > 0 or max(pred) > 3 * torch.mean(pred):
            if conf > 0:
                
                # print(img_path)
                # print("Confidence:", conf) 
                # print("Prediction Weights:", pred)
                # this may give O alot bc they are all 0
                if torch.sum(out[row,col,:5]) == 0:
                    shape = "?"
                else:   
                    shape = catagories[(out[row,col,:5].argmax()).item()]
                print("Prediction:", shape)
                # print(out[row,col, 6:])
                
                bbs.append(out[row,col, 6:])
                rcs.append((row, col))
                colors.append("red" if conf > 0 else "black")
                labs.append(shape)
                
                # print()
            
    if jupyter:
        return display_image(os.path.join(basepath, img_path), bbs, rcs, colors, labs, jupyter=jupyter)
    
    
    # return ""
    # return out
    
    
def id_ui(basepath, model, catagories):
    while True:
        im_path = input("Enter Image Path: ")
        id(basepath, im_path, model, catagories, jupyter=False)

def id_ui_test(basepath, model, catagories):
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
    # model.load_state_dict(torch.load("yolo_small.pt"))
    model.load_state_dict(torch.load("models/yolo.pt"))
    
    model.eval()
    model.requires_grad_(False)

    # print(compute_test_accuracy(model, manager))
    print(manager.categories)
    id_ui_test("DEMO/test/", model, manager.categories)
    
if __name__ == "__main__":
    main()