import torch
from datamanager import DataManager, DataPartition, retrieve_image
# from model import build_net
from specs import additional_yolo_specs, pretrain_specs, pretrain_small_specs
from train import get_yolo_net
import os
from random import randint
from display_bb import display
# from statistics import mean

def compute_test_accuracy(model, manager):
    model.eval()
    dev_accuracy = manager.evaluate(model, "test")
    return dev_accuracy
    # print(dev_accuracy)
    
def id(basepath, img_path, model, catagories):
    try:
        img = retrieve_image(os.path.join(basepath, img_path), 112)
    except:
        print("ERROR: File Not Found")
        return
    img = img.unsqueeze(0)
    out = model(img)
    out = out.reshape((-1, 3, 3, 1 * 5 + 5))
    print(out.shape)
    out = out.squeeze()
    
    bbs = []
    rcs = []
    colors = []
    for row in range(3):
        for col in range(3):
            pred = out[row,col,:5]
            conf = out[row,col,5]
            if conf > 0 or max(pred) > 3 * torch.mean(pred):
                print(img_path)
                print("Confidence:", conf) 
                print("Prediction Weights:", pred)
                # this may give O alot bc they are all 0
                
                print("Prediction:", catagories[(out[row,col,:5].argmax()).item()])
                # print(out[row,col, 6:])
                
                bbs.append(out[row,col, 6:])
                rcs.append((row, col))
                colors.append("red" if conf > 0 else "black")
                
                print()
            
    display(os.path.join(basepath, img_path), bbs, rcs, colors)
    
    return ""
    # return out
    
    
def id_ui(basepath, model, catagories):
    while True:
        im_path = input("Enter Image Path: ")
        print("Images Contained the Shape:", id(basepath, im_path, model, catagories))

def id_ui_test(basepath, model, catagories):
    while True:
        im_path = input("Enter Image Path: ")
        print("Images Contained the Shape:", id(basepath, os.path.join(im_path, str(randint(0, 99)) + ".png"), model, catagories))    


def main():
    data_config = 'one_shape/data.json'
    image_width = 112
    train_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    model = get_yolo_net(pretrain_small_specs, additional_yolo_specs)
    model.load_state_dict(torch.load("yolo_small.pt"))
    model.eval()
    model.requires_grad_(False)

    # print(compute_test_accuracy(model, manager))
    print(manager.categories)
    id_ui_test("quicktest/test", model, manager.categories)
    
if __name__ == "__main__":
    main()