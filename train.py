import torch
import torch.optim as optim
from training_protocols import minibatch_training, yolo_training
from model import build_net, build_yolo_net
from datamanager import DataPartition, DataManager
from loss import yolo_loss, nlog_softmax_loss
from specs import pretrain_specs, additional_yolo_specs
import sys


def pretrain_helper(data_config, inter_name=None, n_epochs=10, image_width=112, batch_size=32):    
    """Runs the pretraining regime for our image recognition task

    Args:
        data_config: data.json file to extract the data from 
        inter_name: (optional): name to save intermediate weights as while training,
        if none it doesn't save. Defaults to None.
        n_epochs (int, optional): Number of epochs to run for. Defaults to 10.
        image_width (int, optional): width and height of input image. Defaults to 112.
        batch_size (int, optional): size of minibatch. Defaults to 32.
        

    Returns:
        pretrained model
    """
    
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .001

    net = build_net(pretrain_specs)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = minibatch_training(net, manager, 
                                           batch_size=batch_size, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss, inter_name=inter_name)
    return best_net
 
def run_yolo(data_config, inter_name=None, pt_file=None, n_epochs=10, image_width=112, batch_size=32):    
    """Runs the yolo training regime

    Args:
        data_config: data.json file to extract the data from 
        inter_name: (optional): name to save intermediate weights as while training,
        if none it doesn't save. Defaults to None.
        n_epochs (int, optional): Number of epochs to run for. Defaults to 10.
        image_width (int, optional): width and height of input image. Defaults to 112.
        batch_size (int, optional): size of minibatch. Defaults to 32.

    Returns:
        yolo model
    """
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = yolo_loss
    learning_rate = .001
    net = build_yolo_net(pretrain_specs, additional_yolo_specs, pt_file=pt_file)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = yolo_training(net, manager, 
                                           batch_size=batch_size, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss, inter_name=inter_name)

    return best_net

def pretrain(json, save_name, inter_name):
    """Simple call to run and save the pretraining"""
    best_net = pretrain_helper(json, inter_name=inter_name, n_epochs=20)
    torch.save(best_net.state_dict(), save_name)
    
def train(json, save_name, inter_name, pt_file):
    """Simple call to run and save the training"""
    best_net = run_yolo(json, inter_name=inter_name, pt_file=pt_file, n_epochs=100)
    torch.save(best_net.state_dict(), save_name)

def main():

    pretrain_or_train = sys.argv[1]
    json = sys.argv[2]
    save_name = sys.argv[3]
    inter_name = sys.argv[4]
    pt_file = sys.argv[5]


    
    if pretrain_or_train == "p":
        pretrain(json, save_name, inter_name)
    elif pretrain_or_train == "t":
        train(json, save_name, inter_name, pt_file)
    else:
        print("Usage: python train.py [p/t] [data.json file] [name to save model as] [intermediate name] [pretrain file]")

if __name__ == "__main__":
    main()