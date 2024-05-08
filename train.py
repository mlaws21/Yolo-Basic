import torch
import torch.optim as optim
from training_helper import nlog_softmax_loss, minibatch_training, yolo_training
from model import build_net, build_yolo_net
from datamanager import DataPartition, DataManager
from loss import yolo_loss_func
from specs import pretrain_specs, additional_yolo_specs

import sys


def pretrain_helper(data_config, inter_name, n_epochs=10):    
    """
    Runs a training regime for a CNN.
    
    """
    image_width = 112
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .001

    net = build_net(pretrain_specs)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = minibatch_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss, inter_name=inter_name)
    return best_net
 
def run_yolo(data_config, inter_name, n_epochs=10):    
    """
    Runs a training regime for a CNN.
    
    """
    image_width = 112
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = yolo_loss_func
    learning_rate = .001
    net = build_yolo_net(pretrain_specs, additional_yolo_specs)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = yolo_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss, inter_name=inter_name)

    return best_net

def pretrain(json, save_name, inter_name):
    best_net = pretrain_helper(json, inter_name=inter_name, n_epochs=20)
    torch.save(best_net.state_dict(), save_name)
    
def train(json, save_name, inter_name):
    best_net = run_yolo(json, inter_name=inter_name, n_epochs=100)
    torch.save(best_net.state_dict(), save_name)

def main():

    pretrain_or_train = sys.argv[1]
    json = sys.argv[2]
    save_name = sys.argv[3]
    inter_name = sys.argv[4]
    
    
    
    if pretrain_or_train == "p":
        pretrain(json, save_name, inter_name)
    elif pretrain_or_train == "t":
        train(json, save_name, inter_name)
    else:
        print("Usage: python train [p/t] [data.json file] [name to save model as] [intermediate name]")

if __name__ == "__main__":
    main()