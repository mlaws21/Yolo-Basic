import torch.optim as optim
import torch
from data_reader import import_synth_data
from training_helper import TrainingMonitor, nlog_softmax_loss, minibatch_training, yolo_training
from model import build_net
from datamanager import DataPartition, DataManager
from loss import yolo_loss_func
from specs import pretrain_specs, yolo_specs, additional_yolo_specs, pretrain_small_specs
from model import Flatten, ConvLayer, ReLU
from torch.nn import Sequential

def pre_train(data_config, n_epochs=10):    
    """
    Runs a training regime for a CNN.
    
    """
    image_width = 112
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .001

    net = build_net(pretrain_small_specs)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = minibatch_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    return best_net
 
 
def get_yolo_net(pt_specs, yolo_added_spec):
    

    model = build_net(pt_specs)
    model.load_state_dict(torch.load("pretrain_small.pt"))
    # model.eval()
    # model.requires_grad_(False)
    net = Sequential()
    ctr = 0
    for layer in model:

        if isinstance(layer, Flatten):
            break
        net.add_module(str(ctr), layer)
        ctr += 1
        
        # if isinstance(layer, ConvLayer):
        #     net.add_module(str(ctr), ReLU())
        #     ctr += 1
            
        #     net.add_module(str(ctr), ConvLayer(20, 20, 1, 1, 0))
        #     ctr += 1

    # net.requires_grad_(False)
    
    newnet = build_net(yolo_added_spec, previous_num_k=32)
    for layer in newnet:
        net.add_module(str(ctr), layer)
        ctr += 1
    
    print(net)
    return net
 
def run_yolo(data_config, n_epochs=10):    
    """
    Runs a training regime for a CNN.
    
    """
    image_width = 112
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = yolo_loss_func
    learning_rate = .001
    net = get_yolo_net(pretrain_specs, additional_yolo_specs)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, _ = yolo_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)

    return best_net

def pretrain():
    best_net = pre_train('one_shape/data.json', n_epochs=20)
    torch.save(best_net.state_dict(), "pretrain_small.pt")
    
def train():
    best_net = run_yolo('large/data.json', n_epochs=100)
    torch.save(best_net.state_dict(), "yolo_small.pt")

def main():

    # pretrain()
    # print(get_yolo_net(additional_yolo_specs))
    train()

    
    


if __name__ == "__main__":
    main()