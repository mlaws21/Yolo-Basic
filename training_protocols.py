
import torch
from tqdm import tqdm
from copy import deepcopy
from datamanager import TrainingMonitor

DEVICE='cpu'

def minibatch_training(net, manager, batch_size, 
                       n_epochs, optimizer, loss, inter_name):
    """
    Arguments
    - net: the Module you want to train.
    - manager: the DataManager
    - batch_size: the desired size of each minibatch
    - n_epochs: the desired number of epochs for minibatch training
    - optimizer: the desired optimization algorithm to use (should be an 
                 instance of torch.optim.Optimizer)
    - loss: the loss function to optimize
    - inter_name: name to save intermediate layers as
    
    """
    monitor = TrainingMonitor()
    train_loader = manager.train(batch_size)
    best_accuracy = float('-inf')
    best_net = None
    monitor.start(len(train_loader))
    for epoch in range(n_epochs):
        monitor.start_epoch(epoch)
        net.train() # puts the module in "training mode", e.g. ensures
                    # requires_grad is on for the parameters
        for i, data in tqdm(enumerate(train_loader, 0)):
            features, response = manager.features_and_response(data)
            optimizer.zero_grad()
            output = net(features)
            batch_loss = loss(output, response)
            batch_loss.backward()
            optimizer.step()
            monitor.report_batch_loss(epoch, i, batch_loss.data.item())            
        net.eval() # puts the module in "evaluation mode", e.g. ensures
                   # requires_grad is off for the parameters
        dev_accuracy = manager.evaluate(net, "test")
        monitor.report_accuracies(epoch, None, dev_accuracy)
        if dev_accuracy >= best_accuracy:
            best_net = deepcopy(net)     
            best_accuracy = dev_accuracy
            if inter_name is not None:
                torch.save(best_net.state_dict(), inter_name)
            
    monitor.stop()
    return best_net, monitor

def yolo_training(net, manager, batch_size, 
                       n_epochs, optimizer, loss, inter_name):
    """
    Trains a neural network using the training partition of the 
    provided DataManager.
    
    Arguments
    - net: the Module you want to train.
    - manager: the DataManager
    - batch_size: the desired size of each minibatch
    - n_epochs: the desired number of epochs for minibatch training
    - optimizer: the desired optimization algorithm to use (should be an 
                 instance of torch.optim.Optimizer)
    - loss: the loss function to optimize
    - inter_name: name to save intermediate layers as
    
    """
    monitor = TrainingMonitor()
    train_loader = manager.train(batch_size)
    best_accuracy = float('-inf')
    best_net = None
    monitor.start(len(train_loader))
    for epoch in range(n_epochs):
        monitor.start_epoch(epoch)
        net.train() # puts the module in "training mode", e.g. ensures
                    # requires_grad is on for the parameters
        for i, data in tqdm(enumerate(train_loader, 0)):
            features, response = manager.features_boxes_and_response(data)
            optimizer.zero_grad()
            output = net(features)
            batch_loss = loss(output, response)
            batch_loss.backward()
            optimizer.step()
            monitor.report_batch_loss(epoch, i, batch_loss.data.item())            
        net.eval() # puts the module in "evaluation mode", e.g. ensures
                   # requires_grad is off for the parameters
        dev_accuracy = manager.yolo_evaluate(net, "test")
        monitor.report_accuracies(epoch, None, dev_accuracy)
        if dev_accuracy >= best_accuracy:
            best_net = deepcopy(net)     
            best_accuracy = dev_accuracy
            if inter_name is not None:
                torch.save(best_net.state_dict(), inter_name)
    monitor.stop()
    return best_net, monitor
