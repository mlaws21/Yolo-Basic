
import torch
from tqdm import tqdm
from copy import deepcopy
from datamanager import TrainingMonitor
# from eval import evaluate


# def nlog_softmax_loss(X, y):
#     """
#     A loss function based on softmax, described in colonels2.ipynb. 
#     X is the (batch) output of the neural network, while y is a response 
#     vector.
    
#     See the unit tests in test.py for expected functionality.
    
#     """    
#     # print("X", X)
#     smax = torch.softmax(X, dim=1)
#     # print(smax.shape)
#     # print(smax)
#     # print(y.unsqueeze(1).shape, smax.shape)
#     correct_probs = torch.gather(smax, 1, y.unsqueeze(1))
#     # print(correct_probs)
    
    
#     nlog_probs = -torch.log(correct_probs) #IS THIS OK?
#     return torch.mean(nlog_probs) 



def nlog_softmax_loss(X, y):
    """
    A loss function based on softmax, described in colonels2.ipynb. 
    X is the (batch) output of the neural network, while y is a response 
    vector.
    
    See the unit tests in test.py for expected functionality.
    
    """    
    smax = torch.softmax(X, dim=1)
    correct_probs = torch.gather(smax, 1, y.unsqueeze(1))
    nlog_probs = -torch.log(correct_probs)
    return torch.mean(nlog_probs) 

def minibatch_training(net, manager, batch_size, 
                       n_epochs, optimizer, loss):
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
    monitor.stop()
    return best_net, monitor

def yolo_training(net, manager, batch_size, 
                       n_epochs, optimizer, loss):
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
            # print(response)
            # oh_response = one_hot(response)
            # ground = torch.cat((oh_response, boxes), dim=1)
            
            # print(ground )
            # print(output.shape, response.shape)
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
    monitor.stop()
    return best_net, monitor
