import time
import sys
import torch
from tqdm import tqdm

def nlog_softmax_loss(X, y):
    """
    A loss function based on softmax, described in colonels2.ipynb. 
    X is the (batch) output of the neural network, while y is a response 
    vector.
    
    See the unit tests in test.py for expected functionality.
    
    """    
    smax = torch.softmax(X, dim=1)
    # print(y.unsqueeze(1).shape)
    correct_probs = torch.gather(smax, 1, y)
    
    nlog_probs = -torch.log(correct_probs)
    return torch.mean(nlog_probs) 

def minibatch_training(model, data_loader, 
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

    best_accuracy = float('-inf')
    best_net = None
    monitor.start(len(data_loader))
    for epoch in range(n_epochs):
        monitor.start_epoch(epoch)
        model.train() # puts the module in "training mode", e.g. ensures
                    # requires_grad is on for the parameters
        for i, data in tqdm(enumerate(data_loader, 0)):
            features, response = data
            optimizer.zero_grad()
            output = model(features)
            
            # print(output.shape, response.shape)
            # print(output)
            # print(response)
            
            batch_loss = loss(output, response)
            batch_loss.backward()
            optimizer.step()
            monitor.report_batch_loss(epoch, i, batch_loss.data.item())            
        # model.eval() # puts the module in "evaluation mode", e.g. ensures
        #            # requires_grad is off for the parameters
        # dev_accuracy = manager.evaluate(net, "test")
        # monitor.report_accuracies(epoch, None, dev_accuracy)
        # if dev_accuracy >= best_accuracy:
        #     best_net = deepcopy(net)     
        #     best_accuracy = dev_accuracy
    monitor.stop()
    return best_net, monitor

class TrainingMonitor:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def start(self, n_batches):
        print("=== STARTING TRAINING ===")
        self.training_start_time = time.time()
        self.n_batches = n_batches
        self.print_every = n_batches // 10   
        self.train_profile = []

    def start_epoch(self, epoch):
        sys.stdout.write("Epoch {}".format(epoch))
        self.running_loss = 0.0
        self.start_time = time.time()
        self.total_train_loss = 0

    def report_batch_loss(self, epoch, batch, loss):
        self.total_train_loss += loss
        if self.verbose:
            self.running_loss += loss
            if (batch + 1) % (self.print_every + 1) == 0:               
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (batch+1) / self.n_batches), 
                        self.running_loss / self.print_every, time.time() - self.start_time))
                self.running_loss = 0.0
                self.start_time = time.time()                    
            
    def report_accuracies(self, epoch, train_acc, dev_acc):
        self.train_profile.append((train_acc, dev_acc))
        epoch_time = time.time() - self.start_time
        if train_acc is not None:
            print("Train accuracy = {:.2f}".format(train_acc))
        if dev_acc is not None:
            print("[{:.2f}sec] train loss = {:.2f}; test accuracy = {:.2f} ".format(
                    epoch_time, self.total_train_loss, dev_acc))

    def training_profile_graph(self):
        return [[x[0] for x in self.train_profile], 
                [x[1] for x in self.train_profile]]

    def stop(self):
        print("Training finished, took {:.2f}s".format(
                time.time() - self.training_start_time))



