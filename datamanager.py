import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision import io, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import time
import sys
from torch.nn.functional import one_hot


def retrieve_image(filename, W=112):
    """
    Loads an image, resizes it to WxW pixels, and then converts it into a
    Torch tensor of shape (3, W, W). The "3" dimension corresponds to 
    the blue, green, and red channels of the input image.

    """
    try:
        img = io.read_image(filename, io.ImageReadMode.RGB)
        resize = transforms.Resize((W, W))
        grayscale = transforms.Grayscale(num_output_channels=1)
        img = grayscale(img)
        img = (resize(img).float() - 127) / 256
        
        return img
    except:
        raise Exception(f"The following image is missing or corrupted: {filename}")

class DataPartition(Dataset):

    def __init__(self, json_file, data_dir, partition, resize_width=112):  
        """
        Creates a DataPartition from a JSON configuration file.
        
        - json_file is the filename of the JSON config file.
        - data_dir is the directory where the images are stored.
        - partition is a string indicating which partition this object
                    represents, i.e. "train" or "test"
        - resize_width indicates the dimensions to which each image will
                    be resized (the images are resized to a square WxW
                    image)
        
        """
        print(f"Loading image data ({partition}):")
        with open(json_file) as f:
            data = json.load(f)
        self.data = []
        self._categories = set()
        for datum in tqdm(data):
            if datum['partition'] == partition:
                img_filename = os.path.join(data_dir, datum['filename'])
                instance = {'image': retrieve_image(img_filename, resize_width), 
                            'category': datum['label'], 
                            'filename': datum['filename'],
                            'box': datum['box'] # TODO COMMENT IF NO BOUNDING
                            }
                self.data.append(instance)
                self._categories.add(datum['label'])
        self._categories = sorted(list(self._categories))

    def __len__(self):
        """
        Returns the number of data (datums) in this DataPartition.
        
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        Converts the ith datum into the following dictionary and then
        returns it:
            
            {'image': retrieve_image(img_filename), 
             'category': datum['category'], 
             'filename': datum['filename'] }
        
        """   
        return self.data[i]
   
    def categories(self):
        """
        Returns an alphabetically sorted list of categories in this
        DataPartition. The categories are all distinct values
        associated with the key "category" in any datum.
        
        """
        return self._categories


# TODO this only works for 1 shape per image
# C is catagories, S is grid size
# assume batched input
def convert_to_output(labs_list, boxes_list, S=7, C=5):
    assert len(labs_list) == len(boxes_list)
    batch_out = []
    for i in range(len(labs_list)):
        box_string = boxes_list[i]
        label = labs_list[i]
        box_list = (box_string.split(","))
        conf, cx, cy, width, height = [float(x) for x in box_list]
        out_labs = torch.zeros(S, S, C + 5)
        row, col = int(S * cy), int(S * cx)
        x_cell, y_cell = S * cx - col, S * cy - row
        width_cell, height_cell = width * S, height * S
        out_labs[row, col, label] = 1
        out_labs[row, col, C:] = torch.tensor([conf, x_cell, y_cell, width_cell, height_cell])
        batch_out.append(out_labs)
    
    return torch.stack(batch_out)
    # labs = 
    # ground = torch.cat((oh_response, boxes), dim=1)
    
    # return torch.Tensor(f_box_list)

# def combine_box_and_labs(labs, boxes):

    

# boxes = torch.stack([convert_box(box) for box in batch[self.box_key]], dim=0)

class DataManager:
    
    def __init__(self, train_partition, test_partition, 
                 feature_key = 'image', response_key='category', box_key='box'):
        """
        Creates a DataManager from a JSON configuration file. The job
        of a DataManager is to manage the data needed to train and
        evaluate a neural network.
        
        - train_partition is the DataPartition for the training data.
        - test_partition is the DataPartition for the test data.
        - feature_key is the key associated with the feature in each
          datum of the data partitions, i.e. train_partition[i][feature_key]
          should be the ith feature tensor in the training partition.
        - response_key is the key associated with the response in each
          datum of the data partitions, i.e. train_partition[i][response_key]
          should be the ith response tensor in the training partition.
        
        """
        self.train_set = train_partition
        self.test_set = test_partition
        self.feature_key = feature_key
        self.response_key = response_key
        self.box_key = box_key
        try:
            self.categories = sorted(list(set(train_partition.categories()) |
                                          set(test_partition.categories())))
        except AttributeError:
            pass
    
    def train(self, batch_size):
        """
        Returns a torch.DataLoader for the training examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
        
        - batch_size is the number of desired training examples per batch
        
        """
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  sampler=RandomSampler(self.train_set))
        return(train_loader)
    
    def test(self):
        """
        Returns a torch.DataLoader for the test examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
                
        """
        return DataLoader(self.test_set, batch_size=4, 
                          sampler=SequentialSampler(self.test_set))

    def features_boxes_and_response(self, batch):
        """
        Converts a batch obtained from either the train or test DataLoader
        into an feature tensor and a response tensor.
        
        The feature tensor returned is just batch[self.feature_key].
        
        To build the response tensor, one starts with batch[self.response_key],
        where each element is a "response value". Each of these response
        values is then mapped to the index of that response in the sorted set of
        all possible response values. The resulting tensor should be
        a LongTensor.

        The return value of this function is:
            feature_tensor, response_tensor
        
        See the unit tests in test.py for example usages.
        
        """
        def category_index(category):
            return self.categories.index(category)
        inputs = batch[self.feature_key].float()
        labels = [category_index(c) for c 
                               in batch[self.response_key]]
        
        
        resp = convert_to_output(labels, batch[self.box_key])

        # print(resp)
        return inputs, resp
    
    def features_and_response(self, batch):
        """
        Converts a batch obtained from either the train or test DataLoader
        into an feature tensor and a response tensor.
        
        The feature tensor returned is just batch[self.feature_key].
        
        To build the response tensor, one starts with batch[self.response_key],
        where each element is a "response value". Each of these response
        values is then mapped to the index of that response in the sorted set of
        all possible response values. The resulting tensor should be
        a LongTensor.

        The return value of this function is:
            feature_tensor, response_tensor
        
        See the unit tests in test.py for example usages.
        
        """
        def category_index(category):
            return self.categories.index(category)
        inputs = batch[self.feature_key].float()
        labels = torch.Tensor([category_index(c) for c 
                               in batch[self.response_key]]).long()
     
        return inputs, labels

    def evaluate(self, classifier, partition):
        """
        Given a classifier that maps an feature tensor to a response
        tensor, this evaluates the classifier on the specified data
        partition ("train" or "test") by computing the percentage of
        correct responses.
        
        See the unit test ```test_evaluate``` in test.py for expected usage.
        
        """
        def loader(partition):
            if partition == 'train':
                return self.train(40)
            else:
                return self.test()
    
        def accuracy(outputs, labels):
            correct = 0
            total = 0
            for (output, label) in zip(outputs, labels):
                total += 1
                if label == output.argmax():
                    correct += 1
            return correct, total
        correct = 0
        total = 0       
        for data in loader(partition):
            inputs, labels = self.features_and_response(data)
            outputs = classifier(inputs) 
            correct_inc, total_inc = accuracy(outputs, labels)
            correct += correct_inc
            total += total_inc
        return correct / total
    

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

    @staticmethod
    def plot_average_and_max(monitors, description=""):
        valuelists = [monitor.training_profile_graph()[1] for monitor in monitors]
        values = [sum([valuelist[i] for valuelist in valuelists])/len(valuelists) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='average' + description)
        values = [max([valuelist[i] for valuelist in valuelists]) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='max' + description)
        plt.xlabel('epoch')
        plt.ylabel('test accuracy')
        plt.legend()

    @staticmethod
    def plot_average(monitors, description=""):
        valuelists = [monitor.training_profile_graph()[1] for monitor in monitors]
        values = [sum([valuelist[i] for valuelist in valuelists])/len(valuelists) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='average' + description)
        plt.xlabel('epoch')
        plt.ylabel('test accuracy')
        plt.legend()



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