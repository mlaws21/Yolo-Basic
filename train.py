import torch.optim as optim

from data_reader import import_bounding_data, import_pretrain_data
from training_helper import TrainingMonitor, nlog_softmax_loss, minibatch_training
from model import yolo

LAST_NUM_K = 256
GRID_SIZE = 7

model_specs = [
    ("conv", (7, 32, 2)),
    ("pool", (2, -1, 2)),
    ("conv", (3, 64, 1)),
    ("pool", (2, -1, 2)),
    ("conv", (1, 32, 1)),
    ("conv", (3, 32, 1)),
    ("conv", (1, 32, 1)),
    ("conv", (3, 32, 1)),
    ("pool", (2, -1, 2)),
    ("conv", (1, 32, 1)),
    ("conv", (3, 32, 1)),
    ("conv", (1, 32, 1)),
    ("conv", (3, 32, 1)),
    ("conv", (3, 128, 1)),
    ("conv", (3, 256, 1)),
    ("flatten", ()),
    ("dense", (GRID_SIZE*GRID_SIZE*LAST_NUM_K, 4096, -1)),
    ("relu", ()),
    ("dense", (4096, 10, -1)),
    # ("relu", ()),
    
    
    
    
    
  
]

def train(train_specs, training_data, testing_data, n_epochs=10):

    """
    Runs a training regime for a CNN.
    
    """
    # train_set = DataPartition(data_config, './data', 'train')
    # test_set = DataPartition(data_config, './data', 'test')
    # manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .1
    model = yolo(train_specs)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    best_net, monitor = minibatch_training(model, training_data, testing_data, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    # classifier = Classifier(best_net, num_kernels, kernel_size, 
    #                         dense_hidden_size, manager.categories, image_width)
    return best_net

def main():
    training_data = import_pretrain_data("imagenette2/train")
    testing_data = import_pretrain_data("imagenette2/val")
    
    print(train(model_specs, training_data, testing_data))
    pass

if __name__ == "__main__":
    main()