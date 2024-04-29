import torch.optim as optim

from data_reader import import_bounding_data, import_pretrain_data, import_synth_data
from training_helper import TrainingMonitor, nlog_softmax_loss, minibatch_training
from model import yolo

LAST_NUM_K = 256
GRID_SIZE = 7

model_specs = [
    ("conv", (7, 32, 2)),
    ("relu", ()),
    ("pool", (2, -1, 2)),
    ("conv", (3, 64, 1)),
    ("relu", ()),
    ("pool", (2, -1, 2)),
    ("conv", (1, 32, 1)),
    ("relu", ()),
    ("conv", (3, 32, 1)),
    ("relu", ()),
    ("conv", (1, 32, 1)),
    ("relu", ()),
    ("conv", (3, 32, 1)),
    ("relu", ()),
    ("pool", (2, -1, 2)),
    ("conv", (1, 32, 1)),
    ("relu", ()),
    ("conv", (3, 32, 1)),
    ("relu", ()),
    ("conv", (1, 32, 1)),
    ("relu", ()),
    ("conv", (3, 32, 1)),
    ("relu", ()),
    ("conv", (3, 128, 1)),
    ("relu", ()),
    ("conv", (3, 256, 1)),
    ("relu", ()),
    ("flatten", ()),
    ("dense", (GRID_SIZE*GRID_SIZE*LAST_NUM_K, 4096, -1)),
    ("relu", ()),
    ("norm", (4096)),
    ("dense", (4096, 2048, -1)),
    ("relu", ()),
    ("norm", (2048)),
    ("dense", (2048, 2, -1)),
    
    ("relu", ()),
    
    
    
    
    
  
]

from model import create_cnn 

def train(train_specs, training_data, testing_data, n_epochs=10):

    """
    Runs a training regime for a CNN.
    
    """
    n_epochs = 10
    num_kernels = 20
    kernel_size = 3 
    dense_hidden_size = 64
    # train_set = DataPartition(data_config, './data', 'train')
    # test_set = DataPartition(data_config, './data', 'test')
    # manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .001
    # model = yolo(train_specs)  
    model = create_cnn(num_kernels = num_kernels, kernel_size= kernel_size, 
                            output_classes=2, image_width=112,
                            dense_hidden_size=dense_hidden_size,
                            use_maxpool = False, is_grayscale=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    best_net, monitor = minibatch_training(model, training_data, testing_data, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    # classifier = Classifier(best_net, num_kernels, kernel_size, 
    #                         dense_hidden_size, manager.categories, image_width)
    return best_net

def main():
    # training_data = import_pretrain_data("imagenette2/train", speed=10)
    training_data = import_synth_data("XO/train/", speed=1, batch_size=32)
    testing_data = import_synth_data("XO/test/")
    
    print(train(model_specs, training_data, testing_data))
    pass

if __name__ == "__main__":
    main()