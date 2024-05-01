import torch.optim as optim

from data_reader import import_synth_data
from training_helper import TrainingMonitor, nlog_softmax_loss, minibatch_training, yolo_training
from model import build_net
from datamanager import DataPartition, DataManager
from loss import yolo_loss_func


LAST_NUM_K = 256
GRID_SIZE = 7

# model_specs = [
#     ("conv", (7, 32, 2)),
#     ("relu", ()),
#     ("pool", (2, -1, 2)),
#     ("conv", (3, 64, 1)),
#     ("relu", ()),
#     ("pool", (2, -1, 2)),
#     ("conv", (1, 32, 1)),
#     ("relu", ()),
#     ("conv", (3, 32, 1)),
#     ("relu", ()),
#     ("conv", (1, 32, 1)),
#     ("relu", ()),
#     ("conv", (3, 32, 1)),
#     ("relu", ()),
#     ("pool", (2, -1, 2)),
#     ("conv", (1, 32, 1)),
#     ("relu", ()),
#     ("conv", (3, 32, 1)),
#     ("relu", ()),
#     ("conv", (1, 32, 1)),
#     ("relu", ()),
#     ("conv", (3, 32, 1)),
#     ("relu", ()),
#     ("conv", (3, 128, 1)),
#     ("relu", ()),
#     ("conv", (3, 256, 1)),
#     ("relu", ()),
#     ("flatten", ()),
#     ("dense", (GRID_SIZE*GRID_SIZE*LAST_NUM_K, 4096, -1)),
#     ("relu", ()),
#     ("norm", (4096)),
#     ("dense", (4096, 2048, -1)),
#     ("relu", ()),
#     ("norm", (2048)),
#     ("dense", (2048, 2, -1)),
    
#     ("relu", ()),

# ]


# 
    # model = create_cnn(num_kernels = num_kernels, kernel_size= kernel_size, 
    #                         output_classes=2, image_width=112,
    #                         dense_hidden_size=dense_hidden_size,
    #                         use_maxpool = False, is_grayscale=True)
    
from model import ConvLayer, ReLU, Flatten, MaxPool, Dense
from torch.nn import Sequential

def create_cnn(num_kernels, kernel_size, 
               output_classes, dense_hidden_size,
               image_width, is_grayscale=False,
               use_maxpool=True):
    """
    Builds a CNN with two convolutional layers and two feedforward layers.
    
    Maxpool is added by default, but can be disabled.

    This function is already completed.
    
    """  
    padding = kernel_size//2
    output_width = image_width
    if use_maxpool:
        output_width = output_width // 16
    model = Sequential()
    if is_grayscale:
        num_input_channels = 1
    else:
        num_input_channels = 3
    model.add_module("conv1", ConvLayer(num_input_channels, num_kernels,
                                   kernel_size=kernel_size, 
                                   stride=1, padding=padding))
    model.add_module("relu1", ReLU())
    if use_maxpool:
        model.add_module("pool1", MaxPool(kernel_size=4, stride=4, padding=0))
    model.add_module("conv2", ConvLayer(num_kernels, num_kernels,
                                              kernel_size=kernel_size, 
                                              stride=1, padding=padding))
    model.add_module("relu2", ReLU())
    if use_maxpool:
        model.add_module("pool2", MaxPool(kernel_size=4, stride=4, padding=0))
    model.add_module("flatten", Flatten())
    model.add_module("dense1", Dense(num_kernels * output_width**2, 
                                     dense_hidden_size))
    model.add_module("relu3", ReLU())
    model.add_module("dense2", Dense(dense_hidden_size, output_classes))
    return model


#   net = create_cnn(num_kernels = num_kernels, kernel_size= kernel_size, 
#                                  output_classes=5, image_width=image_width,
#                                  dense_hidden_size=dense_hidden_size,
#                                  use_maxpool = use_maxpool, is_grayscale=True)
  
    #   n_epochs = 10,
    # num_kernels = 20, 
    # kernel_size = 3, 
    # dense_hidden_size = 64)
    
pretrain_specs = [
    ("conv", (3, 20, 1)),
    ("relu", ()),
    # ("conv", (3, 20, 1)),
    # ("relu", ()),
    ("pool", (4, -1, 4)),
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("pool", (4, -1, 4)),
    ("flatten", ()),
    ("dense", (20 * (112 // 16)**2, 64)),
    ("relu", ()),
    ("dense", (64, 5)), 
    
]

yolo_specs = [
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("pool", (4, -1, 4)),
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("pool", (4, -1, 4)),
    ("flatten", ()),
    ("dense", (20 * (224 // 16)**2, 2048)),
    
    ("relu", ()),
    ("dense", (2048, 7 * 7 * 10)),
    ("relu", ()),
    
    # ("dense", (64, 5)),
]
# def train(train_specs, training_data, testing_data, n_epochs=10):

#     """
#     Runs a training regime for a CNN.
    
#     """
#     n_epochs = 10
#     loss = nlog_softmax_loss
#     learning_rate = .001
#     # model = yolo(train_specs)
#     n_epochs = 8
#     num_kernels = 20
#     kernel_size = 7
#     dense_hidden_size = 64
#     model = create_cnn(num_kernels = num_kernels, kernel_size= kernel_size, 
#                             output_classes=5, image_width=112,
#                             dense_hidden_size=dense_hidden_size,
#                             use_maxpool = False, is_grayscale=True)

#     print(model)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
#     best_net, monitor = minibatch_training(model, training_data, testing_data, n_epochs=n_epochs, 
#                                            optimizer=optimizer, loss=loss)
#     return best_net


def pre_train(data_config, n_epochs, num_kernels, 
        kernel_size, dense_hidden_size, 
        use_maxpool=True):    
    """
    Runs a training regime for a CNN.
    
    """
    train_set = DataPartition(data_config, './', 'train')
    test_set = DataPartition(data_config, './', 'test')
    manager = DataManager(train_set, test_set)
    loss = nlog_softmax_loss
    learning_rate = .001
    image_width = 112
    # net = create_cnn(num_kernels = num_kernels, kernel_size= kernel_size, 
    #                              output_classes=5, image_width=image_width,
    #                              dense_hidden_size=dense_hidden_size,
    #                              use_maxpool = use_maxpool, is_grayscale=True)
    net = build_net(pretrain_specs)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, monitor = minibatch_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    # classifier = Classifier(best_net, num_kernels, kernel_size, 
    #                         dense_hidden_size, manager.categories, image_width)
    return monitor
 
 
def run_yolo(data_config, n_epochs, num_kernels, 
        kernel_size, dense_hidden_size, 
        use_maxpool=True):    
    """
    Runs a training regime for a CNN.
    
    """
    image_width = 224
    
    train_set = DataPartition(data_config, './', 'train', resize_width=image_width)
    test_set = DataPartition(data_config, './', 'test', resize_width=image_width)
    manager = DataManager(train_set, test_set)
    loss = yolo_loss_func
    learning_rate = .001
    net = build_net(yolo_specs)
    # net = yolo(model_specs)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, monitor = yolo_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    # classifier = Classifier(best_net, num_kernels, kernel_size, 
    #                         dense_hidden_size, manager.categories, image_width)
    return monitor

def main():

    # training_data = import_synth_data("shapes/train/", speed=1, batch_size=32)
    # testing_data = import_synth_data("shapes/test/")
    
    # print(train(model_specs, training_data, testing_data))
    
    classifier = run_yolo('one_shape/data.json',
    n_epochs = 10,
    num_kernels = 20, 
    kernel_size = 3, 
    dense_hidden_size = 64)


if __name__ == "__main__":
    main()