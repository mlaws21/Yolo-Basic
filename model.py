import torch
from torch.nn import Unfold, Parameter, Module, init, Sequential, LayerNorm
from cnn_helper import convolve

class ConvLayer(Module):
    """A convolutional layer for images.
    
    This class is already completed.    
    """    
    
    def __init__(self, input_channels, num_kernels, 
                 kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.stride = stride
        self.kappa = Parameter(torch.empty(num_kernels, input_channels, 
                                           kernel_size, kernel_size))
        self.offset = Parameter(torch.empty(num_kernels, 1, 1))
        self.padding = padding
        # initializes the parameters
        init.kaiming_uniform_(self.kappa)
        init.zeros_(self.offset)
    
    def forward(self, x):
        """This will only work after you've implemented convolve (above)."""
        out = self.offset + convolve(self.kappa, x, 
                                      self.stride, self.padding)
        
        return out
        
        
class MaxPool(Module):
    """
    A MaxPool layer for images. See unit tests for expected input
    and output.
    
    """
    def __init__(self, kernel_size, stride, padding): 
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        

        num_images, num_channels, _, width  = x.size()

        unfolder = Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        unfolded = unfolder(x)

        reshaped_unfold = torch.reshape(unfolded, (num_images, num_channels, -1, self.kernel_size * self.kernel_size))
        
        max_pooled, _ = torch.max(reshaped_unfold, dim=3)
        
        padded_img_size = width + self.padding*2
        
        pool_size = ((padded_img_size - self.kernel_size) // self.stride) + 1
        
        reshaped_pool = torch.reshape(max_pooled, (num_images, num_channels, pool_size, pool_size))
        
        return reshaped_pool
    
    
class Flatten(Module):
    """
    Flattens a tensor into a matrix. The first dimension of the input
    tensor and the output tensor should agree.
    
    For instance, a 3x4x5x2 tensor would be flattened into a 3x40 matrix.
    
    See the unit tests for example input and output.
    
    """   
    def __init__(self): 
        super().__init__()
    
    def forward(self, x):
        # print(torch.flatten(x, start_dim=1))
        return torch.flatten(x, start_dim=1)
    
    
class ReLU(torch.nn.Module):
    """
    Implements a rectified linear unit. The ```forward``` method takes
    a torch.Tensor as its argument, and returns a torch.Tensor of the
    same shape, where all negative entries are replaced by 0. 
    For instance:

        t = torch.tensor([[-3., 0., 3.2], 
                          [2., -3.5, 1.]])
        relu = ReLU()
        relu.forward(t)

    should return the tensor:
        
        torch.tensor([[0., 0., 3.2], 
                      [2., 0., 1.]])
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(min=0) 
    

class Dense(torch.nn.Module):
    
    def __init__(self, input_size, output_size):
        """
        A Module that multiplies an feature matrix (with input_size
        feature vectors) with a weight vector to produce output_size
        output vectors per feature vector. See the unit tests in test.py
        for specific examples.
    
        - An offset variable is added to each feature vector. e.g. Suppose 
          the feature tensor (a batch of feature vectors) is 
          [[6.2, 127.0], [5.4, 129.0]]. The forward method will first
          prepend offset variables to each feature vector, i.e. it 
          becomes [[1.0, 6.2, 127.0], [1.0, 5.4, 129.0]]. Then, it takes the
          dot product of the weight vector with each of the new feature
          vectors.
        
        """
        super(Dense, self).__init__()
        self.weight = Parameter(torch.empty(output_size, 1+input_size))
        init.kaiming_uniform_(self.weight)
        
    def unit_weights(self):
        """Resets all weights to 1. For testing purposes."""
        init.ones_(self.weight)
        
    def forward(self, x):
        """
        Computes the linear combination of the weight vector with
        each of the feature vectors in the feature matrix x.
        
        Note that this function will add an offset variable to each
        feature vector. e.g. Suppose the feature matrix (a batch of
        feature vectors) is [[6.2, 127.0], [5.4, 129.0]]. The forward
        method will first prepend offset variables to each feature vector,
        i.e. the feature matrix becomes 
        [[1.0, 6.2, 127.0], [1.0, 5.4, 129.0]]. Then, it takes the dot 
        product of the weight vector with each of the new feature vectors.
        
        """        
        x2 = torch.cat([torch.ones(x.shape[0],1),x], dim=1)
        # print(torch.matmul(self.weight,x2.t()).t())
        # print(torch.softmax(torch.matmul(self.weight,x2.t()).t(), dim=1))

        return torch.matmul(self.weight,x2.t()).t() 
    
# format of input list of (layer, (conv, kernels, stride))

# model_specs = [
#     ("conv", (7, 32, 2)),
#     ("pool", (2, -1, 2)),
#     ("conv", (3, 64, 1)),
#     ("pool", (2, -1, 2)),
#     ("conv", (1, 32, 1)),
#     ("conv", (3, 32, 1)),
#     ("pool", (2, -1, 2)),
  
# ]

def build_net(specs, is_greyscale=True, previous_num_k=None):
    """
    Builds a CNN with two convolutional layers and two feedforward layers.
    
    Maxpool is added by default, but can be disabled.

    This function is already completed.
    
    """  
    
    model = Sequential()
    # input_channels = 1 # bc greyscale
    layer_num = 0
    if previous_num_k is None:
        previous_num_k = 1 if is_greyscale else 3
    for layer_type, layer_spec in specs:
        layer_name = layer_type + str(layer_num)
        if layer_type == "conv":
            k_size, num_k, stride = layer_spec
            padding = k_size // 2
            
            model.add_module(layer_name, 
                             ConvLayer(previous_num_k, num_k, k_size, stride, padding))
            previous_num_k = num_k
            
        elif layer_type == "pool":
            k_size, _, stride = layer_spec
            
            model.add_module(layer_name, MaxPool(kernel_size=k_size, stride=stride, padding=0))
            
        elif layer_type == "flatten":
            model.add_module(layer_name, Flatten())
        
        elif layer_type == "dense":
            input_size, output_size = layer_spec
            
            model.add_module(layer_name, Dense(input_size, output_size))
            
        elif layer_type == "relu":
            model.add_module(layer_name, ReLU())
            
        elif layer_type == "norm":
            model.add_module(layer_name, LayerNorm(layer_spec))
        else:
            pass
            
        layer_num += 1
        
        
    # model.add_module("relu1", ReLU())
    # if use_maxpool:
    #     model.add_module("pool1", MaxPool(kernel_size=4, stride=4, padding=0))
    # model.add_module("conv2", ConvLayer(num_kernels, num_kernels,
    #                                           kernel_size=kernel_size, 
    #                                           stride=1, padding=padding))
    # model.add_module("relu2", ReLU())
    # if use_maxpool:
    #     model.add_module("pool2", MaxPool(kernel_size=4, stride=4, padding=0))
    # model.add_module("flatten", Flatten())
    # model.add_module("dense1", Dense(num_kernels * output_width**2, 
    #                                  dense_hidden_size))
    # model.add_module("relu3", ReLU())
    # model.add_module("dense2", Dense(dense_hidden_size, output_classes))
    return model


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