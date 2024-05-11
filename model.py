import torch
from torch.nn import Unfold, Parameter, Module, init, Sequential
from convolve import convolve
DEVICE='cpu'

class ConvLayer(Module):
    """A class for a convolutional neural network (CNN)."""
    
    def __init__(self, input_channels, num_kernels, 
                 kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.stride = stride
        self.kappa = Parameter(torch.empty(num_kernels, input_channels, 
                                           kernel_size, kernel_size, device=DEVICE))
        self.offset = Parameter(torch.empty(num_kernels, 1, 1, device=DEVICE))
        self.padding = padding
        
        # initializes the parameters
        init.kaiming_uniform_(self.kappa)
        init.zeros_(self.offset)
    
    def forward(self, x):   
        
        out = self.offset + convolve(self.kappa, x, 
                                      self.stride, self.padding)
        
        return out
        
        
class MaxPool(Module):
    """A class for a Max pooling layer of a network. Implementation taken from
    Matt's Kernels lab    
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
    """A class to flatten a CNN (or any non flat layer) into a MxN where M is batch size and N
    is the resulting size after flattening the old layer"""

    def __init__(self): 
        super().__init__()
    
    def forward(self, x):
         
        return torch.flatten(x, start_dim=1)
    
    
class Leaky_ReLU(torch.nn.Module):
    """A class that implements the leaky ReLU activation / non-lnearity"""
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        lrlu = torch.nn.LeakyReLU(0.1)
        return lrlu(x)
    
class Dropout(torch.nn.Module):
    """A class that implements the dropout"""
    
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        drop = torch.nn.Dropout(p=self.p)
        return drop(x)
    

class Dense(torch.nn.Module):
    """A class that implements a dense layer of a neural net"""
    
    
    def __init__(self, input_size, output_size):

        super(Dense, self).__init__()
        self.weight = Parameter(torch.empty(output_size, 1+input_size, device=DEVICE))
        init.kaiming_uniform_(self.weight)
        
        
    def forward(self, x):

        x_and_offset = torch.cat([torch.ones((x.shape[0],1), device=DEVICE),x], dim=1)
        return torch.matmul(self.weight, x_and_offset.t()).t() 
    
def build_net(specs, is_greyscale=True, previous_num_k=None):
    """A function to build a model from a spec list --- a list of specifications defining each
    layer of the model
    
    Spec Format:
    
    CNN (conv): (kernel size, num kernels, stride)
    MaxPool (pool): (kernel size, -1, stride)
    Flatten (flatten): None
    Dense (dense): None
    Leaky ReLU (l_relu): None
    Dropout (drop): None
    
    Args:
        specs (list): list of all specifications following the spec format
        is_greyscale (bool, optional): is the image read in using greyscale. Defaults to True.
        previous_num_k (_type_, optional): Number of kernels in the previous layer. Only relavent
        if we are starting from the output of a convolutionallayer not from an image. Defaults to None.

    Returns:
        torch model
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
            
        elif layer_type == "l_relu":
            model.add_module(layer_name, Leaky_ReLU())
        
        elif layer_type == "drop":
            model.add_module(layer_name, Dropout(layer_spec))
        else:
            print("WARNING: Layer type not recognized")
            
        layer_num += 1
        
    return model


def build_yolo_net(pt_specs, yolo_added_spec, pt_file=None):
    """Build the yolo model

    Args:
        pt_specs (list): Specification list that the pretraining data used
        yolo_added_spec (list): Specification list of additional layers
        pt_file (str, optional): .pt file if you want to import pretrained weights. 
        Defaults to None.

    Returns:
        torch model
    """

    model = build_net(pt_specs)
    if pt_file is not None:
        model.load_state_dict(torch.load(pt_file))

    net = Sequential()
    ctr = 0
    for layer in model:

        if isinstance(layer, Flatten):
            break
        net.add_module(str(ctr), layer)
        ctr += 1
    
    newnet = build_net(yolo_added_spec, previous_num_k=32)
    for layer in newnet:
        net.add_module(str(ctr), layer)
        ctr += 1
    
    return net
