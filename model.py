import torch
from torch.nn import Unfold, Parameter, Module, init, Sequential, LayerNorm
from convolve import convolve
DEVICE='cpu'

class ConvLayer(Module):
    
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

    def __init__(self): 
        super().__init__()
    
    def forward(self, x):
         
        return torch.flatten(x, start_dim=1)
    
    
class ReLU(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return x.clamp(min=0) 
        lrlu = torch.nn.LeakyReLU(0.1)
        return lrlu(x)
    

class Dense(torch.nn.Module):
    
    def __init__(self, input_size, output_size):

        super(Dense, self).__init__()
        self.weight = Parameter(torch.empty(output_size, 1+input_size, device=DEVICE))
        init.kaiming_uniform_(self.weight)
        
        
    def forward(self, x):

        x_and_offset = torch.cat([torch.ones((x.shape[0],1), device=DEVICE),x], dim=1)
        return torch.matmul(self.weight, x_and_offset.t()).t() 
    
def build_net(specs, is_greyscale=True, previous_num_k=None):

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
            # model.add_module(layer_name, Conv2d(previous_num_k, num_k, k_size, stride=stride, padding=padding))
            previous_num_k = num_k
            
        elif layer_type == "pool":
            k_size, _, stride = layer_spec
            
            model.add_module(layer_name, MaxPool(kernel_size=k_size, stride=stride, padding=0))
            # model.add_module(layer_name, MaxPool2d(k_size, stride=stride))
            
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
        
    return model


def build_yolo_net(pt_specs, yolo_added_spec, pt_file=None):
    

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
    
    # print(net)
    return net



def resume_train(pt_specs, yolo_added_spec):
    

    model = build_net(pt_specs)
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
    
    net.load_state_dict(torch.load("models/yolo.pt"))
    return net