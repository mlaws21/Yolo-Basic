# Here we are implementing the loss function describes on page 4

import torch
import torch.nn

# Tensor layout for each cell of 7x7:
# [  c01  ]
# [  c02  ]
# [  c03  ]
# [  c04  ]
# [  c05  ] 
# [  c06  ]
# [  c07  ]
# [  c08  ]
# [  c09  ]
# [  c10  ]  category probabilities
# [  c11  ]
# [  c12  ]
# [  c13  ]
# [  c14  ]
# [  c15  ]
# [  c16  ]
# [  c17  ]
# [  c18  ]
# [  c19  ]
# [  c20  ]
# something here for whether there is an object?
# [   x   ]
# [   y   ]
# [   w   ]
# [   h   ]
# [ conf? ]
# [   x   ]
# [   y   ]
# [   w   ]
# [   h   ]
# [ conf? ]

# take in 2 tensors and calculate IOU values
def IOU(a, b):
    # some code

class Loss(nn.Module):
    """
    Computes loss function according to YOLO v1.

    """
    def __init__(self, S = 7, B = 2, C = 20, l_coord = 5, l_noobj = 0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    def forward(self, predictions, target):
        # predictions should be of dimension (N, S, S, B*5+C)
        predictions.reshape_(-1, self.S, self.S, self.B * 5 + self.C)

        # get best IOU for box1 and box2
        box1_IOU = # take IOU with predictions and box 1
        box2_IOU = # take IOU with predictions and box 2
        box1_best = max(...) # find which cell is responsible for box1
        box2_best = max(...) # find which cell is responsible for box2

        # box coordinate loss (x,y)

        # box dimension loss (h,w)

        # object loss

        # no object loss

        # class loss