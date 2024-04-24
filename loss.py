# Here we are implementing the loss function describes on page 4

import torch
import torch.nn

# Tensor layout for each cell of 7x7:
# 
#
#        TRAIN
#   0  [   0   ]
#   1  [   0   ]
#   2  [   0   ]
#   3  [   0   ]
#   4  [   0   ] 
#   5  [   0   ]
#   6  [   0   ]
#   7  [   0   ]
#   8  [   0   ]
#   9  [   0   ]  one hot encoding of true value
#  10  [   0   ]
#  11  [   0   ]
#  12  [   1   ]
#  13  [   0   ]
#  14  [   0   ]
#  15  [   0   ]
#  16  [   0   ]
#  17  [   0   ]
#  18  [   0   ]
#  19  [   0   ]
#  20  [   1   ]  object exists here (indicator)
#  21  [   x   ]  true x
#  22  [   y   ]  true y
#  23  [   w   ]  etc.
#  24  [   h   ]
#
#
#         OUT
#   0  [  c01  ]
#   1  [  c02  ]
#   2  [  c03  ]
#   3  [  c04  ]
#   4  [  c05  ] 
#   5  [  c06  ]
#   6  [  c07  ]
#   7  [  c08  ]
#   8  [  c09  ]
#   9  [  c10  ]  category probs p̂_i(c)
#  10  [  c11  ]
#  11  [  c12  ]
#  12  [  c13  ]
#  13  [  c14  ]
#  14  [  c15  ]
#  15  [  c16  ]
#  16  [  c17  ]
#  17  [  c18  ]
#  18  [  c19  ]
#  19  [  c20  ]
#  20  [ conf. ]  Ĉ for box 1
#  21  [   x   ]  x for box 1
#  22  [   y   ]  etc.
#  23  [   w   ]
#  24  [   h   ]
#  25  [ conf. ]  Ĉ for box 2
#  26  [   x   ]  x for box 2
#  27  [   y   ]  etc.
#  28  [   w   ]
#  29  [   h   ]

class YoloLoss(torch.nn.Module):
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

    def IOU(self, box1, box2):
        # take x y w h, calculate IOU
        return -1
    
    def forward(self, predictions, target):
        # predictions should be of dimension (N, S, S, B*5+C)
        predictions.reshape_(-1, self.S, self.S, self.B * 5 + self.C)

        # get best IOU for box1 and box2
        box1_IOU = # take IOU with predictions and box 1
        box2_IOU = # take IOU with predictions and box 2
        box1_best = max(...) # find which cell is responsible for box1
        box2_best = max(...) # find which cell is responsible for box2

        # box coordinate loss (x,y)
        coordinate_loss = 0

        # box dimension loss (h,w)
        dimension_loss = 0

        # object loss
        object_loss = 0

        # no object loss
        no_object_loss = 0

        # class loss
        class_loss = 0

        # make sure to include lambdas
        overall_loss = coordinate_loss + dimension_loss + object_loss + no_object_loss + class_loss
        return overall_loss