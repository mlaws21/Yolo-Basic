# Here we are implementing the loss function describes on page 4
# Code is, at different parts, adapted from, inspired by, and borrowed from 
# https://github.com/aladdinpersson/

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
#  12  [   1   ]  in this example the category is c13 (12 + 1)
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

def intersection_over_union(predictions, targets, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        predictions (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        targets (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = predictions[..., 0:1] - predictions[..., 2:3] / 2
        box1_y1 = predictions[..., 1:2] - predictions[..., 3:4] / 2
        box1_x2 = predictions[..., 0:1] + predictions[..., 2:3] / 2
        box1_y2 = predictions[..., 1:2] + predictions[..., 3:4] / 2
        box2_x1 = targets[..., 0:1] - targets[..., 2:3] / 2
        box2_y1 = targets[..., 1:2] - targets[..., 3:4] / 2
        box2_x2 = targets[..., 0:1] + targets[..., 2:3] / 2
        box2_y2 = targets[..., 1:2] + targets[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = predictions[..., 0:1]
        box1_y1 = predictions[..., 1:2]
        box1_x2 = predictions[..., 2:3]
        box1_y2 = predictions[..., 3:4]  # (N, 1)
        box2_x1 = targets[..., 0:1]
        box2_y1 = targets[..., 1:2]
        box2_x2 = targets[..., 2:3]
        box2_y2 = targets[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class YoloLoss(torch.nn.Module):
    """
    Computes loss function according to YOLO v1.

    """
    def __init__(self, S = 7, B = 2, C = 20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_coord = 5
        self.l_noobj = 0.5
        self.sse = torch.nn.MSELoss(reduction="sum") # sum instead of averaging
    
    def forward(self, predictions, target):
        # predictions should be of dimension (N, S, S, B*5+C)
        predictions.reshape_(-1, self.S, self.S, self.B * 5 + self.C)

        # get best IOU for box1 and box2
        box1_IOU = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        box2_IOU = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        IOUs = torch.cat(box1_IOU.unsqueeze(0), box2_IOU.unsqueeze(0), dim=0)
        resp_box = torch.max(IOUs, dim=0)[1]

        # get indicator 𝟙obj_i
        indicator = target[..., 20].unsqueeze(3)

        ## box coordinate + dimension loss ##
        box_targets = indicator * target[..., 21:25]
        # select predicted coordinates
        box_preds = indicator * (resp_box * predictions[..., 21:25] + (1 - resp_box) * predictions[..., 21:25])
        # select predicted dimensions and take sqrt
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        # take sqrt of target dimensions as well
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # calculate loss with sse
        box_loss = self.sse(torch.flatten(box_preds, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ## object loss ##
        # confidence score for responsible box (highest IOU)
        resp_confidence = resp_box * predictions[..., 25:26] + (1 - resp_box) * predictions[..., 20:21]
        object_loss = self.sse(torch.flatten(indicator * resp_confidence), torch.flatten(indicator * target[..., 20:21]))

        # no object loss
        noobj_loss_b1 = self.sse(torch.flatten((1 - indicator) * predictions[..., 20:21], start_dim=1),
                              torch.flatten((1 - indicator) * target[..., 20:21], start_dim=1))
        noobj_loss_b2 = self.sse(torch.flatten((1 - indicator) * predictions[..., 25:26], start_dim=1),
                              torch.flatten((1 - indicator) * target[..., 20:21], start_dim=1))
        noobj_loss = noobj_loss_b1 + noobj_loss_b2

        # class loss: MSE between 20 class predicted probabilities and one-hot target
        class_loss = self.sse(torch.flatten(indicator * predictions[..., :20], end_dim=-2), torch.flatten(indicator * target[..., :20], end_dim=-2))

        # make sure to include lambdas
        overall_loss = self.l_coord * box_loss + object_loss + self.l_noobj * noobj_loss + class_loss
        return overall_loss