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
#   2  [   0   ] one hot encoding of true value
#   3  [   1   ]
#   4  [   0   ]
#   5  [   1   ]  object exists here (indicator)
#   6  [   x   ]  true x
#   7  [   y   ]  true y
#   8  [   w   ]  etc.
#   9  [   h   ]
#
#
#         OUT
#   0  [  c01  ]
#   1  [  c02  ]
#   2  [  c03  ] category probs p̂_i(c)
#   3  [  c04  ]
#   4  [  c05  ] 
#   5  [ conf. ]  Ĉ for box 1
#   6  [   x   ]  x for box 1
#   7  [   y   ]  etc.
#   8  [   w   ]
#   9  [   h   ]

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
    def __init__(self, S = 7, B = 1, C = 5):
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
        IOU = intersection_over_union(predictions[..., self.C + 1:], target[..., self.C + 1:])
        # box2_IOU = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # IOUs = torch.cat(box1_IOU.unsqueeze(0), box2_IOU.unsqueeze(0), dim=0)
        resp_box = torch.max(IOU, dim=0)[1] # Is this the max between IOUs or somehting else

        # get indicator 𝟙obj_i
        indicator = target[..., self.C].unsqueeze(3)

        ## box coordinate + dimension loss ##
        box_targets = indicator * target[..., self.C + 1:]
        # select predicted coordinates
        box_preds = indicator * (resp_box * predictions[..., self.C + 1:] + (1 - resp_box) * predictions[..., self.C + 1:])
        # select predicted dimensions and take sqrt
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        # take sqrt of target dimensions as well
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # calculate loss with sse
        box_loss = self.sse(torch.flatten(box_preds, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ## object loss ##
        # confidence score for responsible box (highest IOU)
        resp_confidence = resp_box * predictions[..., self.C]
        object_loss = self.sse(torch.flatten(indicator * resp_confidence), torch.flatten(indicator * target[..., self.C]))

        # no object loss
        noobj_loss_b1 = self.sse(torch.flatten((1 - indicator) * predictions[..., self.C], start_dim=1),
                              torch.flatten((1 - indicator) * target[..., self.C], start_dim=1))
        # noobj_loss_b2 = self.sse(torch.flatten((1 - indicator) * predictions[..., 25:26], start_dim=1),
        #                       torch.flatten((1 - indicator) * target[..., 20:21], start_dim=1))
        noobj_loss = noobj_loss_b1 # + noobj_loss_b2

        # class loss: MSE between 20 class predicted probabilities and one-hot target
        class_loss = self.sse(torch.flatten(indicator * predictions[..., :self.C], end_dim=-2), torch.flatten(indicator * target[..., :self.C], end_dim=-2))

        # make sure to include lambdas
        overall_loss = self.l_coord * box_loss + object_loss + self.l_noobj * noobj_loss + class_loss
        return overall_loss
    
def yolo_loss_func(predictions, target, S=3, B=1, C=5):
    
        l_coord = 5
        l_noobj = 0.5
        sse = torch.nn.MSELoss(reduction="sum") # sum instead of averaging
        
        # predictions should be of dimension (N, S, S, B*5+C)
        # predictions.reshape_(-1, S, S, B * 5 + C)
        predictions = predictions.reshape((-1, S, S, B * 5 + C))
        # print(predictions.shape, target.shape)

        # get best IOU for box1 and box2
        resp_box = intersection_over_union(predictions[..., C + 1:], target[..., C + 1:])
        # box2_IOU = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # IOUs = torch.cat(box1_IOU.unsqueeze(0), box2_IOU.unsqueeze(0), dim=0)
        # resp_box = torch.max(IOU, dim=0)[1] # Is this the max between IOUs or somehting else

        # get indicator 𝟙obj_i
        indicator = target[..., C].unsqueeze(3)

        ## box coordinate + dimension loss ##
        box_targets = indicator * target[..., C + 1:]
        # select predicted coordinates
        box_preds = indicator * (resp_box * predictions[..., C + 1:] + (1 - resp_box) * predictions[..., C + 1:])
        # select predicted dimensions and take sqrt
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        # take sqrt of target dimensions as well
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # calculate loss with sse
        box_loss = sse(torch.flatten(box_preds, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ## object loss ##
        # confidence score for responsible box (highest IOU)
        resp_confidence = resp_box * predictions[..., C:C+1] # TODO could be 1 - resp_box
        # print(torch.sum(predictions[..., C:C+1]))
        # print(predictions[..., C:C+1].shape)
        
        object_loss = sse(torch.flatten(indicator * resp_confidence), torch.flatten(indicator * target[..., C:C+1]))

        # no object loss
        noobj_loss_b1 = sse(torch.flatten((1 - indicator) * predictions[..., C:C+1], start_dim=1),
                              torch.flatten((1 - indicator) * target[..., C:C+1], start_dim=1))
        # noobj_loss_b2 = sse(torch.flatten((1 - indicator) * predictions[..., 25:26], start_dim=1),
        #                       torch.flatten((1 - indicator) * target[..., 20:21], start_dim=1))
        noobj_loss = noobj_loss_b1 # + noobj_loss_b2

        # class loss: MSE between 20 class predicted probabilities and one-hot target
        class_loss = sse(torch.flatten(indicator * predictions[..., :C], end_dim=-2), torch.flatten(indicator * target[..., :C], end_dim=-2))

        # make sure to include lambdas
        overall_loss = l_coord * box_loss + object_loss + l_noobj * noobj_loss + class_loss
        return overall_loss