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
#   8  [   w   ]  true width
#   9  [   h   ]  true height
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

def intersection_over_union(predictions, targets):
    """
    Calculates intersection over union

    Parameters:
        predictions (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        targets (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """


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

    
def yolo_loss(predictions, target, S=3, B=1, C=5):
    
        l_coord = 5
        l_noobj = 0.5
        sse = torch.nn.MSELoss(reduction="sum") # sum instead of averaging
        
        # predictions should be of dimension (N, S, S, B*5+C)
        # -1 is for the batch
        predictions = predictions.reshape((-1, S, S, B * 5 + C))


        # get best IOU for bounding box
        resp_box = intersection_over_union(predictions[..., C + 1:], target[..., C + 1:])

        # get indicator 𝟙obj_i
        indicator = target[..., C].unsqueeze(3)

        ## box coordinate + box dimension loss ##
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

        resp_confidence = resp_box * predictions[..., C:C+1]
        
        object_loss = sse(torch.flatten(indicator * resp_confidence), torch.flatten(indicator * target[..., C:C+1]))

        # no object loss
        noobj_loss_b1 = sse(torch.flatten((1 - indicator) * predictions[..., C:C+1], start_dim=1),
                              torch.flatten((1 - indicator) * target[..., C:C+1], start_dim=1))

        noobj_loss = noobj_loss_b1

        # class loss: SSE between 5 class predicted probabilities and one-hot target
        class_loss = sse(torch.flatten(indicator * predictions[..., :C], end_dim=-2), torch.flatten(indicator * target[..., :C], end_dim=-2))

        overall_loss = l_coord * box_loss +  object_loss + l_noobj * noobj_loss + class_loss
        return overall_loss
    

def nlog_softmax_loss(X, y):
    """
    A loss function based on softmax, described in colonels2.ipynb. 
    X is the (batch) output of the neural network, while y is a response 
    vector.
    
    See the unit tests in test.py for expected functionality.
    
    """    
    smax = torch.softmax(X, dim=1)
    correct_probs = torch.gather(smax, 1, y.unsqueeze(1))
    nlog_probs = -torch.log(correct_probs)
    return torch.mean(nlog_probs) 