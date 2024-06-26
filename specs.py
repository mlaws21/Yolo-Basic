### This file contains the specs we will use

pretrain_specs = [
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("pool", (4, -1, 4)),
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("pool", (4, -1, 4)),
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("pool", (2, -1, 2) ),
    ("flatten", ()),
    ("dense", (32 * (112 // 32)**2, 64)),
    ("l_relu", ()),
    ("drop", (0.5)),
    ("dense", (64, 5)), 
    
]

additional_yolo_specs = [
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("conv", (3, 32, 1)),
    ("l_relu", ()),
    ("conv", (1, 32, 1)),
    ("l_relu", ()),
    ("flatten", ()),
    ("dense", (32 * (112 // 32)**2, 2048)),
    ("l_relu", ()),
    ("drop", (0.5)),
    ("dense", (2048, 3 * 3 * 10)),
    
]
