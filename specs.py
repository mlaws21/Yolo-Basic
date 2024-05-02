pretrain_specs = [
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("conv", (3, 20, 1)),
    ("relu", ()),
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
    # ("conv", (3, 20, 1)),
    # ("relu", ()),
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

IMG_SIZE = 112

additional_yolo_specs = [
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("conv", (1, 20, 1)),
    ("relu", ()),
    ("conv", (3, 20, 1)),
    ("relu", ()),
    ("conv", (1, 20, 1)),
    ("relu", ()),
    ("flatten", ()),
    ("dense", (20 * (IMG_SIZE // 16)**2, 2048)),
    
    ("relu", ()),
    ("dense", (2048, 7 * 7 * 10)),
    ("relu", ()),
    
]

# TODO I think we need a softmax somewhere...

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
