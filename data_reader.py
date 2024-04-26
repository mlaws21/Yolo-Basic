import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import os
import pandas as pd
from torch import tensor

class ImageDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

label_to_Y = {
    "chainsaw": 0,
    "dog": 1,
    "fhorn": 2,
    "gas": 3,
    "music": 4,
    "church": 5,
    "dumptruck": 6,
    "fish": 7,
    "golf": 8,
    "parachute": 9
}

def import_pretrain_data(data_path, W=112, batch_size=4, speed=20):
    training_data = []
    # ytrain = []
    
    labs = sorted(os.listdir(data_path))
    if ".DS_Store" in labs: labs.remove(".DS_Store")

    for lab in labs:
        images = os.listdir(os.path.join(data_path, lab))
        if ".DS_Store" in images: images.remove(".DS_Store")
        
        for i in images[::speed]:
            img_path = os.path.join(data_path, lab, i)
            try:
                img = io.read_image(img_path, io.ImageReadMode.RGB)
            except:
                raise Exception(f"The following image is missing or corrupted: {img_path}")
            resize = transforms.Resize((W, W))
            img = resize(img)
            grayscale = transforms.Grayscale(num_output_channels=1)
            img = grayscale.forward(img)
             
            training_data.append((img, torch.nn.functional.one_hot(tensor(label_to_Y[lab]), 10)))

        print(lab, end=" ", flush=True)
    print()
    image_dataset = ImageDataset(training_data)
    dl = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dl

def import_bounding_data(data_path, routine="train", W=128, batch_size=4):

    training_data = []
    resize = transforms.Resize((W, W))
    grayscale = transforms.Grayscale(num_output_channels=1)
    
    
    csv = os.path.join(data_path, f"{routine}.csv")
    
    train = pd.read_csv(csv)
    for _, data in train.iterrows():
        x_datapath, y_datapath = data
        
        ### IMAGE DATA  (X) ###
        x_file = csv = os.path.join(data_path, "images", x_datapath)
        try:
            img = io.read_image(x_file, io.ImageReadMode.RGB)
        except:
            raise Exception(f"The following image is missing or corrupted: {x_file}")
        
        img = resize(img)
        img = grayscale.forward(img)
        
        
        ### BOUDING DATA  (Y) ###
        
        y_file = csv = os.path.join(data_path, "labels", y_datapath)

        
        # boundaries = []
        bctr = 0
        try:
            with open(y_file, 'r') as file:
        # Read the file line by line
                for line in file:
                    # Process each line (e.g., print it)
                    
                    # append(img, )
                    # print(line.split())
                    l_split = line.split()
                    

                    prob_dist = (torch.nn.functional.one_hot(tensor(int(l_split[0])), 20))
                    l_split[0] = 1
                    float_split = [float(x) for x in l_split]
                    output = torch.cat((prob_dist, tensor(float_split, dtype=torch.float32)))
                    # boundaries.append(output)
                    training_data.append((img, output))
                    bctr += 1
                    
        except:
            raise Exception(f"The following text is missing or corrupted: {y_file}")
        
        # if len(boundaries) == 0:
        #     boundaries.append(torch.zeros((25,)))
        # training_data.append((img, boundaries))
        
        # TODO how do we ecode multple boxes or zero boxes
        if bctr == 0:
            training_data.append((img, torch.zeros((25,))))
        

    image_dataset = ImageDataset(training_data)
    dl = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dl


def show_image(img):
    """Shows an image, represented as a torch tensor of shape (3, W, W)."""
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2)
    # imshow(img)



def main():

    # dl = import_pretrain_data("imagenette2/train")

    dl = import_bounding_data("voc")
    
    for i, ele in enumerate(dl):
        if i == 0:
            print(ele)
if __name__ == "__main__":
    main()