import os

def make(folder):
    f = open("shapes.json", "w")
    f.write("[\n")
    
    # train
    train_catagories = os.listdir(os.path.join(folder, "train"))
    
    for cat_data in train_catagories:
        for ele in (os.listdir(os.path.join(folder, "train", cat_data))):
            # posOrNeg = "positive" if "X" in ele else "negative"
            f.write("{\n\"label\": \"" + cat_data + "\",\n")
            f.write("\"filename\": \"" + os.path.join(folder, "train", cat_data, ele) + "\",\n")
            # if i < len(train_data) - 1:
            f.write("\"partition\": \"train\"\n},\n")


        # else: 
            # f.write("\"partition\": \"train\"\n}\n")
            
        

    
    test_catagories = os.listdir(os.path.join(folder, "test"))
    
    for cat_data in test_catagories:
    
        for  ele in (os.listdir(os.path.join(folder, "test", cat_data))):

            # posOrNeg = "positive" if "X" in ele else "negative"
            f.write("{\n\"label\": \"" + cat_data  + "\",\n")
            f.write("\"filename\": \"" + os.path.join(folder, "test", cat_data, ele) + "\",\n")
            # if i < len(train_data) - 1:
            f.write("\"partition\": \"test\"\n},\n")

            
            # else: 
            #     f.write("\"partition\": \"test\"\n}\n")
                
        
    
    f.write("]\n")
    
    f.close()
    
if __name__ == "__main__":
    make("shapes")