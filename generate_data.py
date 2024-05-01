from PIL import Image, ImageDraw
from random import randint
import os
from math import sin, cos
# ratio_size = 244/400

size = 400

# TODO return relative size (0, 1)
box_adjust = 6

def drawX(name):

# Create a new image with white background
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    # Get the center coordinates of the image
    

    
    # Set the length of the lines
    line_length = randint(50, 200)
    
    ele_size = int(line_length / (2**0.5))
    # center_x, center_y = 200 + randint(-max_pos , max_pos), 200 + randint(-max_pos , max_pos)
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)
    
    # Draw the first diagonal line (/)
    draw.line([(top_left_x, top_left_y), (top_left_x + ele_size, top_left_y + ele_size)], fill="red", width=5)

    # Draw the second diagonal line (\)
    draw.line([(top_left_x + ele_size, top_left_y), (top_left_x, top_left_y + ele_size)], fill="red", width=5)

    # Save the image to a file
    image.save(name + ".png")


    # Close the image
    image.close()
    cx, cy = top_left_x + ele_size /2 , top_left_y+ ele_size //2
    
    return ele_size + box_adjust, cx, cy
    
    
def drawO(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    draw.ellipse([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="blue", width=5)
    # draw.ellipse([center_x, center_y, 300, 300], outline="blue", width=5)
    

    image.save(name + ".png")

    image.close()
    cx, cy = top_left_x + ele_size /2 , top_left_y+ ele_size //2
    
    return ele_size + box_adjust, cx, cy
    
def drawSquare(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    draw.rectangle([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="green", width=5)
    # draw.rectangle([top_left_x - 5, top_left_y - 5, top_left_x + ele_size + 5, top_left_y + ele_size + 5], outline="black", width=2)
    
    # draw.ellipse([center_x, center_y, 300, 300], outline="blue", width=5)
    

    image.save(name + ".png")

    image.close()
    cx, cy = top_left_x + ele_size /2 , top_left_y+ ele_size //2
    
    return ele_size + box_adjust, cx, cy

from PIL import Image, ImageDraw
from random import randint

def drawStar(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    # draw.rectangle([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="green", width=5)
    # Define the points of the star
    points = [
        (top_left_x + ele_size // 2, top_left_y),
        (top_left_x + 2 * ele_size // 3, top_left_y + ele_size // 3),
        (top_left_x + ele_size, top_left_y + ele_size // 2),
        (top_left_x + 2 * ele_size // 3, top_left_y + 2 * ele_size // 3),
        (top_left_x + ele_size // 2, top_left_y + ele_size),
        (top_left_x + ele_size // 3, top_left_y + 2 * ele_size // 3),
        (top_left_x, top_left_y + ele_size // 2),
        (top_left_x + ele_size // 3, top_left_y + ele_size // 3),
    ]

    draw.polygon(points, outline="yellow", width=5)
    # draw.rectangle([top_left_x - 5, top_left_y - 5, top_left_x + ele_size + 5, top_left_y + ele_size + 5], outline="black", width=2)

    image.save(name + ".png")
    image.close()
    cx, cy = top_left_x + ele_size /2 , top_left_y+ ele_size //2
    
    return ele_size + box_adjust, cx, cy
    

# Example usage

def drawPent(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)
    cx, cy = top_left_x + ele_size /2 , top_left_y+ ele_size //2
    # draw.rectangle([top_left_x - 5, top_left_y - 5, top_left_x + ele_size + 5, top_left_y + ele_size + 5], outline="black", width=2)
    # Define the points of the star
    radius = ele_size
    points = []
    for i in range(5):
        angle = i * 2 * 3.14159 / 5 - 3.14159 / 2
        x = cx + int(radius * 0.5 * cos(angle))
        y = cy + int(radius * 0.5 * sin(angle))
        points.append((x, y))

    draw.polygon(points, outline="orange", width=5)

    image.save(name + ".png")
    image.close()
    return ele_size + box_adjust, cx, cy

def drawBlank(name):
    image = Image.new("RGB", (size, size), "white")

    image.save(name + ".png")
    image.close()
    return None
    
def make_synth_data(folder, numX):
    
    shapes = [("X",drawX), ("O", drawO), ("Pent", drawPent), ("Square", drawSquare), ("Star",drawStar)]
    
    
    try:
        os.makedirs(folder)
    except:
            print("WARNING: directory already exists")

    root = folder.split("/")[0]
    train_test = folder.split("/")[1]
    
    
    json_out = os.path.join(root, "data.json")
    f = open(json_out, "a+")
    
    # f.write("[\n")
    for lab, func in shapes:
        
        lab_path = os.path.join(folder, lab)
        # bounding_path = os.path.join(folder, "bounding")
        
        try:
            os.mkdir(lab_path)
            # os.mkdir(bounding_path)
        except:
            print("WARNING: directory already exists")
            
            
        

        for j in range(numX):

            bounding_box = func(os.path.join(lab_path, str(j)))
            if bounding_box is not None:
                hw, cx, cy = bounding_box
                height, width = hw, hw
                bb_data = f"{1},{cx / size},{cy / size},{cx / width},{height / size}"
            else:
                bb_data = f"{0},{0},{0},{0},{0}"
                
                
            f.write("{\n\"label\": \"" + lab  + "\",\n")
            f.write("\"filename\": \"" + os.path.join(folder, lab, str(j) + ".png") + "\",\n")
            f.write("\"partition\": \"" + train_test +  "\",\n")
            f.write("\"box\": \"" + bb_data + "\"\n},\n")
            

    # f.write("]\n")
    f.close()
                
            
            
            
            
            
        

        
    
        
def main():
    make_synth_data("one_shape/train/", 1000)
    
    make_synth_data("one_shape/test/", 100)
        
    # drawPent("pent")
    # drawO("o")
    # drawStar("st")
    # drawSquare("sq")
    # drawBlank("bl")
    # drawX("x")

if __name__ == "__main__":
    main()
