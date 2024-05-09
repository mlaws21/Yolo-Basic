from PIL import Image, ImageDraw
from random import randint
import os
from math import sin, cos
import sys
# ratio_size = 244/400

size = 400

# TODO return relative size (0, 1)
box_adjust = 6

def drawX(name):
    
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    
    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)
    
    # Draw X
    draw.line([(top_left_x, top_left_y), (top_left_x + ele_size, top_left_y + ele_size)], fill="red", width=5)
    draw.line([(top_left_x + ele_size, top_left_y), (top_left_x, top_left_y + ele_size)], fill="red", width=5)
    image.save(name + ".png")
    image.close()
    
    cx, cy = top_left_x + ele_size / 2, top_left_y + ele_size / 2
    return ele_size + box_adjust, cx, cy
    
    
def drawO(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    # Draw O
    draw.ellipse([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="blue", width=5)
    image.save(name + ".png")
    image.close()
    
    cx, cy = top_left_x + ele_size / 2, top_left_y + ele_size / 2
    return ele_size + box_adjust, cx, cy
    
def drawSquare(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    # Draw Square
    draw.rectangle([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="green", width=5)
    image.save(name + ".png")
    image.close()

    cx, cy = top_left_x + ele_size / 2, top_left_y+ ele_size / 2
    return ele_size + box_adjust, cx, cy

def drawStar(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    # Draw Star
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
    image.save(name + ".png")
    image.close()
    

    cx, cy = top_left_x + ele_size / 2, top_left_y + ele_size / 2
    return ele_size + box_adjust, cx, cy

def drawPent(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)
    
    # Draw Pent
    cx, cy = top_left_x + ele_size / 2, top_left_y + ele_size / 2
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
    
def generate_set(folder, numX):
    
    shapes = [("X",drawX), ("O", drawO), ("Pent", drawPent), ("Square", drawSquare), ("Star",drawStar)]
    
    
    try:
        os.makedirs(folder)
    except:
            print("WARNING: directory already exists")

    root = folder.split("/")[0]
    train_test = folder.split("/")[1]
    
    
    json_out = os.path.join(root, "data.json")
    f = open(json_out, "a+")
    
    for lab, func in shapes:
        
        lab_path = os.path.join(folder, lab)
        
        try:
            os.mkdir(lab_path)
        except:
            print("WARNING: directory already exists")
        for j in range(numX):

            bounding_box = func(os.path.join(lab_path, str(j)))
            if bounding_box is not None:
                hw, cx, cy = bounding_box
                height, width = hw, hw
                bb_data = f"{1},{cx / size},{cy / size},{width / size},{height / size}"
            else:
                bb_data = f"{0},{0},{0},{0},{0}"
                
                
            f.write("{\n\"label\": \"" + lab  + "\",\n")
            f.write("\"filename\": \"" + os.path.join(folder, lab, str(j) + ".png") + "\",\n")
            f.write("\"partition\": \"" + train_test +  "\",\n")
            f.write("\"box\": \"" + bb_data + "\"\n},\n")
            
    f.close()

def draw_star_and_O(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    # Draw Star
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
    
    
    line_length = randint(50, 200)
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)
    
    draw.ellipse([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="blue", width=5)

    
    image.save(name + ".png")
    image.close()

def remove_last_char(file_path):
    with open(file_path, 'r+') as file:
        contents = file.read()
        file.seek(0, 2)  # Move the cursor to the end of the file
        file.truncate(len(contents) - 1)  # Truncate the file to remove the last character

def generate_data(root: str, num_train: int, num_test: int):
    
    try:
        os.mkdir(root)
    except:
        print(f"ERROR: Data Directory {root} already exists")
        print(f"if you would like to continue run rm -r {root} in your terminal")
        return
        
    json_out = os.path.join(root, "data.json")
    f = open(json_out, "a+")
    f.write("[\n")
    f.close()
    
    generate_set(os.path.join(root, "train"), num_train)
    generate_set(os.path.join(root, "test"), num_test)
    
    remove_last_char(json_out) # removes return
    remove_last_char(json_out) #removes comma
    
    f = open(json_out, "a+")
    f.write("\n]\n")
    f.close()

    
                    
def main():
    
    generate_data("shapes", 1000, 10)
    
    # draw_star_and_O("multi")


if __name__ == "__main__":
    main()
