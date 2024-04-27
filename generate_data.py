from PIL import Image, ImageDraw
from random import randint
import os
size = 400


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
    draw.line([(top_left_x, top_left_y), (top_left_x + ele_size, top_left_y + ele_size)], fill="red", width=10)

    # Draw the second diagonal line (\)
    draw.line([(top_left_x + ele_size, top_left_y), (top_left_x, top_left_y + ele_size)], fill="red", width=10)

    # Save the image to a file
    image.save(name + ".png")


    # Close the image
    image.close()
    
    
def drawO(name):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    line_length = randint(50, 200)
    
    ele_size = int(line_length / (2**0.5))
    top_left_x, top_left_y = randint(0, size - ele_size), randint(0, size - ele_size)

    draw.ellipse([top_left_x, top_left_y, top_left_x + ele_size, top_left_y + ele_size ], outline="blue", width=10)
    # draw.ellipse([center_x, center_y, 300, 300], outline="blue", width=10)
    

    image.save(name + ".png")

    image.close()
    
    
def make_synth_data(folder, numX, numO=None):
    try:
        os.makedirs(folder)
    except:
        print("WARNING: directory already exists")
    if numO is None:
        numO = numX
        
    for i in range(numX):
        drawX(os.path.join(folder, f"X{i}"))
        
    for i in range(numO):
        drawO(os.path.join(folder, f"O{i}"))
        
def main():
    make_synth_data("XO/train/", 20000)
    
    make_synth_data("XO/test/", 1000)
        
if __name__ == "__main__":
    main()
