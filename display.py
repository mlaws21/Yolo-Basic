from PIL import Image, ImageDraw, ImageFont
from IPython.display import display




# Load the image

def add_box(drawer, font, bb, rc, color, text, size, s):
    cx_rel, cy_rel, width_rel, height_rel = bb
    width, height = (width_rel / s) * size, (height_rel / s) * size
    row, col = rc
    cx = (size / s) * (col + cx_rel)
    cy = (size / s) * (row + cy_rel)
    top_left_x, top_left_y = cx - width / 2, cy - height / 2
    
    drawer.rectangle([top_left_x, top_left_y, top_left_x + width, top_left_y + height], outline=color, width=2)
    

# Get the size of the text

    drawer.text((top_left_x, top_left_y - 15), text, fill=color, font=font)  
    # Calculate the position to center the text
    # x = (drawer.width - text_width) // 2
    # y = (drawer.height - text_height) // 2

    # # Draw the text on the image
    # drawer.text((x, y), text, fill="black", font=font)



def display_image(image_path, bb, rc, colors, text, size=400, s=3, jupyter=True):
    

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    assert len(bb) == len(rc) == len(colors) == len(text)
    
    for i in range(len(bb)):
        add_box(draw, font, bb[i], rc[i], colors[i], text[i], size, s)
    
    # image.save(".cheat.png")
    if jupyter:
        return image
    else:
        image.show()
    
    


def main():
    image_path = "./quicktest/test/X/31.png"

    BOUNDING_BOX_1 = [0.6825, 0.9763, 0.3619, 0.3612]
    RC_1 = [0, 0]
    display(image_path, [BOUNDING_BOX_1], [RC_1], ["red"], ["Square"])
    # display(image_path [BOUNDING_BOX_1], [RC_1], colors, shape)
    
    
    
if __name__ == "__main__":
    main()
# # Get the dimensions of the image
# width, height = image.size

# # Calculate the size of each grid cell
# cell_width = width //3
# cell_height = height //3

# # Draw vertical grid lines
# for i in range(1, 3):
#     x = i * cell_width
#     draw.line((x, 0, x, height), fill="black", width=2)

# # Draw horizontal grid lines
# for i in range(1, 3):
#     y = i * cell_height
#     draw.line((0, y, width, y), fill="black", width=2)



# O44
# BOUNDING_BOX = [0.5544, 0.6305, 0.7230, 0.7038]
# RC = [0, 1]

# # X31
# BOUNDING_BOX_1 = [0.6825, 0.9763, 0.3619, 0.3612]
# RC_1 = [0, 0]
# add_box(BOUNDING_BOX_1, RC_1)

# BOUNDING_BOX_2 = [0.7897, 0.6743, 0.3488, 0.3502]
# RC_2 = [2, 0]
# add_box(BOUNDING_BOX_2, RC_2)


# Save or display the modified image
# image.show()
# image.save("path_to_save_modified_image.jpg")
