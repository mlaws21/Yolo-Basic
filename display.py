from PIL import Image, ImageDraw, ImageFont
from IPython.display import display


def add_box(drawer, font, bb, rc, color: str, text: str, size: int, s: int):
    """ Takes in an image, and bounding box details and adds the box to the image.
    This is a helper method to the display_image function

    Args:
        drawer: ImageDraw object of target image
        font: PIL font object 
        bb: a iterable of length four storing the relative x, y, width, height of the
            bounding box respectively
        rc: a tuple containing the row and column of the grid that the bounding box
            is relative to
        color (str): color to make the box
        text (str): label for the box
        size (int): size in pixels of input image
        s (int): grid size
    """

    cx_rel, cy_rel, width_rel, height_rel = bb
    width, height = (width_rel / s) * size, (height_rel / s) * size
    row, col = rc
    cx = (size / s) * (col + cx_rel)
    cy = (size / s) * (row + cy_rel)
    top_left_x, top_left_y = cx - width / 2, cy - height / 2
    
    drawer.rectangle([top_left_x, top_left_y, top_left_x + width, top_left_y + height], outline=color, width=2)
    drawer.text((top_left_x, top_left_y - 15), text, fill=color, font=font)  



def display_image(image_path: str, bb, rc, colors: list[str], text, size=400, s=3, jupyter=True):
    """Returns image to be displayed if jupyter flag is set to true else 
    displays image in a preview window. Note the lenght of bb, rc, colors, and 
    text must all be the same.

    Args:
        image_path (str): path to image to display
        bb : list of bounding arrays to be added to image
        rc (): list of associated grid positions to the bounding boxes
        colors (list[str]): list of associated colors for each box
        text (list[str]): list of associated labels for each box
        size (int, optional): image size. Defaults to 400.
        s (int, optional): grid size. Defaults to 3.
        jupyter (bool, optional): Whether or not display_image is being called in a
        jupyter notebook. Defaults to True.

    Returns:
        PIL image if jupyter is set to true else None
    """

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    assert len(bb) == len(rc) == len(colors) == len(text)
    
    for i in range(len(bb)):
        add_box(draw, font, bb[i], rc[i], colors[i], text[i], size, s)
    
    if jupyter:
        return image
    else:
        image.show()
    
    


def main():
    image_path = "./quicktest/test/X/31.png"

    BOUNDING_BOX_1 = [0.6825, 0.9763, 0.3619, 0.3612]
    RC_1 = [0, 0]
    display(image_path, [BOUNDING_BOX_1], [RC_1], ["red"], ["Square"])
    
if __name__ == "__main__":
    main()
