from PIL import Image
import numpy as np
import tools
import matplotlib.pyplot as plt

BMP_PATH = "Img\lena.bmp"

if __name__ == "__main__":
    img = Image.open(BMP_PATH)
    tools.CompressionImg(img, "Img/lena.jpg")