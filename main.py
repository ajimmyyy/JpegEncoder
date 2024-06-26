from PIL import Image
import numpy as np
import tools

BMP_PATH = "Img\lena.bmp"

if __name__ == "__main__":
    img = Image.open(BMP_PATH)
    imgMatrix = np.array(img)

    yuvImg = tools.TransformRgbToYuc(imgMatrix)