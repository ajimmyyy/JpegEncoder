from PIL import Image
import numpy as np
import os
import cv2
from . import filesaver
from .huffman import Huffman, HuffmanTable

# LUMINANCE_QUANTIZATION_TABLE = np.array([
#     [16, 11, 10, 16, 24, 40, 51, 61],
#     [12, 12, 14, 19, 26, 58, 60, 55],
#     [14, 13, 16, 24, 40, 57, 69, 56],
#     [14, 17, 22, 29, 51, 87, 80, 62],
#     [18, 22, 37, 56, 68, 109, 103, 77],
#     [24, 35, 55, 64, 81, 104, 113, 92],
#     [49, 64, 78, 87, 103, 121, 120, 101],
#     [72, 92, 95, 98, 112, 100, 103, 99]
# ])

LUMINANCE_QUANTIZATION_TABLE = np.array([
    [3, 2, 2, 3, 5, 8, 10, 12],
    [2, 2, 3, 4, 5, 12, 12, 11],
    [3, 3, 3, 5, 8, 11, 14, 11],
    [3, 3, 4, 6, 10, 17, 16, 12],
    [4, 4, 7, 11, 14, 22, 21, 15],
    [5, 7, 11, 13, 16, 21, 23, 18],
    [10, 13, 16, 17, 21, 24, 24, 20],
    [14, 18, 19, 20, 22, 20, 21, 20]
])

# CHROMINANCE_QUANTIZATION_TABLE = np.array([
#     [17, 18, 24, 47, 99, 99, 99, 99],
#     [18, 21, 26, 66, 99, 99, 99, 99],
#     [24, 26, 56, 99, 99, 99, 99, 99],
#     [47, 66, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99]
# ])

CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [3, 4, 5, 9, 20, 20, 20, 20],
    [4, 4, 5, 13, 20, 20, 20, 20],
    [5, 5, 11, 20, 20, 20, 20, 20],
    [9, 13, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20]
])

def Padding(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    if (height % 8 != 0):
        height = height // 8 * 8 + 8
    if (width % 8 != 0):
        width = width // 8 * 8 + 8

    padImage = np.zeros((height, width, 3), dtype=np.uint8)
    padImage[:height,:width] = img

    return padImage

def TransformRgbToYCbCr(rgbImg: np.ndarray) -> np.ndarray:
    transform_matrix = np.array([
        [    0.299,     0.587,     0.114],
        [-0.168736, -0.331264,       0.5],
        [      0.5, -0.418688, -0.081312]])
    
    offset = np.array([0, 128, 128])
    ycbcrImg = rgbImg @ transform_matrix.T + offset
    ycbcrImg = np.clip(ycbcrImg, 0, 255)
    ycbcrImg = np.round(ycbcrImg)

    return ycbcrImg.astype(np.uint8)

def TransformDCT(img: np.ndarray) -> np.ndarray:
    return cv2.dct(np.float32(img))

def Quantize(block: np.ndarray, type: str) -> np.ndarray:
    if type == "luminance":
        q = LUMINANCE_QUANTIZATION_TABLE
    elif type == "chrominance":
        q = CHROMINANCE_QUANTIZATION_TABLE
    else:
        raise ValueError("type should be either 'luminance' or 'chrominance'")

    return (block / q).round().astype(np.int32)

def ZigZag(block: np.ndarray) -> np.ndarray:
    zigzag = []
    rows, cols = block.shape
    row, col = 0, 0
    direction = 1

    for _ in range(rows * cols):
        zigzag.append(block[row, col])
        if direction == 1:
            if col == cols - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            if row == rows - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1

    return np.array(zigzag)

def CompressionImg(imgAddr, outputAddr = ".jpg") -> np.ndarray:
    img = Image.open(imgAddr)
    imgMatrix = np.array(img)
    padImg = Padding(imgMatrix)
    ycbcrImg = TransformRgbToYCbCr(padImg)

    rows, cols = ycbcrImg.shape[:2]

    if rows % 8 == cols % 8 == 0:
        blocksCount = rows // 8 * cols // 8
    else:
        raise ValueError("Image dimensions should be divisible by 8")
    
    dcMatrix = np.zeros((blocksCount, 3), dtype=np.int32)
    acMatrix = np.zeros((blocksCount, 3, 63), dtype=np.int32)

    previousDC = np.zeros(3, dtype=np.int32)
    
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                blockIndex += 1
            except NameError:
                blockIndex = 0

            for k in range(3):
                block = ycbcrImg[i:i+8, j:j+8, k]
                dctMatrix  = TransformDCT(block)

                quantMatrix = Quantize(dctMatrix, "luminance" if k == 0 else "chrominance")
                zigzagMatrix = ZigZag(quantMatrix)

                dcDifference = zigzagMatrix[0] - previousDC[k]
                dcMatrix[blockIndex, k] = dcDifference
                previousDC[k] = zigzagMatrix[0]
                acMatrix[blockIndex, k, :] = zigzagMatrix[1:]

    huffman = Huffman(dcMatrix, acMatrix)
    bitStream, tables = huffman.EncodeDCAC(useDefault=True)

    filesaver.WriteJpeg(bitStream, tables, LUMINANCE_QUANTIZATION_TABLE, CHROMINANCE_QUANTIZATION_TABLE, rows, cols, outputAddr)

    return os.stat(imgAddr).st_size / 1024, os.stat(outputAddr).st_size / 1024
