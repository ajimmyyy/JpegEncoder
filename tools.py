import os
import numpy as np
from scipy.fftpack import dct
from utils import *
from huffman import HuffmanTree

LUMINANCE_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
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

def TransformRgbToYuc(rgbImg: np.ndarray) -> np.ndarray:
    yuvMatrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.419, -0.081]
    ])
    
    yuvImg = np.dot(rgbImg, yuvMatrix.T)
    yuvImg[:, :, 1] += 128
    yuvImg[:, :, 2] += 128
    yuvImg = np.int32(yuvImg) - 127

    return yuvImg

def TransformDCT(yuvImg: np.ndarray) -> np.ndarray:
    return dct(dct(yuvImg.T, norm='ortho').T, norm='ortho')

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
    direction = 1  # 1 for moving up, -1 for moving down

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

# reference from https://github.com/ghallak/jpeg-python/tree/master
def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values

def WriteToFile(filepath, dc, ac, rows, cols, tables):
    try:
        f = open(filepath, 'wb')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

    # write jpeg header
    f.write(bytes.fromhex('FFD8FFE000104A46494600010100000100010000'))
    # write y Quantization table
    f.write(bytes.fromhex('FFDB004300'))
    f.write(LUMINANCE_QUANTIZATION_TABLE.flatten().tobytes())
    # write c Quantization table
    f.write(bytes.fromhex('FFDB004301'))
    f.write(CHROMINANCE_QUANTIZATION_TABLE.flatten().tobytes())
    # write hight and width
    f.write(bytes.fromhex('FFC0001108'))
    f.write(bytes.fromhex(hex(rows)[2:].rjust(4,'0')))
    f.write(bytes.fromhex(hex(cols)[2:].rjust(4,'0')))

    # write Huffman table
    f.write(bytes.fromhex('03011100021101031101'))
    for name, table in tables.items():
        f.write(bytes.fromhex('FFC4'))
        f.write(bytes.fromhex('00'))

    # write data
    f.write(bytes.fromhex('FFDA000C03010002110311003F00'))
    f.write(bytes.fromhex())
    f.write(bytes.fromhex('FFD9'))
    f.close()

def CompressionImg(img, addr = ".jpg") -> np.ndarray:
    imgMatrix = np.array(img)
    padImg = Padding(imgMatrix)
    yuvImg = TransformRgbToYuc(padImg)

    rows, cols = yuvImg.shape[:2]

    if rows % 8 == cols % 8 == 0:
        blocksCount = rows // 8 * cols // 8
    else:
        raise ValueError("Image dimensions should be divisible by 8")
    
    dcMatrix = np.zeros((blocksCount, 3), dtype=np.int32)
    acMatrix = np.zeros((blocksCount, 63, 3), dtype=np.int32)
    
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                blockIndex += 1
            except NameError:
                blockIndex = 0

            for k in range(3):
                block = yuvImg[i:i+8, j:j+8, k]
                dctMatrix  = TransformDCT(block)
                quantMatrix = Quantize(dctMatrix, "luminance" if k == 0 else "chrominance")
                zigzagMatrix = ZigZag(quantMatrix)

                dcMatrix[blockIndex, k] = zigzagMatrix[0]
                acMatrix[blockIndex, :, k] = zigzagMatrix[1:]

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dcMatrix[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dcMatrix[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(acMatrix[i, :, 0])[0]
                    for i in range(blocksCount)))
    H_AC_C = HuffmanTree(
            flatten(run_length_encode(acMatrix[i, :, j])[0]
                    for i in range(blocksCount) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}
    
    print(tables['ac_c'])

    # WriteToFile(addr, dcMatrix, acMatrix, rows, cols, tables)

    return            