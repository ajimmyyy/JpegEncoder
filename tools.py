import numpy as np
from scipy.fftpack import dct


def BlockReduce(image, block_size, func=np.mean):
    assert len(image.shape) == 2, "Input image must be 2D"
    assert isinstance(block_size, tuple), "Block size must be a tuple (rows, cols)"
    assert all(isinstance(size, int) and size > 0 for size in block_size), "Block size must contain positive integers"
    assert image.shape[0] >= block_size[0] and image.shape[1] >= block_size[1], "Block size must be smaller than image size"

    rows, cols = image.shape[0] // block_size[0], image.shape[1] // block_size[1]
    result = np.empty((rows, cols), dtype=image.dtype)

    for i in range(rows):
        for j in range(cols):
            block = image[i * block_size[0]: (i + 1) * block_size[0], j * block_size[1]: (j + 1) * block_size[1]]
            result[i, j] = func(block)

    return result


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


def TrandformDCT(yuvImg: np.ndarray) -> np.ndarray:
    grayImg = yuvImg[:, :, 0]

    blocks = BlockReduce(grayImg, (8, 8), np.mean)
    dctBlocks = np.zeros_like(blocks)
    
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dctBlocks[i, j] = dct(dct(blocks[i, j], norm='ortho'), norm='ortho')

    return dctBlocks