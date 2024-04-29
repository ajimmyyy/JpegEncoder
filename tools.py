import numpy as np


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
