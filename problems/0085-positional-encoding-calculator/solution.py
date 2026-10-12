import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    pos = np.arange(position)[:, np.newaxis]
    dim = np.arange(d_model)

    angle_rates = 1 / np.power(10000, (2 * (dim // 2)) / d_model)
    angles = pos * angle_rates

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles.astype(np.float16)