import struct
import numpy as np
from typing import Literal

def read_data(file_path, mode: Literal['image', 'label'] = 'image'):
    try:
        with open(file_path, mode='rb') as file:
            magic = int.from_bytes(file.read(4), byteorder='big')
            image_count = int.from_bytes(file.read(4), byteorder='big')
            if mode == 'image':
                row_count = int.from_bytes(file.read(4), byteorder='big')
                col_count = int.from_bytes(file.read(4), byteorder='big')
                images_data = np.frombuffer(file.read(), dtype=np.uint8)
                images_data = images_data.reshape(image_count, row_count * col_count)
                images_data = images_data / 255.0

                return images_data
            else:
                labels = np.frombuffer(file.read(), dtype=np.uint8)
                return labels

    except FileNotFoundError:
        print("Error, file not found!")

if __name__ == "__main__":
    print(np.shape(read_data(r'S:\NeuralNetwork\t10k-images.idx3-ubyte')))