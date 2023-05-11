import numpy as np
import h5py
import glob
from PIL import Image

batch_size = 1000  # 每批包含的图片数
img_shape = (224, 224)  # 假设所有图片的尺寸均为 (224, 224)
image_folder = 'path/to/images'
file_list = glob.glob(f'{image_folder}/*.jpg')

def load_images(file_list):
    """将文件列表中的图片读取并返回一个 NumPy 数组"""
    images = []
    for file in file_list:
        img = Image.open(file)
        img = img.resize(img_shape)
        arr = np.array(img, dtype='uint8')
        images.append(arr)
    return np.array(images)

with h5py.File('data.hdf5', 'w') as f:
    for i, batch_files in enumerate(batch_file_list):
        print(f'Processing batch {i+1}')
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(file_list))

        # 将文件列表划分成批次
        batch_files = file_list[start_idx:end_idx]

        # 读取批次中的所有图片，并将它们合并成一个 NumPy 数组
        batch_images = load_images(batch_files)

        # 将 NumPy 数组转换为 Tensor
        batch_tensor = np.expand_dims(batch_images, axis=1)

        # 将 Tensor 存储到 HDF5 文件中
        f.create_dataset(f'batch_{i}', data=batch_tensor)


# python -m multiprocess image_to_hdf5.py