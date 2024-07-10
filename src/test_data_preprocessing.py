from pathlib import Path
import h5py
from PIL import Image
import numpy as np
import io
from tqdm import tqdm


def decode_image(image_data):
    # Try to decode the image data
    try:
        image_data = image_data[()]  # to get the value of the dataset
        img = Image.open(io.BytesIO(image_data))
        return np.array(img)
    except Exception as e:
        print(f"Failed to decode image: {e}")
        return None


def save_images(test_data_hdf5, save_dir):
    with h5py.File(test_data_hdf5, 'r') as file:
        images_indexes = list(file.keys())
        for idx in tqdm(images_indexes):
            img = file[idx]
            img = decode_image(img)
            if img is not None:
                img = Image.fromarray(img)
                img.save(save_dir / f"{idx}.png")


if __name__ == "__main__":
    data_dir = Path("data")
    test_data = data_dir / "test-image.hdf5"
    # save the test images to a folder
    save_dir = Path("tmp/test-images")
    save_dir.mkdir(exist_ok=True)
    save_images(test_data, save_dir)
