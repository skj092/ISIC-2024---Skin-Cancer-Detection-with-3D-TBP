from pathlib import Path
import h5py
from PIL import Image
import numpy as np
import io
from tqdm import tqdm
import joblib

def decode_and_save_image(args):
    idx, image_path, save_dir = args
    try:
        with h5py.File(image_path, 'r') as file:
            image_data = file[idx][()]
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        img.save(save_dir / f"{idx}.png")
        return True
    except Exception as e:
        print(f"Failed to decode or save image {idx}: {e}")
        return False

def save_images_parallel(test_data_hdf5, save_dir, n_jobs=-1):
    with h5py.File(test_data_hdf5, 'r') as file:
        images_indexes = list(file.keys())

    args_list = [(idx, test_data_hdf5, save_dir) for idx in images_indexes]

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(decode_and_save_image)(args) for args in tqdm(args_list)
    )

    successful = sum(results)
    print(f"Successfully processed {successful} out of {len(results)} images.")

if __name__ == "__main__":
    data_dir = Path("data")
    test_data = data_dir / "test-image.hdf5"
    save_dir = Path("tmp/test-images")
    save_dir.mkdir(exist_ok=True, parents=True)

    save_images_parallel(test_data, save_dir)
