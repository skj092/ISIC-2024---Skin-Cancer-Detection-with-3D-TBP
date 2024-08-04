import h5py
from tqdm import tqdm

def combine_hdf5_files(file1_path, file2_path, output_path):
    with h5py.File(file1_path, 'r') as file1, h5py.File(file2_path, 'r') as file2, h5py.File(output_path, 'w') as output_file:
        # Get total number of datasets
        total_datasets = len(file1.keys()) + len(file2.keys())
        
        with tqdm(total=total_datasets, desc="Combining HDF5 files") as pbar:
            # Copy contents from the first file
            for key in file1.keys():
                file1.copy(key, output_file)
                pbar.update(1)
            
            # Copy contents from the second file
            for key in file2.keys():
                if key in output_file:
                    print(f"Warning: Dataset '{key}' already exists in the output file. Skipping...")
                else:
                    file2.copy(key, output_file)
                pbar.update(1)

    print(f"Combined HDF5 file saved to {output_path}")

# Usage example
file1_path = "data/train-image.hdf5"
file2_path = 'tmp/image_256sq.hdf5'
output_path = 'data/combined_file.h5'

combine_hdf5_files(file1_path, file2_path, output_path)