import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

def create_subsets_from_merged(scene_name, input_file, output_dir, subset_size=256):
    """
    Splits a merged Landsat GeoTIFF into 256x256 subsets, with padding if needed, and saves each subset as a separate GeoTIFF.

    Parameters:
        scene_name (str): Name of the Landsat scene (prefix).
        input_file (str): Path to the merged GeoTIFF file.
        output_dir (str): Directory where the subsets will be saved.
        subset_size (int): Size of each subset (default: 256).
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0]
        count = src.count

        # Calculate number of rows and columns of subsets
        num_rows = (height + subset_size - 1) // subset_size
        num_cols = (width + subset_size - 1) // subset_size

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the window bounds
                row_start = row * subset_size
                col_start = col * subset_size

                row_end = min((row + 1) * subset_size, height)
                col_end = min((col + 1) * subset_size, width)

                window_height = row_end - row_start
                window_width = col_end - col_start

                # Read data from the window
                window = Window(col_start, row_start, window_width, window_height)
                data = src.read(window=window, boundless=True, fill_value=0)

                # Adjust transform for the subset
                subset_transform = src.window_transform(window)

                # Apply padding if necessary
                padded_data = np.zeros((count, subset_size, subset_size), dtype=dtype)
                padded_data[:, :window_height, :window_width] = data

                # Create the output file name
                output_file = os.path.join(output_dir, f"{scene_name}_{row}_{col}.tif")

                # Save the subset as a GeoTIFF
                with rasterio.open(
                    output_file,
                    "w",
                    driver="GTiff",
                    height=subset_size,
                    width=subset_size,
                    count=count,
                    dtype=dtype,
                    crs=crs,
                    transform=subset_transform,
                ) as dst:
                    dst.write(padded_data)

                print(f"Saved subset: {output_file}")

# Example usage
if __name__ == "__main__":
# Example usage
    scene_names = [
        "LT05_L2SP_008046_20100122_20200825_02_T1_SR",
        "LT05_L2SP_009046_20070918_20200829_02_T1_SR"
    ]
# imagery subset
    input_directory = "./imagery_merge"  # Directory containing Landsat GeoTIFF files
    output_directory = "./imagery_subsets"  # Directory to save the subsets
    for scene_name in scene_names:
        input_file=os.path.join(input_directory,f'{scene_name}_merged.tif')
        create_subsets_from_merged(scene_name, input_file, output_directory)
# mask subset 
    input_directory = "./Mask"  # Directory containing Landsat GeoTIFF files
    output_directory = "./mask_subsets"  # Directory to save the subsets
    for scene_name in scene_names:
        input_file=os.path.join(input_directory,f'{scene_name}_mask.TIF')
        create_subsets_from_merged(scene_name, input_file, output_directory)