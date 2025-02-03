import os
import numpy as np
import rasterio
from rasterio.transform import Affine

def merge_scene_bands_to_geotiff(scene_name, input_dir, output_dir):
    """
    Merges Landsat bands for a single scene and saves the merged array as a multi-band GeoTIFF file.

    Parameters:
        scene_name (str): Name of the Landsat scene (prefix).
        input_dir (str): Directory containing Landsat GeoTIFF files.
        output_dir (str): Directory where the merged GeoTIFF file will be saved.

    Returns:
        str: Path to the saved GeoTIFF file.
    """
    band_indices = [1, 2, 3, 4, 5, 7]  # Channels to include
    merged_data = []
    transform = None
    crs = None

    for band_idx in band_indices:
        file_name = f"{scene_name}_B{band_idx}.TIF"
        file_path = os.path.join(input_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Band file not found: {file_path}")

        with rasterio.open(file_path) as src:
            band_data = src.read(1)  # Read the first band
            merged_data.append(band_data)
            if transform is None:
                transform = src.transform
                crs = src.crs

    # Stack all bands into a 3D array (bands, height, width)
    merged_array = np.stack(merged_data, axis=0)

    # Save the merged array as a GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}_merged.tif")
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=merged_array.shape[1],  # Height of the raster
        width=merged_array.shape[2],  # Width of the raster
        count=merged_array.shape[0],  # Number of bands
        dtype=merged_array.dtype,  # Data type (e.g., float32 or uint16)
        crs=crs,  # Coordinate reference system
        transform=transform  # Georeferencing transform
    ) as dst:
        for i in range(merged_array.shape[0]):
            dst.write(merged_array[i, :, :], i + 1)  # Write each band to the GeoTIFF

    print(f"Merged GeoTIFF for {scene_name} saved to {output_path}")
    return output_path


def process_all_scenes_to_geotiff(scene_names, input_dir, output_dir):
    """
    Processes all Landsat scenes by merging their bands and saving to GeoTIFF files.

    Parameters:
        scene_names (list of str): List of Landsat scene names (prefixes).
        input_dir (str): Directory containing Landsat GeoTIFF files.
        output_dir (str): Directory where the merged GeoTIFF files will be saved.
    """
    for scene_name in scene_names:
        try:
            merge_scene_bands_to_geotiff(scene_name, input_dir, output_dir)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")




if __name__ == "__main__":
# Example usage
    scene_names = [
        "LT05_L2SP_008046_20100122_20200825_02_T1_SR",
        "LT05_L2SP_009046_20070918_20200829_02_T1_SR"
    ]

    input_directory = "./imagery"  # Directory containing Landsat GeoTIFF files
    output_directory = "./imagery_merge"  # Directory to save .npy files

process_all_scenes_to_geotiff(scene_names, input_directory, output_directory)