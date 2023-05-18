import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

def create_plot(data_array, description):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_array, cmap="gray")

    ax.set_title(f"{description}")
    ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Y-axis Label")
    plt.xticks(np.arange(0, data_array.shape[1], 50))
    plt.yticks(np.arange(0, data_array.shape[0], 50))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Data Values")

    return fig

input_dir = "/Users/rs/Projects/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC_VIIRS_NEW/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/VIIRS_prior/"
index_dir = "/Users/rs/Projects/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC_VIIRS_NEW/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/"

os.system(f"gdal_translate -tr 500 500 {index_dir}validation_source_index.tif {index_dir}validation_source_index_500.tif")
index_file = index_dir + 'validation_source_index_500.tif'
index_dataset = gdal.Open(index_file)

for band_num in range(1, index_dataset.RasterCount + 1):
    band = index_dataset.GetRasterBand(band_num)
    data_array = band.ReadAsArray()
    print(data_array)
    data_array = data_array.astype(float)
    data_array[data_array == 0] = np.nan

    output_dir = os.path.join(os.getcwd(), "output_pngs", os.path.splitext(os.path.basename(index_file))[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "indexing_500m.png")
    fig = create_plot(data_array, 'indexing_500m')
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved band {band_num} as {output_file}")

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".tif"):
            file_path = os.path.join(root, file)
            dataset = gdal.Open(file_path)

            for band_num in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(band_num)
                description = band.GetDescription()
                data_array = band.ReadAsArray()

                data_array = data_array.astype(float)
                data_array[data_array == 0] = np.nan

                output_dir = os.path.join(os.getcwd(), "output_pngs", os.path.splitext(file)[0])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file = os.path.join(output_dir, f"{description}.png")
                fig = create_plot(data_array, description)
                fig.savefig(output_file, dpi=300)
                plt.close(fig)
                print(f"Saved band {band_num} for {file} as {output_file}")
