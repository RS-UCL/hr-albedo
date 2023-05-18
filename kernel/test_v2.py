import os
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from matplotlib.colors import ListedColormap

def create_plot(data_array, description, cmap, output_dir):

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_array, cmap=cmap)

    ax.set_title(f"{description}")
    ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Y-axis Label")
    plt.xticks(np.arange(0, data_array.shape[1], 50))
    plt.yticks(np.arange(0, data_array.shape[0], 50))

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Data Values")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    output_file = os.path.join(output_dir, f"{description}.png")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    
# # local diretory
# input_dir = "/Users/rs/Projects/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC_VIIRS_NEW/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/VIIRS_prior/"
# index_dir = "/Users/rs/Projects/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC_VIIRS_NEW/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/"
# remote directory
input_dir = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/VIIRS_prior/'
index_dir = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/'

# os.system(f"gdal_translate -r mode -ovr NONE -tr 500 500 {index_dir}validation_source_index.tif {index_dir}validation_source_index_500.tif")
# index_file = index_dir + 'validation_source_index_500.tif'
index_file = index_dir + 'Nairobi_validation_source_index_resampling_mode_ovr_none.tif'

index_dataset = gdal.Open(index_file)
index_data_array = index_dataset.GetRasterBand(1).ReadAsArray()

input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")], key=lambda x: int(os.path.basename(x).split('_')[0]))
print(input_files)
# Assuming all input files have the same dimensions
input_file = gdal.Open(input_files[0])
rows, cols = input_file.RasterYSize, input_file.RasterXSize
geotransform = input_file.GetGeoTransform()
projection = input_file.GetProjection()
print(456)
output_file = "./new_file.tif"
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(output_file, cols, rows, 36, gdal.GDT_Float32)

dst_ds.SetGeoTransform(geotransform)
dst_ds.SetProjection(projection)

temp_arrays = []
print(123)
for input_file in input_files:
    dataset = gdal.Open(input_file)
    temp_bands = []

    for band_num in range(1, 37):
        band = dataset.GetRasterBand(band_num)
        data_array = band.ReadAsArray()
        data_array = data_array.astype(float)
        data_array[data_array == 0] = np.nan
        temp_bands.append(data_array)

    temp_arrays.append(temp_bands)

for band_num in range(1, 37):
    output_data_array = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            index = int(index_data_array[i, j])

            output_data_array[i, j] = temp_arrays[index][band_num - 1][i, j]
            print(i, j, index, temp_arrays[index][band_num - 1][i, j])
    dst_band = dst_ds.GetRasterBand(band_num)
    dst_band.WriteArray(output_data_array)
    dst_band.SetNoDataValue(np.nan)
    dst_band.FlushCache()

dst_ds = None

for input_file in input_files:
    dataset = gdal.Open(input_file)
    output_dir = os.path.join(os.getcwd(), "output_pngs", os.path.splitext(os.path.basename(input_file))[0])

    for band_num in range(1, 37):
        if band_num == 1:
            band = dataset.GetRasterBand(band_num)
            description = f"{os.path.splitext(os.path.basename(input_file))[0]}_Band_{band_num}"
            data_array = band.ReadAsArray()
            create_plot(data_array, description, 'rainbow', output_dir)

colors = plt.cm.tab20(np.linspace(0, 1, 13))  # Replace 10 with the desired number of color bins
discrete_cmap = ListedColormap(colors)

# Plotting and saving index file
index_data_array = index_dataset.GetRasterBand(1).ReadAsArray()
index_output_dir = os.path.join(os.getcwd(), "output_pngs", os.path.splitext(os.path.basename(index_file))[0])
create_plot(index_data_array, "indexing_500m", discrete_cmap, index_output_dir)

# Plotting and saving merged file bands
merged_dataset = gdal.Open(output_file)
merged_output_dir = os.path.join(os.getcwd(), "output_pngs", os.path.splitext(os.path.basename(output_file))[0])

for band_num in range(1, 37):
    band = merged_dataset.GetRasterBand(band_num)
    description = f"Merged_Band_{band_num}"
    data_array = band.ReadAsArray()
    create_plot(data_array, description, 'rainbow', merged_output_dir)