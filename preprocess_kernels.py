#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    preprocess_kernels.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        18/05/2023 12:34

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import logging
import os

# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

def create_plot(data_array, description, cmap, output_dir):

    try:
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

    except Exception as e:
        logger.error("Error occurred in create_plot function", exc_info=True)

input_dir = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/VIIRS_prior/'
index_dir = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/'
index_file = index_dir + 'Nairobi_validation_source_index_resampling_mode_ovr_none.tif'

def preprocess_kernels(input_dir, index_file, logger):

    logger.info('Entered preprocess_kernels function')
    try:
        index_dataset = gdal.Open(index_file)
        index_data_array = index_dataset.GetRasterBand(1).ReadAsArray()
        input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")], key=lambda x: int(os.path.basename(x).split('_')[0]))

        input_file = gdal.Open(input_files[0])
        rows, cols = input_file.RasterYSize, input_file.RasterXSize
        geotransform = input_file.GetGeoTransform()
        projection = input_file.GetProjection()

        temp_arrays = []
        band_descriptions = []

        for input_file in input_files:
            dataset = gdal.Open(input_file)
            temp_bands = []

            band_descriptions = []
            for band_num in range(1, 37):
                band = dataset.GetRasterBand(band_num)
                data_array = band.ReadAsArray()
                data_array = data_array.astype(float)
                data_array[data_array == 0] = np.nan
                temp_bands.append(data_array)
                band_description = band.GetDescription()
                band_descriptions.append(band_description)
            temp_arrays.append(temp_bands)

        for band_num in range(1, 37):
            output_data_array = np.zeros((rows, cols), dtype=np.float32)
            band_description = band_descriptions[band_num - 1]
            for i in range(rows):
                for j in range(cols):
                    index = int(index_data_array[i, j])

                    try:
                        temp_array = temp_arrays[index]
                    except IndexError:
                        logger.error(f"Index error: 'index' {index} is out of range.")
                        continue

                    try:
                        temp_band_array = temp_array[band_num - 1]
                    except IndexError:
                        logger.error(f"Index error: 'band_num - 1' {band_num - 1} is out of range.")
                        continue

                    try:
                        output_data_array[i, j] = temp_band_array[i, j]
                    except IndexError:
                        logger.error(f"Index error: 'i, j' {i, j} is out of range.")
                        continue

                    except IndexError as ie:
                        logger.error(
                            f"Index error at row {i}, col {j}, index {index}, band_num {band_num}. Error: {str(ie)}")

            logger.info('Completed preprocess kernels for mosaic band %s'%band_description)
            np.save(input_dir + f"mosaic_band_{band_description}.npy", output_data_array)
            create_plot(output_data_array, band_description, 'rainbow', input_dir)

    except Exception as e:
        logger.error("Error occurred in preprocess_kernels function", exc_info=True)