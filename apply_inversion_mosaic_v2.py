#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    apply_inversion_mosaic_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        18/05/2023 16:59

from sklearn.linear_model import LinearRegression
from pysptools.abundance_maps import FCLS
import matplotlib.pyplot as plt
from scipy import interpolate
from osgeo import gdal
import numpy as np
import glob
import os

def apply_inversion(sentinel2_directory, patch_size, patch_overlap):
    """
    :param sentinel2_file: directory and filename of Sentinel-2 .SAFE format data.
    :param mcd43a1_file: directory and filename of corresponding MCD43A1 data.
    :param patch_size: retrieval patch size.
    :param patch_overlap: retrieval patch overlap.
    """
    file_subdirectory = sentinel2_directory
    tbd_directory = file_subdirectory + '/tbd'  # temporal directory, to be deleted in the end.
    fig_directory = file_subdirectory + '/Figures'  # temporal directory, to be deleted in the end.

    s2_500m_abundance = np.load('%s/s2_500m_abundance.npy' % tbd_directory)

    # Sentinel-2 band to be retrieved
    inverse_band_id = ['02', '03', '04', '8A', '11', '12']

    # matrix to store the regression coefficients
    dhr_coef_a = np.zeros((9, 4))
    dhr_coef_b = np.zeros((9, 4))
    bhr_coef_a = np.zeros((9, 4))
    bhr_coef_b = np.zeros((9, 4))

    # build the albedo-to-brf ratio based regression
    for m in range(len(inverse_band_id)):

        modis_brf = np.load('%s/brf_band%s.npy' % (tbd_directory, inverse_band_id[m]))
        modis_brf[modis_brf < 0] = 0
        modis_dhr = np.load('%s/dhr_band%s.npy' % (tbd_directory, inverse_band_id[m]))
        modis_dhr[modis_dhr < 0] = 0
        modis_bhr = np.load('%s/bhr_band%s.npy' % (tbd_directory, inverse_band_id[m]))
        modis_bhr[modis_bhr < 0] = 0

        ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # colorbar settings
        for i in range(4):
            # loop for dhr regressions.
            threshold = 0.5

            endmember_brf = modis_brf[s2_500m_abundance[:, :, i].reshape(modis_brf.shape) > threshold]
            endmember_dhr = modis_dhr[s2_500m_abundance[:, :, i].reshape(modis_dhr.shape) > threshold]

            x1 = endmember_brf.reshape(endmember_brf.size, 1)
            y1 = endmember_dhr.reshape(endmember_dhr.size, 1)

            x1_filter = x1[(x1 > 0) & (y1 > 0)]
            y1_filter = y1[(x1 > 0) & (y1 > 0)]

            if x1_filter.size > 0:
                x1_filter = x1_filter.reshape((x1_filter.size, 1))
                y1_filter = y1_filter.reshape((y1_filter.size, 1))

                model_a1 = LinearRegression()
                model_a1.fit(x1_filter, y1_filter)
                x1_new = np.linspace(0, 0.8, 20)
                y1_new = model_a1.predict(x1_new[:, np.newaxis])

                dhr_coef_a[m, i] = model_a1.intercept_[0]
                dhr_coef_b[m, i] = model_a1.coef_[0][0]

                colors = ['blue', 'green', 'red', 'black']
                fig, ax = plt.subplots(figsize=(16, 16))

                plt.scatter(endmember_brf.reshape(endmember_brf.size, 1), endmember_dhr.reshape(endmember_dhr.size, 1),
                            color=colors[i], alpha=0.3)
                plt.plot(x1_new, y1_new, lw=2, color=colors[i])
                plt.text(0.025, 0.175, r'$albedo = %.3f$ + %.3f*BRF' % (model_a1.intercept_[0], model_a1.coef_[0][0]),
                         fontsize=26)

                plt.xlim([0., .8])
                plt.ylim([0., .8])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(22)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(22)

                plt.xlabel('BRF', fontsize=26)
                plt.ylabel('DHR', fontsize=26)
                plt.savefig('%s/dhr_to_brf_band%s_type%s.png' % (fig_directory, inverse_band_id[m], ascii_uppercase[i]))
                plt.close()

            else:
                dhr_coef_a[m, i] = 0
                dhr_coef_b[m, i] = 1.

        for i in range(4):
            # loop for bhr regressions.
            threshold = 0.5

            endmember_brf = modis_brf[s2_500m_abundance[:, :, i].reshape(modis_brf.shape) > threshold]
            endmember_bhr = modis_bhr[s2_500m_abundance[:, :, i].reshape(modis_bhr.shape) > threshold]

            x1 = endmember_brf.reshape(endmember_brf.size, 1)
            y1 = endmember_bhr.reshape(endmember_bhr.size, 1)
            x1_filter = x1[(x1 > 0) & (y1 > 0)]
            y1_filter = y1[(x1 > 0) & (y1 > 0)]

            x1_filter = x1_filter.reshape((x1_filter.size, 1))
            y1_filter = y1_filter.reshape((y1_filter.size, 1))

            if x1_filter.size > 0:
                model_a1 = LinearRegression()
                model_a1.fit(x1_filter, y1_filter)
                x1_new = np.linspace(0, 0.35, 20)
                y1_new = model_a1.predict(x1_new[:, np.newaxis])

                bhr_coef_a[m, i] = model_a1.intercept_[0]
                bhr_coef_b[m, i] = model_a1.coef_[0][0]

                colors = ['blue', 'green', 'red', 'black']
                fig, ax = plt.subplots(figsize=(16, 16))

                plt.scatter(endmember_brf.reshape(endmember_brf.size, 1), endmember_bhr.reshape(endmember_bhr.size, 1),
                            color=colors[i], alpha=0.3)
                plt.plot(x1_new, y1_new, lw=2, color=colors[i])
                plt.text(0.025, 0.175, r'$albedo = %.3f$ + %.3f*BRF' % (model_a1.intercept_[0], model_a1.coef_[0][0]),
                         fontsize=26)

                plt.xlim([0., .8])
                plt.ylim([0., .8])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(22)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(22)

                plt.xlabel('BRF', fontsize=26)
                plt.ylabel('BHR', fontsize=26)
                plt.savefig('%s/bhr_to_brf_band%s_type%s.png' % (fig_directory, inverse_band_id[m], ascii_uppercase[i]))
            else:
                bhr_coef_a[m, i] = 0
                bhr_coef_a[m, i] = 1.

    band02_20m = None
    band03_20m = None
    band04_20m = None
    band8A_20m = None
    band11_20m = None
    band12_20m = None

    for file in os.listdir(file_subdirectory):
        if file.endswith("B02.tif"):
            band02_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band02_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B03.tif"):
            band03_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B04.tif"):
            band04_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B8A.tif"):
            band8A_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B11.tif"):
            band11_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B12.tif"):
            band12_10m = gdal.Open('%s/%s' % (file_subdirectory, file))

    # get Sentinel-2 20m and 10m proj
    geotransform_10m = band02_10m.GetGeoTransform()
    proj_10m = band02_10m.GetProjection()

    s2_10m_cols = band02_10m.RasterXSize
    s2_10m_rows = band02_10m.RasterYSize
    print(s2_10m_cols, s2_10m_rows)

    # Sentinel-2 reflectance data scaling factor
    s2_scaling_factor = 1.e4

    boa_band02_10m = band02_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band03_10m = band03_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band04_10m = band04_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band8A_10m = band8A_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band11_10m = band11_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band12_10m = band12_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor

    boa_band02_10m = band02_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band03_10m = band03_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor
    boa_band04_10m = band04_10m.GetRasterBand(1).ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / s2_scaling_factor

    num_row = np.floor(s2_10m_rows / (patch_size - patch_overlap)) + 1
    col_row = np.floor(s2_10m_cols / (patch_size - patch_overlap)) + 1

    parallel_process(num_row=int(num_row), col_row=int(col_row), pool_size=multiprocessing.cpu_count())

def apply_uncertainty(sentinel2_directory):
    """

    :param sentinel2_file:  directory and filename of Sentinel-2 .SAFE format data.
    :return:
    """
    # retrieve 10m/ albedo uncertainties are derived from SIAC output uncertainties.

    file_subdirectory = sentinel2_directory

    tbd_directory = file_subdirectory + '/tbd'  # temporal directory, to be deleted in the end.
    fig_directory = file_subdirectory + '/Figures'  # temporal directory, to be deleted in the end.

    # Sentinel-2 band to be retrieved
    inverse_band_id = ['02', '03', '04', 'VIS', 'NIR', 'SW', '8A', '11', '12']

    for band in ['02', '03', '04', '8A', '11', '12']:
        if band in inverse_band_id:
            for file in os.listdir(f"{file_subdirectory}/"):
                if file.endswith(f"B{band}.tif"):
                    boa_band = gdal.Open(f"{file_subdirectory}/{file}")
                    cols_i, rows_i = boa_band.RasterXSize, boa_band.RasterYSize
                    boa_band_array = boa_band.GetRasterBand(1).ReadAsArray(0, 0, cols_i, rows_i) / 1.e4

                    boa_band_unc = gdal.Open(f"{file_subdirectory}/{file[:-4]}_unc.tif")
                    cols_i, rows_i = boa_band_unc.RasterXSize, boa_band_unc.RasterYSize
                    boa_band_unc_array = boa_band_unc.GetRasterBand(1).ReadAsArray(0, 0, cols_i, rows_i) / 1.e4

                    print(
                        f"-----------> Mean B{band} boa reflectance is: {np.mean(boa_band_array[boa_band_array > 0])} -------")
                    print(
                        f"-----------> Mean B{band} boa reflectance uncertainty is: {np.mean(boa_band_unc_array[boa_band_unc_array > 0])} -------")

                    unc_relative = boa_band_unc_array / boa_band_array
                    print(
                        f"-----------> Mean B{band} relative uncertainty is: {np.mean(unc_relative[unc_relative < 1.])} -------")
                    np.save(f"{tbd_directory}/unc_relative_B{band}.npy", unc_relative)

    for file in os.listdir('%s/' % file_subdirectory):
        if file.endswith("B02.tif"):
            band02_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band02_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B03.tif"):
            band03_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band03_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B04.tif"):
            band04_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band04_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B8A.tif"):
            band8A_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band8A_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B11.tif"):
            band11_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band11_10m_file = '%s/%s' % (file_subdirectory, file)
        if file.endswith("B12.tif"):
            band12_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
            band12_10m_file = '%s/%s' % (file_subdirectory, file)

    ulx, xres, xskew, uly, yskew, yres = band02_10m.GetGeoTransform()
    lrx = ulx + (band02_10m.RasterXSize * xres)
    lry = uly + (band02_10m.RasterYSize * yres)

    s2_10m_cols = band02_10m.RasterXSize
    s2_10m_rows = band02_10m.RasterYSize

    boa_band02_10m = band02_10m.GetRasterBand(1)
    boa_band03_10m = band03_10m.GetRasterBand(1)
    boa_band04_10m = band04_10m.GetRasterBand(1)
    boa_band8A_10m = band8A_10m.GetRasterBand(1)
    boa_band11_10m = band11_10m.GetRasterBand(1)
    boa_band12_10m = band12_10m.GetRasterBand(1)

    boa_band02_10m = boa_band02_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band03_10m = boa_band03_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band04_10m = boa_band04_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band8A_10m = boa_band8A_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band11_10m = boa_band11_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band12_10m = boa_band12_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4

    for file in os.listdir('%s/' % file_subdirectory):
        if file.endswith("B02_unc.tif"):
            band02_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B03_unc.tif"):
            band03_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B04_unc.tif"):
            band04_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B8A_unc.tif"):
            band8A_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B11_unc.tif"):
            band11_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))
        if file.endswith("B12_unc.tif"):
            band12_unc_10m = gdal.Open('%s/%s' % (file_subdirectory, file))

    boa_band02_unc_10m = band02_unc_10m.GetRasterBand(1)
    boa_band03_unc_10m = band03_unc_10m.GetRasterBand(1)
    boa_band04_unc_10m = band04_unc_10m.GetRasterBand(1)
    boa_band8A_unc_10m = band8A_unc_10m.GetRasterBand(1)
    boa_band11_unc_10m = band11_unc_10m.GetRasterBand(1)
    boa_band12_unc_10m = band12_unc_10m.GetRasterBand(1)

    boa_band02_unc_10m = boa_band02_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band03_unc_10m = boa_band03_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band04_unc_10m = boa_band04_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band8A_unc_10m = boa_band8A_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band11_unc_10m = boa_band11_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4
    boa_band12_unc_10m = boa_band12_unc_10m.ReadAsArray(0, 0, s2_10m_cols, s2_10m_rows) / 1.e4

    # narrow to broadband conversion coefficients, available from: https://doi.org/10.1016/j.rse.2018.08.025
    VIS_coefficient = [-0.0048, 0.5673, 0.1407, 0.2359, 0, 0, 0]
    SW_coefficient = [-0.0049, 0.2688, 0.0362, 0.1501, 0.3045, 0.1644, 0.0356]
    NIR_coefficient = [-0.0073, 0., 0., 0., 0.5595, 0.3844, 0.0290]

    boa_bandVIS_10m = VIS_coefficient[0] + VIS_coefficient[1] * boa_band02_10m + VIS_coefficient[
        2] * boa_band03_10m + VIS_coefficient[3] * boa_band04_10m + VIS_coefficient[4] * boa_band8A_10m + \
                      VIS_coefficient[5] * boa_band11_10m + VIS_coefficient[6] * boa_band12_10m
    boa_bandSW_10m = SW_coefficient[0] + SW_coefficient[1] * boa_band02_10m + SW_coefficient[
        2] * boa_band03_10m + SW_coefficient[3] * boa_band04_10m + SW_coefficient[4] * boa_band8A_10m + \
                     SW_coefficient[5] * boa_band11_10m + SW_coefficient[6] * boa_band12_10m
    boa_bandNIR_10m = NIR_coefficient[0] + NIR_coefficient[1] * boa_band02_10m + NIR_coefficient[
        2] * boa_band03_10m + NIR_coefficient[3] * boa_band04_10m + NIR_coefficient[4] * boa_band8A_10m + \
                      NIR_coefficient[5] * boa_band11_10m + NIR_coefficient[6] * boa_band12_10m

    boa_bandVIS_unc_10m = np.sqrt(
        (VIS_coefficient[1] * boa_band02_unc_10m) ** 2 + (VIS_coefficient[2] * boa_band03_unc_10m) ** 2 + (
                VIS_coefficient[3] * boa_band04_unc_10m) ** 2 + (
                VIS_coefficient[4] * boa_band8A_unc_10m) ** 2 + (
                VIS_coefficient[5] * boa_band11_unc_10m) ** 2 + (
                VIS_coefficient[6] * boa_band12_unc_10m) ** 2)
    boa_bandSW_unc_10m = np.sqrt(
        (SW_coefficient[1] * boa_band02_unc_10m) ** 2 + (SW_coefficient[2] * boa_band03_unc_10m) ** 2 + (
                SW_coefficient[3] * boa_band04_unc_10m) ** 2 + (
                SW_coefficient[4] * boa_band8A_unc_10m) ** 2 + (
                SW_coefficient[5] * boa_band11_unc_10m) ** 2 + (
                SW_coefficient[6] * boa_band12_unc_10m) ** 2)
    boa_bandNIR_unc_10m = np.sqrt(
        (NIR_coefficient[1] * boa_band02_unc_10m) ** 2 + (NIR_coefficient[2] * boa_band03_unc_10m) ** 2 + (
                NIR_coefficient[3] * boa_band04_unc_10m) ** 2 + (
                NIR_coefficient[4] * boa_band8A_unc_10m) ** 2 + (
                NIR_coefficient[5] * boa_band11_unc_10m) ** 2 + (
                NIR_coefficient[6] * boa_band12_unc_10m) ** 2)

    VIS_unc_relative = boa_bandVIS_unc_10m / boa_bandVIS_10m
    SW_unc_relative = boa_bandSW_unc_10m / boa_bandSW_10m
    NIR_unc_relative = boa_bandNIR_unc_10m / boa_bandNIR_10m

    np.save(tbd_directory + '/unc_relative_BVIS.npy', VIS_unc_relative)
    np.save(tbd_directory + '/unc_relative_BSW.npy', SW_unc_relative)
    np.save(tbd_directory + '/unc_relative_BNIR.npy', NIR_unc_relative)

    print('-----------> Mean VIS albedo is: %s.' % (np.mean(boa_bandVIS_10m[boa_bandVIS_10m > 0])))
    print('-----------> Mean VIS albedo uncertainty is: %s.' % (np.mean(boa_bandVIS_unc_10m[boa_bandVIS_unc_10m > 0])))
    print('-----------> Mean SW albedo is: %s.' % (np.mean(boa_bandSW_10m[boa_bandSW_10m > 0])))
    print('-----------> Mean SW albedo uncertainty is: %s.' % (np.mean(boa_bandSW_unc_10m[boa_bandSW_unc_10m > 0])))
    print('-----------> Mean NIR albedo is: %s.' % (np.mean(boa_bandNIR_10m[boa_bandNIR_10m > 0])))
    print('-----------> Mean NIR albedo uncertainty is: %s.' % (np.mean(boa_bandNIR_unc_10m[boa_bandNIR_unc_10m > 0])))

