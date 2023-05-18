#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    cal_endmember_mosaic_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        13/04/2023 18:06

from pysptools.abundance_maps import FCLS
from pysptools.eea import NFINDR
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from osgeo import gdal
import os

def _plot_2d_abundance(abundane_array, fig_directory, colortablbe):
    """
    :param abundane_array: 2D abundance array
    :return:
    """
    fig, ax = plt.subplots(figsize=(20, 18))
    plt.imshow(abundane_array, cmap='jet')
    cbar = plt.colorbar(shrink=0.5, extend='both')
    cbar.set_label('Abundance', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize=26)
    plt.ylabel('Pixels', fontsize=26)

    plt.savefig('%s/abundance_type_%s.png' % (fig_directory, colortablbe))
    plt.close()

def _plot_solar_angluar(angle_array, save_fig_name):
    """
    :param angle_array: 2D solar angle array
    :param save_fig_name: saved fig name
    :return:
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.imshow(angle_array)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    plt.colorbar(shrink=0.7, extend='both')
    plt.xlabel('Pixels', fontsize=26)
    plt.ylabel('Pixels', fontsize=26)
    plt.title('SAA', fontsize=26)
    plt.savefig('%s' % save_fig_name)
    plt.close()

def _plot_instrument_angluar(angle_array, fig_title, save_fig_name):
    """
    :param angle_array: 2D instrument angle array
    :param fig_title: figure title
    :param save_fig_name: saved fig name
    :return:
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.imshow(angle_array, vmin=160, vmax=320)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    plt.colorbar(shrink=0.7, extend='both')
    plt.xlabel('Pixels', fontsize=26)
    plt.ylabel('Pixels', fontsize=26)
    plt.title('%s' % fig_title, fontsize=26)
    plt.savefig('%s' %save_fig_name)
    plt.close()

def _plot_kernel(kernel_array, fig_title, save_fig_name):

    fig, ax = plt.subplots(figsize=(16, 16))
    plt.imshow(kernel_array)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    plt.colorbar(shrink=0.7, extend='both')
    plt.xlabel('Pixels', fontsize=26)
    plt.ylabel('Pixels', fontsize=26)
    plt.title('%s' % fig_title, fontsize=26)
    plt.savefig('%s' %save_fig_name)
    plt.close()

def _plot_2d_brf(brf_array, save_fig_name):
    """
    :param brf_array: 2D BRF array
    :param save_fig_name: saved fig name
    :return:
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.imshow(brf_array, cmap='jet', vmin=0., vmax=0.3)
    cbar = plt.colorbar(shrink=0.5, extend='both')
    cbar.set_label('BRF', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize=26)
    plt.ylabel('Pixels', fontsize=26)

    plt.savefig('%s' % save_fig_name)
    plt.close()

def brdf_f1(sza, vza, phi):
    """
    :param sza: solar zenith angle
    :param vza: solar azimuth angle
    :param phi: phi angle value
    :return:
    """
    sza = np.deg2rad(sza)
    vza = np.deg2rad(vza)
    phi = np.deg2rad(phi)
    parameter_1 = 1. / (2. * np.pi) * ((np.pi - phi) * np.cos(phi) + np.sin(phi)) * np.tan(sza) * np.tan(vza)
    parameter_2 = 1. / np.pi * (np.tan(sza) + np.tan(vza) + np.sqrt(
        np.tan(sza) ** 2 + np.tan(vza) ** 2 - 2 * np.tan(sza) * np.tan(vza) * np.cos(phi)))

    return (parameter_1 - parameter_2)

# BRDF function 2
def brdf_f2(sza, vza, phi):
    """
    :param sza: solar zenith angle
    :param vza: solar azimuth angle
    :param phi: phi angle value
    :return:
    """
    sza = np.deg2rad(sza)
    vza = np.deg2rad(vza)
    phi = np.deg2rad(phi)
    ci = np.arccos(np.cos(sza) * np.cos(vza) + np.sin(vza) * np.sin(sza) * np.cos(phi))
    return 4. / (3. * np.pi) / (np.cos(sza) + np.cos(vza)) * ((np.pi / 2. - ci) * np.cos(ci) + np.sin(ci)) - 1. / 3.

def cal_endmember(sentinel2_directory):

    sample_interval = 30
    # file subdirectory in the format of sentinel2_directory/GRANULE/L1C****/IMG_DATA
    file_subdirectory = sentinel2_directory

    # kernel_weights = np.load(file_subdirectory + '/kernel_weights.npz')
    # print(kernel_weights.files)
    # print(kernel_weights['fs'].shape)

    tbd = file_subdirectory + '/tbd'
    if not os.path.exists(tbd):
        os.makedirs(tbd)

    fig_directory = file_subdirectory + '/Figures'  # temporal directory, to be deleted in the end.
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)

    # use gdalwarp from os.system() to process a file in the subdirectory
    for file in os.listdir(file_subdirectory):
        if file.endswith(("B02.tif", "B03.tif", "B04.tif", "B8A.tif", "B11.tif", "B12.tif", "cloud_confidence.tif")):
            os.system(f'gdalwarp -tr 500 500 "{file_subdirectory}/{file}" "{tbd}/{file[:-4]}_500m.tif"')

    for file in os.listdir(file_subdirectory):
        if file.endswith(("B02.tif", "B03.tif", "B04.tif", "B8A.tif", "B11.tif", "B12.tif", "cloud_confidence.tif")):
            os.system(f'gdalwarp -tr 20 20 "{file_subdirectory}/{file}" "{tbd}/{file[:-4]}_20m.tif"')

    cloud_dataset = gdal.Open(file_subdirectory + '/cloud_confidence.tif')
    cloud_raster_band = cloud_dataset.GetRasterBand(1)  # Assuming you want to read the first band
    cloud_raster_data = cloud_raster_band.ReadAsArray()
    # Create a plot using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cloud_raster_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cloud Confidence')
    plt.title('Cloud Confidence Map - Nairobi')
    plt.savefig('%s/cloud_confidence.png' % fig_directory)

    # initialize variables with None for 500-m data
    s2_band02_500m_data = None
    s2_band03_500m_data = None
    s2_band04_500m_data = None
    s2_band8A_500m_data = None
    s2_band11_500m_data = None
    s2_band12_500m_data = None
    s2_mask_500m_data = None

    # initialize variables with None for 20-m data
    s2_band02_20m_data = None
    s2_band03_20m_data = None
    s2_band04_20m_data = None
    s2_band8A_20m_data = None
    s2_band11_20m_data = None
    s2_band12_20m_data = None
    s2_mask_data = None

    for file in os.listdir(tbd):
        if file.endswith("B02_500m.tif"):
            s2_band02_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B03_500m.tif"):
            s2_band03_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B04_500m.tif"):
            s2_band04_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B8A_500m.tif"):
            s2_band8A_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B11_500m.tif"):
            s2_band11_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B12_500m.tif"):
            s2_band12_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("_mask_500m.tif"):
            s2_mask_500m_data = gdal.Open('%s/%s' % (tbd, file))

    for file in os.listdir(tbd):
        if file.endswith("B02_20m.tif"):
            s2_band02_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B03_20m.tif"):
            s2_band03_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B04_20m.tif"):
            s2_band04_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B8A_20m.tif"):
            s2_band8A_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B11_20m.tif"):
            s2_band11_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B12_20m.tif"):
            s2_band12_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("_mask_20m.tif"):
            s2_mask_data = gdal.Open('%s/%s' % (tbd, file))

    # check if variables were assigned
    if s2_band02_500m_data and s2_band03_500m_data and s2_band04_500m_data and s2_band8A_500m_data and s2_band11_500m_data and s2_band12_500m_data:
        # load sentinel-2 500m geo-reference data
        s2_500m_geotransform = s2_band02_500m_data.GetGeoTransform()
        s2_500m_proj = s2_band02_500m_data.GetProjection()
    else:
        # handle the case where one or more variables were not assigned
        print("Error: One or more Sentinel-2 500m bands not found.")

    if s2_band02_20m_data and s2_band03_20m_data and s2_band04_20m_data and s2_band8A_20m_data and s2_band11_20m_data and s2_band12_20m_data:
        # load sentinel-2 20m geo-reference data
        s2_20m_geotransform = s2_band02_20m_data.GetGeoTransform()
        s2_20m_proj = s2_band02_20m_data.GetProjection()
    else:
        # handle the case where one or more variables were not assigned
        print("Error: One or more Sentinel-2 20m bands not found.")

    # get sentinel-2 500m data number of rows and cols
    s2_cols_500m = s2_band02_500m_data.RasterXSize
    s2_rows_500m = s2_band02_500m_data.RasterYSize

    s2_cols_20m = s2_band02_20m_data.RasterXSize
    s2_rows_20m = s2_band02_20m_data.RasterYSize

    # get raster band for 500m data
    boa_band02_500m = s2_band02_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_band03_500m = s2_band03_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_band04_500m = s2_band04_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_band8A_500m = s2_band8A_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_band11_500m = s2_band11_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_band12_500m = s2_band12_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)
    boa_mask_500m = s2_mask_500m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)

    # get raster band for 20m data
    boa_band02_20m = s2_band02_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_band03_20m = s2_band03_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_band04_20m = s2_band04_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_band8A_20m = s2_band8A_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_band11_20m = s2_band11_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_band12_20m = s2_band12_20m_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)
    boa_mask_20m = s2_mask_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_20m, s2_rows_20m)

    boa_band02_20m_resampled = boa_band02_20m[::sample_interval, ::sample_interval]
    boa_band03_20m_resampled = boa_band03_20m[::sample_interval, ::sample_interval]
    boa_band04_20m_resampled = boa_band04_20m[::sample_interval, ::sample_interval]
    boa_band8A_20m_resampled = boa_band8A_20m[::sample_interval, ::sample_interval]
    boa_band11_20m_resampled = boa_band11_20m[::sample_interval, ::sample_interval]
    boa_band12_20m_resampled = boa_band12_20m[::sample_interval, ::sample_interval]
    boa_mask_20m_resampled = boa_mask_20m[::sample_interval, ::sample_interval]

    # convert 2d-array to 1-d array
    boa_band02_array = boa_band02_20m_resampled.reshape(boa_band02_20m_resampled.size, 1)
    boa_band03_array = boa_band03_20m_resampled.reshape(boa_band03_20m_resampled.size, 1)
    boa_band04_array = boa_band04_20m_resampled.reshape(boa_band04_20m_resampled.size, 1)
    boa_band8A_array = boa_band8A_20m_resampled.reshape(boa_band8A_20m_resampled.size, 1)
    boa_band11_array = boa_band11_20m_resampled.reshape(boa_band11_20m_resampled.size, 1)
    boa_band12_array = boa_band12_20m_resampled.reshape(boa_band12_20m_resampled.size, 1)
    boa_mask_array = boa_mask_20m_resampled.reshape(boa_mask_20m_resampled.size, 1)
    print(boa_band02_array)
    quit()
    s2_20m_matrix = np.zeros((boa_band02_array.size, 1, 6))

    s2_20m_matrix[:, 0, 0] = boa_band02_array[:, 0]
    s2_20m_matrix[:, 0, 1] = boa_band03_array[:, 0]
    s2_20m_matrix[:, 0, 2] = boa_band04_array[:, 0]
    s2_20m_matrix[:, 0, 3] = boa_band8A_array[:, 0]
    s2_20m_matrix[:, 0, 4] = boa_band11_array[:, 0]
    s2_20m_matrix[:, 0, 5] = boa_band12_array[:, 0]

    s2_20m_matrix[s2_20m_matrix == -9999.] = np.nan
    s2_20m_matrix = s2_20m_matrix / 1.e4

    # index to filter out cloud pixels
    valid_index = (s2_20m_matrix[:, 0, 0] > 0) & (s2_20m_matrix[:, 0, 1] > 0) & (s2_20m_matrix[:, 0, 2] > 0) & (
                s2_20m_matrix[:, 0, 3] > 0) & (s2_20m_matrix[:, 0, 4] > 0) & (s2_20m_matrix[:, 0, 5] > 0) & (boa_mask_array[:, 0] == 0.)
    s2_20m_matrix = s2_20m_matrix[valid_index, :, :]

    # resample over the sentinel-2 eea spetral wavelengths
    s2_eea_wavelength = np.asarray([459., 560., 665., 865., 1610., 2190.])
    s2_interp_num = 11 # number of pixels to add between two wavelengths

    s2_wv_band02_band03 = np.linspace(459, 560, num=s2_interp_num, endpoint=True)
    s2_wv_band03_band04 = np.linspace(560, 665, num=s2_interp_num, endpoint=True)
    s2_wv_band04_band8A = np.linspace(665, 865, num=s2_interp_num, endpoint=True)
    s2_wv_band11_band12 = np.linspace(1610, 2190, num=2, endpoint=True)

    s2_wv_resampled = np.concatenate((s2_wv_band02_band03[0:-1], s2_wv_band03_band04), axis=0)
    s2_wv_resampled = np.concatenate((s2_wv_resampled[0:-1], s2_wv_band04_band8A), axis=0)
    s2_wv_resampled = np.concatenate((s2_wv_resampled, s2_wv_band11_band12), axis=0)

    # resample the Sentinel-2 matrix over the resampled wavelength range

    func_wv = interpolate.interp1d(s2_eea_wavelength, s2_20m_matrix, axis=2)
    s2_20m_matrix_interp = func_wv(s2_wv_resampled)

    cal_EEA = NFINDR()
    print('-----------> Start calculating end-members based on Sentinel-2 multispectral data.')
    main_endmember = cal_EEA.extract(M=s2_20m_matrix_interp, q=4, maxit=5, normalize=False, ATGP_init=True)
    print("-----------> Finish calculating end-members processing")
    np.save('%s/endmembers.npy' % tbd, main_endmember)

    # display pure-pixel spectra.
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    fig, ax = plt.subplots(figsize=(22, 12))
    for i in range(main_endmember.shape[0]):
        plt.plot(s2_wv_resampled, main_endmember[i, :], '--o', markersize=12, lw=2, label='%s' % ascii_uppercase[i])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    plt.xlabel('Wavelenth [nm]', fontsize=26)
    plt.ylabel('Surface Reflectance', fontsize=26)
    plt.legend(fontsize=26)
    plt.savefig('%s/endmember_spectrum.png' % fig_directory)
    plt.close()

    boa_band02_500m_array = boa_band02_500m.reshape(boa_band02_500m.size, 1)
    boa_band03_500m_array = boa_band03_500m.reshape(boa_band03_500m.size, 1)
    boa_band04_500m_array = boa_band04_500m.reshape(boa_band04_500m.size, 1)
    boa_band8A_500m_array = boa_band8A_500m.reshape(boa_band8A_500m.size, 1)
    boa_band11_500m_array = boa_band11_500m.reshape(boa_band11_500m.size, 1)
    boa_band12_500m_array = boa_band12_500m.reshape(boa_band12_500m.size, 1)
    boa_mask_500m_array = boa_mask_500m.reshape(boa_mask_500m.size, 1)

    s2_500m_matrix = np.zeros((boa_band02_500m_array.size, 1, 6))

    s2_500m_matrix[:, 0, 0] = boa_band02_500m_array[:, 0]
    s2_500m_matrix[:, 0, 1] = boa_band03_500m_array[:, 0]
    s2_500m_matrix[:, 0, 2] = boa_band04_500m_array[:, 0]
    s2_500m_matrix[:, 0, 3] = boa_band8A_500m_array[:, 0]
    s2_500m_matrix[:, 0, 4] = boa_band11_500m_array[:, 0]
    s2_500m_matrix[:, 0, 5] = boa_band12_500m_array[:, 0]

    s2_500m_matrix = s2_500m_matrix / 1.e4

    func_wv_500m = interpolate.interp1d(s2_eea_wavelength, s2_500m_matrix, axis=2)
    s2_resampled_matrix_filtered_interp_500m = func_wv_500m(s2_wv_resampled)

    CalAbundanceMap = FCLS()
    print("-----------> Start calculating abundance on aggregated S2 scence.\n")
    s2_abundance_500m = CalAbundanceMap.map(s2_resampled_matrix_filtered_interp_500m, main_endmember)
    for k in range(s2_abundance_500m.shape[2]):
        s2_abundance_500m[:, :, k][boa_band02_500m_array < 0] = np.nan
        s2_abundance_500m[:, :, k][boa_mask_500m_array > 0.] = np.nan

    print("-----------> Complete calculating abundance on aggregated S2 scence.\n")
    np.save('%s/s2_500m_abundance.npy' % tbd, s2_abundance_500m)

    # plot 2d abundance figures
    for i in range(main_endmember.shape[0]):
        abundane_i = s2_abundance_500m[:, :, i]
        abundane_i = abundane_i.reshape(s2_rows_500m, s2_cols_500m)
        abundane_i[boa_band02_500m_array.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan

        colortable_i = ascii_uppercase[i]
        _plot_2d_abundance(abundane_i, fig_directory, colortable_i)

    # load solar and sensor angular data
    granule_dir = os.path.join(sentinel2_directory, 'GRANULE')
    L1C_dir = os.path.join(granule_dir, os.listdir(granule_dir)[0])
    angular_dir = os.path.join(L1C_dir, 'ANG_DATA')

    for file in os.listdir(angular_dir):
        if file.endswith(("Mean_VAA_VZA.tif")):
            os.system(f'gdalwarp -tr 500 500 "{angular_dir}/{file}" "{tbd}/{file[:-4]}_500m.tif"')

    # get Sentinel-2 coordinates at 500-m resolution
    s2_band02_500m = gdal.Open('%s/Mean_VAA_VZA_500m.tif' % tbd)
    s2_500m_geotransform = s2_band02_500m.GetGeoTransform()

    s2_500m_ymax = s2_500m_geotransform[3]
    s2_500m_ymin = s2_500m_geotransform[3] + s2_500m_geotransform[5] * s2_band02_500m.RasterYSize
    s2_500m_xmin = s2_500m_geotransform[0]
    s2_500m_xmax = s2_500m_geotransform[0] + s2_500m_geotransform[1] * s2_band02_500m.RasterXSize

    for file in os.listdir(angular_dir):
        if file.endswith(("SAA_SZA.tif")):
            print(f'gdalwarp -tr 500 500 -te {s2_500m_xmin} {s2_500m_ymin} {s2_500m_xmax} {s2_500m_ymax} {angular_dir}/{file} {tbd}/{file[:-4]}_500m.tif')
            os.system(f'gdalwarp -tr 500 500 -te {s2_500m_xmin} {s2_500m_ymin} {s2_500m_xmax} {s2_500m_ymax} {angular_dir}/{file} {tbd}/{file[:-4]}_500m.tif')

    solar_500_data = gdal.Open('%s/SAA_SZA_500m.tif' % tbd)
    saa_data = solar_500_data.GetRasterBand(1)
    sza_data = solar_500_data.GetRasterBand(2)
    saa_angle = saa_data.ReadAsArray() / 100.
    sza_angle = sza_data.ReadAsArray() / 100.

    saa_angle[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan
    _plot_solar_angluar(saa_angle, fig_directory + '/saa_angle.png')

    sza_angle[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan
    _plot_solar_angluar(sza_angle, fig_directory + '/sza_angle.png')

    s2_band_id = ['02','03','04','8A','11','12']

    for i in range(len(s2_band_id)):

        sensor_500_data = gdal.Open('%s/Mean_VAA_VZA_500m.tif' %tbd)
        vaa_data = sensor_500_data.GetRasterBand(1)
        vza_data = sensor_500_data.GetRasterBand(2)
        vaa_angle = vaa_data.ReadAsArray() / 100.
        vza_angle = vza_data.ReadAsArray() / 100.

        fig, ax = plt.subplots(figsize=(16, 16))
        vaa_angle[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan
        _plot_instrument_angluar(vaa_angle, 'VAA Band %s' % s2_band_id[i],
                                 '%s/Mean_VAA_500m.png' %fig_directory)
        vza_angle[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan
        _plot_instrument_angluar(vza_angle, 'VZA Band %s' % s2_band_id[i],
                                 '%s/Mean_VAA_500m.png' %fig_directory)

        # MODIS brdf polynomial parameter
        # please refer to https://modis.gsfc.nasa.gov/data/atbd/atbd_mod09.pdf on page-16
        g_iso = [1, 0, 0]
        g_vol = [-0.007574, -0.070987, 0.307588]
        g_geo = [-1.284909, -0.166314, 0.041840]
        g_white = [1.0, 0.189184, -1.377622]

        # Sentinel-2 band to be retrieved
        inverse_band_id = ['02', '03', '04', '8A', '11', '12']

        for m in range(len(inverse_band_id)):

            sensor_data = gdal.Open('%s/Mean_VAA_VZA_500m.tif' % tbd)

            vaa_data = sensor_data.GetRasterBand(1)
            vza_data = sensor_data.GetRasterBand(2)
            vaa_angle = vaa_data.ReadAsArray() / 100.
            vza_angle = vza_data.ReadAsArray() / 100.

            phi = (saa_angle - vaa_angle) % 180.

            brdf1_val = brdf_f1(sza_angle, vza_angle, phi)
            brdf2_val = brdf_f2(sza_angle, vza_angle, phi)

            if inverse_band_id[m] == '02':
                mcd_dataset = kernel_weights['fs'][:,0,:,:]
            if inverse_band_id[m] == '03':
                mcd_dataset = kernel_weights['fs'][:,1,:,:]
            if inverse_band_id[m] == '04':
                mcd_dataset = kernel_weights['fs'][:,2,:,:]
            if inverse_band_id[m] == '8A':
                mcd_dataset = kernel_weights['fs'][:,3,:,:]
            if inverse_band_id[m] == '11':
                mcd_dataset = kernel_weights['fs'][:,4,:,:]
            if inverse_band_id[m] == '12':
                mcd_dataset = kernel_weights['fs'][:,5,:,:]

            mcd_k0 = mcd_dataset[0,:,:]
            mcd_k1 = mcd_dataset[1,:,:]
            mcd_k2 = mcd_dataset[2,:,:]

            _plot_kernel(mcd_k0, 'k0', fig_directory + '/k0_band%s.png' % inverse_band_id[m])
            _plot_kernel(mcd_k1, 'k1', fig_directory + '/k1_band%s.png' % inverse_band_id[m])
            _plot_kernel(mcd_k2, 'k2', fig_directory + '/k2_band%s.png' % inverse_band_id[m])

            brf_array = mcd_k0 + mcd_k1 * brdf2_val + mcd_k2 * brdf1_val
            brf_array[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = -999.

            np.save('%s/brf_band%s.npy' % (tbd, inverse_band_id[m]), brf_array)

            dhr_array = mcd_k0 * (g_iso[0] + g_iso[1] * np.deg2rad(sza_angle) ** 2 +
                                  g_iso[2] * np.deg2rad(sza_angle) ** 3) +\
                        mcd_k1 * (g_vol[0] + g_vol[1] * np.deg2rad(sza_angle) ** 2 +
                                  g_vol[2] * np.deg2rad(sza_angle) ** 3) + \
                        mcd_k2 * (g_geo[0] + g_geo[1] * np.deg2rad(sza_angle) ** 2 +
                                  g_geo[2] * np.deg2rad(sza_angle) ** 3)

            bhr_array = mcd_k0 * g_white[0] + mcd_k1 * g_white[1] + mcd_k2 * g_white[2]
            dhr_array[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = -999.
            bhr_array[boa_band02_500m.reshape((s2_rows_500m, s2_cols_500m)) < 0] = -999.

            np.save('%s/dhr_band%s.npy' % (tbd, inverse_band_id[m]), dhr_array)
            np.save('%s/bhr_band%s.npy' % (tbd, inverse_band_id[m]), bhr_array)

            _plot_2d_brf(brf_array, '%s/modis_brf_band%s.png' % (fig_directory, inverse_band_id[m]))
            _plot_2d_brf(dhr_array, '%s/modis_dhr_band%s.png' % (fig_directory, inverse_band_id[m]))
            _plot_2d_brf(bhr_array, '%s/modis_bhr_band%s.png' % (fig_directory, inverse_band_id[m]))
