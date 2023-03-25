#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    cal_endmember_val.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        24/03/2023 18:34

# Created by Dr. Rui Song at 24/01/2022
# Email: rui.song@ucl.ac.uk

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

    sample_interval = 10
    # file subdirectory in the format of sentinel2_directory/GRANULE/L1C****/IMG_DATA
    file_subdirectory = os.path.join(sentinel2_directory, 'GRANULE')
    file_subdirectory = os.path.join(file_subdirectory, os.listdir(file_subdirectory)[0])
    file_subdirectory = os.path.join(file_subdirectory, 'IMG_DATA')

    kernel_weights = np.load(file_subdirectory + '/kernel_weights.npz')
    # print(kernel_weights.files)
    # print(kernel_weights['fs'].shape)
    # quit()
    tbd = file_subdirectory + '/tbd'
    if not os.path.exists(tbd):
        os.makedirs(tbd)

    fig_directory = file_subdirectory + '/Figures'  # temporal directory, to be deleted in the end.
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)

    # modis_band001_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band1' % mcd43a1_file
    # modis_band002_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band2' % mcd43a1_file
    # modis_band003_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band3' % mcd43a1_file
    # modis_band004_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band4' % mcd43a1_file
    # modis_band005_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band5' % mcd43a1_file
    # modis_band006_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band6' % mcd43a1_file
    # modis_band007_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band7' % mcd43a1_file
    # modis_bandVIS_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_vis' % mcd43a1_file
    # modis_bandNIR_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_nir' % mcd43a1_file
    # modis_band0SW_file = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_shortwave' % mcd43a1_file
    #
    # # load modis geo-reference data
    # modis_brdf_band001 = gdal.Open(modis_band001_file)
    # modis_brdf_geotransform = modis_brdf_band001.GetGeoTransform()
    # modis_brdf_proj = modis_brdf_band001.GetProjection()
    #
    # modis_brdf_x_resolution = modis_brdf_geotransform[1]
    # modis_brdf_y_resolution = modis_brdf_geotransform[5]
    #
    # # load Sentinel-2 spectral surface reflectance, with masks on.
    # tbd_directory = sentinel2_directory + '/tbd'  # temporal directory, to be deleted in the end.
    # if not os.path.exists(tbd_directory):
    #     os.makedirs(tbd_directory)

    # use gdalwarp from os.system() to process a file in the subdirectory
    for file in os.listdir(file_subdirectory):
        if file.endswith(("B02_sur.tif", "B03_sur.tif", "B04_sur.tif", "B8A_sur.tif", "B11_sur.tif", "B12_sur.tif", "_mask.tif")):
            os.system(f'gdalwarp -tr 500 500 "{file_subdirectory}/{file}" "{tbd}/{file[:-4]}_500m.tif"')

    for file in os.listdir(file_subdirectory):
        if file.endswith(("B02_sur.tif", "B03_sur.tif", "B04_sur.tif", "B8A_sur.tif", "B11_sur.tif", "B12_sur.tif", "_mask.tif")):
            os.system(f'gdalwarp -tr 20 20 "{file_subdirectory}/{file}" "{tbd}/{file[:-4]}_20m.tif"')

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
        if file.endswith("B02_sur_500m.tif"):
            s2_band02_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B03_sur_500m.tif"):
            s2_band03_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B04_sur_500m.tif"):
            s2_band04_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B8A_sur_500m.tif"):
            s2_band8A_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B11_sur_500m.tif"):
            s2_band11_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B12_sur_500m.tif"):
            s2_band12_500m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("_mask_500m.tif"):
            s2_mask_500m_data = gdal.Open('%s/%s' % (tbd, file))

    for file in os.listdir(tbd):
        if file.endswith("B02_sur_20m.tif"):
            s2_band02_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B03_sur_20m.tif"):
            s2_band03_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B04_sur_20m.tif"):
            s2_band04_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B8A_sur_20m.tif"):
            s2_band8A_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B11_sur_20m.tif"):
            s2_band11_20m_data = gdal.Open('%s/%s' % (tbd, file))
        if file.endswith("B12_sur_20m.tif"):
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
    boa_mask_500m = s2_mask_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols_500m, s2_rows_500m)

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
    #
    print("-----------> Complete calculating abundance on aggregated S2 scence.\n")
    np.save('%s/s2_500m_abundance.npy' % tbd, s2_abundance_500m)

    # plot 2d abundance figures
    for i in range(main_endmember.shape[0]):
        abundane_i = s2_abundance_500m[:, :, i]
        abundane_i = abundane_i.reshape(s2_rows_500m, s2_cols_500m)
        abundane_i[boa_band02_500m_array.reshape((s2_rows_500m, s2_cols_500m)) < 0] = np.nan

        colortable_i = ascii_uppercase[i]
        _plot_2d_abundance(abundane_i, fig_directory, colortable_i)

    quit()
    # load solar and sensor angular data
    granule_dir = sentinel2_file + '/GRANULE'
    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            angular_dir = granule_dir + '/%s/ANG_DATA' % file

    solar_file = angular_dir + '/SAA_SZA.tif'
    solar_data = gdal.Open(solar_file)
    solar_proj = solar_data.GetProjection()

    sensor_b02_file = angular_dir + '/VAA_VZA_B02.tif'
    sensor_b03_file = angular_dir + '/VAA_VZA_B03.tif'
    sensor_b04_file = angular_dir + '/VAA_VZA_B04.tif'
    sensor_b8A_file = angular_dir + '/VAA_VZA_B8A.tif'
    sensor_b11_file = angular_dir + '/VAA_VZA_B11.tif'
    sensor_b12_file = angular_dir + '/VAA_VZA_B12.tif'

    sensor_b02_data = gdal.Open(sensor_b02_file)
    sensor_b03_data = gdal.Open(sensor_b03_file)
    sensor_b04_data = gdal.Open(sensor_b04_file)
    sensor_b8A_data = gdal.Open(sensor_b8A_file)
    sensor_b11_data = gdal.Open(sensor_b11_file)
    sensor_b12_data = gdal.Open(sensor_b12_file)

    sensor_b02_proj = sensor_b02_data.GetProjection()
    sensor_b03_proj = sensor_b03_data.GetProjection()
    sensor_b04_proj = sensor_b04_data.GetProjection()
    sensor_b8A_proj = sensor_b8A_data.GetProjection()
    sensor_b11_proj = sensor_b11_data.GetProjection()
    sensor_b12_proj = sensor_b12_data.GetProjection()

    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/solar_angle_500.tiff'
              % (solar_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, solar_file, tbd_directory))

    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b02_500.tiff'
              % (sensor_b02_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b02_file,
                 tbd_directory))
    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b03_500.tiff'
              % (sensor_b03_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b03_file,
                 tbd_directory))
    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b04_500.tiff'
              % (sensor_b04_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b04_file,
                 tbd_directory))
    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b8A_500.tiff'
              % (sensor_b8A_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b8A_file,
                 tbd_directory))
    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b11_500.tiff'
              % (sensor_b11_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b11_file,
                 tbd_directory))
    os.system('gdalwarp -s_srs %s -t_srs %s -tr %s %s -te %s %s %s %s -overwrite %s %s/sensor_angle_b12_500.tiff'
              % (sensor_b12_proj, modis_brdf_proj, modis_brdf_x_resolution, modis_brdf_y_resolution,
                 s2_SIN_500m_xmin, s2_SIN_500m_ymin, s2_SIN_500m_xmax, s2_SIN_500m_ymax, sensor_b12_file,
                 tbd_directory))

    solar_500_data = gdal.Open('%s/solar_angle_500.tiff' % tbd_directory)
    saa_data = solar_500_data.GetRasterBand(1)
    sza_data = solar_500_data.GetRasterBand(2)
    saa_angle = saa_data.ReadAsArray() / 100.
    sza_angle = sza_data.ReadAsArray() / 100.

    saa_angle[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = np.nan
    _plot_solar_angluar(saa_angle, fig_directory + '/saa_angle.png')

    sza_angle[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = np.nan
    _plot_solar_angluar(sza_angle, fig_directory + '/sza_angle.png')

    s2_band_id = ['02','03','04','8A','11','12']
    for i in range(len(s2_band_id)):
        sensor_500_data = gdal.Open('%s/sensor_angle_b%s_500.tiff' % (tbd_directory, s2_band_id[i]))
        vaa_data = sensor_500_data.GetRasterBand(1)
        vza_data = sensor_500_data.GetRasterBand(2)
        vaa_angle = vaa_data.ReadAsArray() / 100.
        vza_angle = vza_data.ReadAsArray() / 100.

        fig, ax = plt.subplots(figsize=(16, 16))
        vaa_angle[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = np.nan
        _plot_instrument_angluar(vaa_angle, 'VAA Band %s' % s2_band_id[i],
                                 '%s/vaa_angle_b%s.png' % (fig_directory, s2_band_id[i]))
        vza_angle[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = np.nan
        _plot_instrument_angluar(vaa_angle, 'VZA Band %s' % s2_band_id[i],
                                 '%s/vza_angle_b%s.png' % (fig_directory, s2_band_id[i]))

        # MODIS brdf polynomial parameter
        # please refer to https://modis.gsfc.nasa.gov/data/atbd/atbd_mod09.pdf on page-16
        g_iso = [1, 0, 0]
        g_vol = [-0.007574, -0.070987, 0.307588]
        g_geo = [-1.284909, -0.166314, 0.041840]
        g_white = [1.0, 0.189184, -1.377622]

        # Sentinel-2 band to be retrieved
        inverse_band_id = ['02', '03', '04', 'VIS', 'NIR', 'SW', '8A', '11', '12']

        for m in range(len(inverse_band_id)):

            sensor_data = gdal.Open('%s/sensor_angle_b%s_500.tiff' % (tbd_directory, s2_band_id[i]))

            if inverse_band_id[i] == 'VIS':
                sensor_data = gdal.Open('%s/sensor_angle_b02_500.tiff' % tbd_directory)
            if inverse_band_id[i] == 'NIR':
                sensor_data = gdal.Open('%s/sensor_angle_b8A_500.tiff' % tbd_directory)
            if inverse_band_id[i] == 'SW':
                sensor_data = gdal.Open('%s/sensor_angle_b02_500.tiff' % tbd_directory)

            vaa_data = sensor_data.GetRasterBand(1)
            vza_data = sensor_data.GetRasterBand(2)
            vaa_angle = vaa_data.ReadAsArray() / 100.
            vza_angle = vza_data.ReadAsArray() / 100.

            phi = (saa_angle - vaa_angle) % 180.

            brdf1_val = brdf_f1(sza_angle, vza_angle, phi)
            brdf2_val = brdf_f2(sza_angle, vza_angle, phi)

            if inverse_band_id[m] == '02':
                mcd_dataset = gdal.Open("%s/modis_brdf_band003_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == '03':
                mcd_dataset = gdal.Open("%s/modis_brdf_band004_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == '04':
                mcd_dataset = gdal.Open("%s/modis_brdf_band001_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == '11':
                mcd_dataset = gdal.Open("%s/modis_brdf_band006_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == '12':
                mcd_dataset = gdal.Open("%s/modis_brdf_band007_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == '8A':
                mcd_dataset = gdal.Open("%s/modis_brdf_band002_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == 'VIS':
                mcd_dataset = gdal.Open("%s/modis_brdf_bandVIS_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == 'NIR':
                mcd_dataset = gdal.Open("%s/modis_brdf_bandNIR_cropped.tiff" % tbd_directory)
            if inverse_band_id[m] == 'SW':
                mcd_dataset = gdal.Open("%s/modis_brdf_band0SW_cropped.tiff" % tbd_directory)

            mcd_k0 = mcd_dataset.GetRasterBand(1)
            mcd_k1 = mcd_dataset.GetRasterBand(2)
            mcd_k2 = mcd_dataset.GetRasterBand(3)

            mcd_k0 = mcd_k0.ReadAsArray() / 1.e3
            mcd_k1 = mcd_k1.ReadAsArray() / 1.e3
            mcd_k2 = mcd_k2.ReadAsArray() / 1.e3

            brf_array = mcd_k0 + mcd_k1 * brdf2_val + mcd_k2 * brdf1_val
            brf_array[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = -999.

            np.save('%s/brf_band%s.npy' % (tbd_directory, inverse_band_id[m]), brf_array)

            dhr_array = mcd_k0 * (g_iso[0] + g_iso[1] * np.sin(np.deg2rad(sza_angle)) +
                                  g_iso[2] * np.sin(np.deg2rad(sza_angle)) ** 2) +\
                        mcd_k1 * (g_vol[0] + g_vol[1] * np.sin(np.deg2rad(sza_angle))
                                  + g_vol[2] * np.sin(np.deg2rad(sza_angle)) ** 2) + \
                        mcd_k2 * (g_geo[0] + g_geo[1] * np.sin(np.deg2rad(sza_angle))
                                  + g_geo[2] * np.sin(np.deg2rad(sza_angle)) ** 2)

            bhr_array = mcd_k0 * g_white[0] + mcd_k1 * g_white[1] + mcd_k2 * g_white[2]
            dhr_array[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = -999.
            bhr_array[s2_band02_SIN_500m_array.reshape((s2_band02_SIN_500m_rows, s2_band02_SIN_500m_cols)) < 0] = -999.
            print(s2_band_id[i], dhr_array.shape)
            np.save('%s/dhr_band%s.npy' % (tbd_directory, inverse_band_id[m]), dhr_array)
            np.save('%s/bhr_band%s.npy' % (tbd_directory, inverse_band_id[m]), bhr_array)

            _plot_2d_brf(brf_array, '%s/modis_brf_band%s.png' % (fig_directory, inverse_band_id[m]))
            _plot_2d_brf(dhr_array, '%s/modis_dhr_band%s.png' % (fig_directory, inverse_band_id[m]))
            _plot_2d_brf(bhr_array, '%s/modis_bhr_band%s.png' % (fig_directory, inverse_band_id[m]))
