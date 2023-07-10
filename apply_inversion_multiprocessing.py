#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    apply_inversion_multiprocessing.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        10/07/2023 17:01

from sklearn.linear_model import LinearRegression
from pysptools.abundance_maps import FCLS
import matplotlib.pyplot as plt
from scipy import interpolate
from osgeo import gdal
import numpy as np
import glob
import os
import multiprocessing
from multiprocessing import Pool

def process_tile(args):

    m, n, tbd_directory = args

    patch_size = 1000
    patch_overlap = 100

    num_row_str = f"{'0' if m < 10 else ''}{m}"
    num_col_str = f"{'0' if n < 10 else ''}{n}"
    tiff_file = f"{tbd_directory}/sub_abundance_10m_h{num_row_str}v{num_col_str}.tiff"

    if glob.glob(tiff_file):
        print(f"tile h{num_row_str}v{num_col_str} is already processed")
        return

    start_row_10m = m * (patch_size - patch_overlap) if m > 0 else m * patch_size
    start_col_10m = n * (patch_size - patch_overlap) if n > 0 else n * patch_size

    end_row_10m = min(start_row_10m + patch_size, s2_10m_rows)
    end_col_10m = min(start_col_10m + patch_size, s2_10m_cols)

    s2_10m_proj_ymax = geotransform_10m[3] + geotransform_10m[5] * start_row_10m
    s2_10m_proj_ymin = geotransform_10m[3] + geotransform_10m[5] * end_row_10m
    s2_10m_proj_xmin = geotransform_10m[0] + geotransform_10m[1] * start_col_10m
    s2_10m_proj_xmax = geotransform_10m[0] + geotransform_10m[1] * end_col_10m

    boa_band02_10m_cut = boa_band02_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]
    boa_band03_10m_cut = boa_band03_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]
    boa_band04_10m_cut = boa_band04_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]
    boa_band8A_10m_cut = boa_band8A_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]
    boa_band11_10m_cut = boa_band11_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]
    boa_band12_10m_cut = boa_band12_10m[start_row_10m:end_row_10m, start_col_10m:end_col_10m]

    print('This 10-m patch has the following boundary: %s %s %s %s' %
          (start_row_10m, end_row_10m, start_col_10m, end_col_10m))

    os.system('gdalwarp -te %s %s %s %s -tr 10 10 -overwrite %s %s/cut_10m_h%sv%s.tiff' %
              (s2_10m_proj_xmin, s2_10m_proj_ymin, s2_10m_proj_xmax, s2_10m_proj_ymax,
               band02_10m_file, tbd_directory, num_row_str, num_col_str))

    patch_10m = gdal.Open('%s/cut_10m_h%sv%s.tiff' % (tbd_directory, num_row_str, num_col_str))

    VIS_coefficient = [-0.0048, 0.5673, 0.1407, 0.2359, 0, 0, 0]
    SW_coefficient = [-0.0049, 0.2688, 0.0362, 0.1501, 0.3045, 0.1644, 0.0356]
    NIR_coefficient = [-0.0073, 0., 0., 0., 0.5595, 0.3844, 0.0290]

    s2_matrix_10m_patch = np.zeros((boa_band02_10m_cut.size, 1, 6))

    boa_band02_10m_cut_array = boa_band02_10m_cut.reshape(boa_band02_10m_cut.size, 1)
    boa_band03_10m_cut_array = boa_band03_10m_cut.reshape(boa_band03_10m_cut.size, 1)
    boa_band04_10m_cut_array = boa_band04_10m_cut.reshape(boa_band04_10m_cut.size, 1)
    boa_band8A_10m_cut_array = boa_band8A_10m_cut.reshape(boa_band8A_10m_cut.size, 1)
    boa_band11_10m_cut_array = boa_band11_10m_cut.reshape(boa_band11_10m_cut.size, 1)
    boa_band12_10m_cut_array = boa_band12_10m_cut.reshape(boa_band12_10m_cut.size, 1)

    s2_matrix_10m_patch[:, 0, 0] = boa_band02_10m_cut_array[:, 0]
    s2_matrix_10m_patch[:, 0, 1] = boa_band03_10m_cut_array[:, 0]
    s2_matrix_10m_patch[:, 0, 2] = boa_band04_10m_cut_array[:, 0]
    s2_matrix_10m_patch[:, 0, 3] = boa_band8A_10m_cut_array[:, 0]
    s2_matrix_10m_patch[:, 0, 4] = boa_band11_10m_cut_array[:, 0]
    s2_matrix_10m_patch[:, 0, 5] = boa_band12_10m_cut_array[:, 0]

    s2_eea_wavelength = np.asarray([459., 560., 665., 865., 1610., 2190.])
    s2_interp_num = 11  # number of pixels to add between two wavelengths

    s2_wv_band02_band03 = np.linspace(459, 560, num=s2_interp_num, endpoint=True)
    s2_wv_band03_band04 = np.linspace(560, 665, num=s2_interp_num, endpoint=True)
    s2_wv_band04_band8A = np.linspace(665, 865, num=s2_interp_num, endpoint=True)
    s2_wv_band11_band12 = np.linspace(1610, 2190, num=2, endpoint=True)

    s2_wv_resampled = np.concatenate((s2_wv_band02_band03[0:-1], s2_wv_band03_band04), axis=0)
    s2_wv_resampled = np.concatenate((s2_wv_resampled[0:-1], s2_wv_band04_band8A), axis=0)
    s2_wv_resampled = np.concatenate((s2_wv_resampled, s2_wv_band11_band12), axis=0)

    func_wv_10m = interpolate.interp1d(s2_eea_wavelength, s2_matrix_10m_patch, axis=2)
    s2_matrix_10m_patch_interp = func_wv_10m(s2_wv_resampled)

    main_endmember = np.load('%s/endmembers.npy' % tbd_directory)
    CalAbundanceMap = FCLS()
    s2_10m_patch_abundance = CalAbundanceMap.map(s2_matrix_10m_patch_interp, main_endmember)
    np.save('%s/s2_10m_patch_abundance_h%sv%s.npy'
            % (tbd_directory, num_row_str, num_col_str), s2_10m_patch_abundance)

    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    geotransform_10m_patch = patch_10m.GetGeoTransform()
    proj_10m_patch = patch_10m.GetProjection()

    # dhr coefficients
    dhr_coefficient_band02_k = [dhr_coef_b[0, 0], dhr_coef_b[0, 1], dhr_coef_b[0, 2], dhr_coef_b[0, 3]]
    dhr_coefficient_band02_a = [dhr_coef_a[0, 0], dhr_coef_a[0, 1], dhr_coef_a[0, 2], dhr_coef_a[0, 3]]

    dhr_coefficient_band03_k = [dhr_coef_b[1, 0], dhr_coef_b[1, 1], dhr_coef_b[1, 2], dhr_coef_b[1, 3]]
    dhr_coefficient_band03_a = [dhr_coef_a[1, 0], dhr_coef_a[1, 1], dhr_coef_a[1, 2], dhr_coef_a[1, 3]]

    dhr_coefficient_band04_k = [dhr_coef_b[2, 0], dhr_coef_b[2, 1], dhr_coef_b[2, 2], dhr_coef_b[2, 3]]
    dhr_coefficient_band04_a = [dhr_coef_a[2, 0], dhr_coef_a[2, 1], dhr_coef_a[2, 2], dhr_coef_a[2, 3]]

    dhr_coefficient_band8A_k = [dhr_coef_b[3, 0], dhr_coef_b[3, 1], dhr_coef_b[3, 2], dhr_coef_b[3, 3]]
    dhr_coefficient_band8A_a = [dhr_coef_a[3, 0], dhr_coef_a[3, 1], dhr_coef_a[3, 2], dhr_coef_a[3, 3]]

    dhr_coefficient_band11_k = [dhr_coef_b[4, 0], dhr_coef_b[4, 1], dhr_coef_b[4, 2], dhr_coef_b[4, 3]]
    dhr_coefficient_band11_a = [dhr_coef_a[4, 0], dhr_coef_a[4, 1], dhr_coef_a[4, 2], dhr_coef_a[4, 3]]

    dhr_coefficient_band12_k = [dhr_coef_b[5, 0], dhr_coef_b[5, 1], dhr_coef_b[5, 2], dhr_coef_b[5, 3]]
    dhr_coefficient_band12_a = [dhr_coef_a[5, 0], dhr_coef_a[5, 1], dhr_coef_a[5, 2], dhr_coef_a[5, 3]]

    # bhr coefficients
    bhr_coefficient_band02_k = [bhr_coef_b[0, 0], bhr_coef_b[0, 1], bhr_coef_b[0, 2], bhr_coef_b[0, 3]]
    bhr_coefficient_band02_a = [bhr_coef_a[0, 0], bhr_coef_a[0, 1], bhr_coef_a[0, 2], bhr_coef_a[0, 3]]

    bhr_coefficient_band03_k = [bhr_coef_b[1, 0], bhr_coef_b[1, 1], bhr_coef_b[1, 2], bhr_coef_b[1, 3]]
    bhr_coefficient_band03_a = [bhr_coef_a[1, 0], bhr_coef_a[1, 1], bhr_coef_a[1, 2], bhr_coef_a[1, 3]]

    bhr_coefficient_band04_k = [bhr_coef_b[2, 0], bhr_coef_b[2, 1], bhr_coef_b[2, 2], bhr_coef_b[2, 3]]
    bhr_coefficient_band04_a = [bhr_coef_a[2, 0], bhr_coef_a[2, 1], bhr_coef_a[2, 2], bhr_coef_a[2, 3]]

    bhr_coefficient_band8A_k = [bhr_coef_b[3, 0], bhr_coef_b[3, 1], bhr_coef_b[3, 2], bhr_coef_b[3, 3]]
    bhr_coefficient_band8A_a = [bhr_coef_a[3, 0], bhr_coef_a[3, 1], bhr_coef_a[3, 2], bhr_coef_a[3, 3]]

    bhr_coefficient_band11_k = [bhr_coef_b[4, 0], bhr_coef_b[4, 1], bhr_coef_b[4, 2], bhr_coef_b[4, 3]]
    bhr_coefficient_band11_a = [bhr_coef_a[4, 0], bhr_coef_a[4, 1], bhr_coef_a[4, 2], bhr_coef_a[4, 3]]

    bhr_coefficient_band12_k = [bhr_coef_b[5, 0], bhr_coef_b[5, 1], bhr_coef_b[5, 2], bhr_coef_b[5, 3]]
    bhr_coefficient_band12_a = [bhr_coef_a[5, 0], bhr_coef_a[5, 1], bhr_coef_a[5, 2], bhr_coef_a[5, 3]]

    hr_albedo_bands = ['02', '03', '04', '8A', '11', '12', 'VIS', 'NIR', 'SW']

    abu_0_10m = s2_10m_patch_abundance[:, :, 0].reshape(boa_band02_10m_cut.shape[0],
                                                        boa_band02_10m_cut.shape[1])
    abu_1_10m = s2_10m_patch_abundance[:, :, 1].reshape(boa_band02_10m_cut.shape[0],
                                                        boa_band02_10m_cut.shape[1])
    abu_2_10m = s2_10m_patch_abundance[:, :, 2].reshape(boa_band02_10m_cut.shape[0],
                                                        boa_band02_10m_cut.shape[1])
    abu_3_10m = s2_10m_patch_abundance[:, :, 3].reshape(boa_band02_10m_cut.shape[0],
                                                        boa_band02_10m_cut.shape[1])

    # save and write 10-m abundance map
    nx, ny = boa_band02_10m_cut.shape
    abundance_10m_FileName = tbd_directory + '/sub_abundance_10m_h%sv%s.tif' % (num_row_str, num_col_str)
    dst_ds = gdal.GetDriverByName('GTiff').Create(abundance_10m_FileName, ny, nx, 4, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_10m_patch)
    dst_ds.SetProjection(proj_10m_patch)
    dst_ds.GetRasterBand(1).WriteArray(abu_0_10m)
    dst_ds.GetRasterBand(2).WriteArray(abu_1_10m)
    dst_ds.GetRasterBand(3).WriteArray(abu_2_10m)
    dst_ds.GetRasterBand(4).WriteArray(abu_3_10m)
    dst_ds.FlushCache()
    dst_ds = None

    for i in range(len(hr_albedo_bands)):

        if hr_albedo_bands[i] == '02':
            # save sentinel-2 band-02 albedo subdatasets
            coffe_k = np.copy(dhr_coefficient_band02_k)
            coffe_a = np.copy(dhr_coefficient_band02_a)
            hr_dhr_b02_10m = abu_0_10m * (boa_band02_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band02_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band02_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band02_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band02_k)
            coffe_a = np.copy(bhr_coefficient_band02_a)
            hr_bhr_b02_10m = abu_0_10m * (boa_band02_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band02_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band02_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band02_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b02_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b02_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b02_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == '03':
            coffe_k = np.copy(dhr_coefficient_band03_k)
            coffe_a = np.copy(dhr_coefficient_band03_a)
            hr_dhr_b03_10m = abu_0_10m * (boa_band03_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band03_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band03_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band03_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band03_k)
            coffe_a = np.copy(bhr_coefficient_band03_a)
            hr_bhr_b03_10m = abu_0_10m * (boa_band03_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band03_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band03_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band03_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b03_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b03_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b03_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == '04':
            coffe_k = np.copy(dhr_coefficient_band04_k)
            coffe_a = np.copy(dhr_coefficient_band04_a)
            hr_dhr_b04_10m = abu_0_10m * (boa_band04_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band04_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band04_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band04_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band04_k)
            coffe_a = np.copy(bhr_coefficient_band04_a)
            hr_bhr_b04_10m = abu_0_10m * (boa_band04_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band04_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band04_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band04_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b04_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b04_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b04_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == '8A':
            coffe_k = np.copy(dhr_coefficient_band8A_k)
            coffe_a = np.copy(dhr_coefficient_band8A_a)
            hr_dhr_b8A_10m = abu_0_10m * (boa_band8A_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band8A_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band8A_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band8A_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band8A_k)
            coffe_a = np.copy(bhr_coefficient_band8A_a)
            hr_bhr_b8A_10m = abu_0_10m * (boa_band8A_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band8A_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band8A_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band8A_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b8A_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b8A_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b8A_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == '11':
            coffe_k = np.copy(dhr_coefficient_band11_k)
            coffe_a = np.copy(dhr_coefficient_band11_a)
            hr_dhr_b11_10m = abu_0_10m * (boa_band11_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band11_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band11_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band11_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band11_k)
            coffe_a = np.copy(bhr_coefficient_band11_a)
            hr_bhr_b11_10m = abu_0_10m * (boa_band11_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band11_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band11_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band11_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b11_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b11_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b11_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == '12':
            coffe_k = np.copy(dhr_coefficient_band12_k)
            coffe_a = np.copy(dhr_coefficient_band12_a)
            hr_dhr_b12_10m = abu_0_10m * (boa_band12_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band12_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band12_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band12_10m_cut * coffe_k[3] + coffe_a[3])

            coffe_k = np.copy(bhr_coefficient_band12_k)
            coffe_a = np.copy(bhr_coefficient_band12_a)
            hr_bhr_b12_10m = abu_0_10m * (boa_band12_10m_cut * coffe_k[0] + coffe_a[0]) + abu_1_10m * (
                    boa_band12_10m_cut * coffe_k[1] + coffe_a[1]) + abu_2_10m * (
                                     boa_band12_10m_cut * coffe_k[2] + coffe_a[2]) + abu_3_10m * (
                                     boa_band12_10m_cut * coffe_k[3] + coffe_a[3])

            nx, ny = hr_dhr_b12_10m.shape

            outputFileName = tbd_directory + '/sub_dhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_b12_10m)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_band%s_h%sv%s.tiff' % (
                hr_albedo_bands[i], num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_b12_10m)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == 'VIS':
            hr_dhr_VIS = VIS_coefficient[0] + VIS_coefficient[1] * hr_dhr_b02_10m + \
                         VIS_coefficient[2] * hr_dhr_b03_10m + VIS_coefficient[3] * hr_dhr_b04_10m + \
                         VIS_coefficient[4] * hr_dhr_b8A_10m + VIS_coefficient[5] * hr_dhr_b11_10m + \
                         VIS_coefficient[6] * hr_dhr_b12_10m

            hr_bhr_VIS = VIS_coefficient[0] + VIS_coefficient[1] * hr_bhr_b02_10m + \
                         VIS_coefficient[2] * hr_bhr_b03_10m + VIS_coefficient[3] * hr_bhr_b04_10m + \
                         VIS_coefficient[4] * hr_bhr_b8A_10m + VIS_coefficient[5] * hr_bhr_b11_10m + \
                         VIS_coefficient[6] * hr_bhr_b12_10m

            nx, ny = hr_dhr_VIS.shape
            outputFileName = tbd_directory + '/sub_dhr_bandVIS_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_VIS)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_bandVIS_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_VIS)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == 'NIR':
            hr_dhr_NIR = NIR_coefficient[0] + NIR_coefficient[1] * hr_dhr_b02_10m + \
                         NIR_coefficient[2] * hr_dhr_b03_10m + NIR_coefficient[3] * hr_dhr_b04_10m + \
                         NIR_coefficient[4] * hr_dhr_b8A_10m + NIR_coefficient[5] * hr_dhr_b11_10m + \
                         NIR_coefficient[6] * hr_dhr_b12_10m
            hr_bhr_NIR = NIR_coefficient[0] + NIR_coefficient[1] * hr_bhr_b02_10m + \
                         NIR_coefficient[2] * hr_bhr_b03_10m + NIR_coefficient[3] * hr_bhr_b04_10m + \
                         NIR_coefficient[4] * hr_bhr_b8A_10m + NIR_coefficient[5] * hr_bhr_b11_10m + \
                         NIR_coefficient[6] * hr_bhr_b12_10m

            nx, ny = hr_dhr_NIR.shape

            outputFileName = tbd_directory + '/sub_dhr_bandNIR_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_NIR)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_bandNIR_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_NIR)
            dst_ds.FlushCache()
            dst_ds = None

        if hr_albedo_bands[i] == 'SW':
            hr_dhr_SW = SW_coefficient[0] + SW_coefficient[1] * hr_dhr_b02_10m + \
                        SW_coefficient[2] * hr_dhr_b03_10m + SW_coefficient[3] * hr_dhr_b04_10m + \
                        SW_coefficient[4] * hr_dhr_b8A_10m + SW_coefficient[5] * hr_dhr_b11_10m + \
                        SW_coefficient[6] * hr_dhr_b12_10m
            hr_bhr_SW = SW_coefficient[0] + SW_coefficient[1] * hr_bhr_b02_10m + \
                        SW_coefficient[2] * hr_bhr_b03_10m + SW_coefficient[3] * hr_bhr_b04_10m + \
                        SW_coefficient[4] * hr_bhr_b8A_10m + SW_coefficient[5] * hr_bhr_b11_10m + \
                        SW_coefficient[6] * hr_bhr_b12_10m

            nx, ny = hr_dhr_SW.shape

            outputFileName = tbd_directory + '/sub_dhr_bandSW_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_dhr_SW)
            dst_ds.FlushCache()
            dst_ds = None

            outputFileName = tbd_directory + '/sub_bhr_bandSW_h%sv%s.tiff' % (
                num_row_str, num_col_str)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform_10m_patch)
            dst_ds.SetProjection(proj_10m_patch)
            dst_ds.GetRasterBand(1).WriteArray(hr_bhr_SW)
            dst_ds.FlushCache()
            dst_ds = None

def parallel_process(num_row, col_row, pool_size, tbd_directory):
    # The pool_size parameter represents the number of worker processes to use; typically, this is set to the number of CPUs.
    with Pool(pool_size) as p:
        p.map(process_tile, [(m, n, tbd_directory) for m in range(num_row) for n in range(col_row)])

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

    parallel_process(num_row=int(num_row), col_row=int(col_row), pool_size=multiprocessing.cpu_count(), tbd_directory=tbd_directory)

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
