#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    apply_mosaic_mosaic_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        18/05/2023 18:22

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import glob
import os

def _save_band(array, outputFileName, projectionRef, geotransform):

        nx, ny = array.shape
        if os.path.exists(outputFileName):
            os.remove(outputFileName)

        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projectionRef)
        array = array * 10000
        array[~(array>0)] = -9999
        array[array>10000] = -9999
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(1).WriteArray(array)
        dst_ds.FlushCache()
        dst_ds = None

def _save_rgb(rgba_array, rgb_scale, outputFileName, projection, geotransform):

        nx, ny       = rgba_array.shape[1:]
        #outputFileName = self.s2_file_dir+'/%s'%name
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 3, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=JPEG"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(1).WriteArray(rgba_array[0])
        dst_ds.GetRasterBand(1).SetScale(rgb_scale)
        dst_ds.GetRasterBand(2).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(2).WriteArray(rgba_array[1])
        dst_ds.GetRasterBand(2).SetScale(rgb_scale)
        dst_ds.GetRasterBand(3).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(3).WriteArray(rgba_array[2])
        dst_ds.GetRasterBand(3).SetScale(rgb_scale)
        #dst_ds.GetRasterBand(4).SetNoDataValue(-9999)
        #dst_ds.GetRasterBand(4).WriteArray(rgba_array[3])
        dst_ds.FlushCache()
        dst_ds = None

def _compose_rgb(band_R,band_G,band_B,cloud_mask,outputFolder,outputFilename, projectionRef, geotransform):

        rgb_scale = 4
        ref_scale   = 1

        r, g, b = band_R * ref_scale, band_G * ref_scale, band_B * ref_scale
        alpha   = (r>0) & (g>0) & (b>0)

        r_rescale = np.clip(r * rgb_scale * 255,0, 255).astype(np.uint8)
        g_rescale = np.clip(g * rgb_scale * 255,0, 255).astype(np.uint8)
        b_rescale = np.clip(b * rgb_scale * 255,0, 255).astype(np.uint8)
        rgb_rescale = np.clip(alpha * rgb_scale * 255,0, 255).astype(np.uint8)

        r_rescale[(r>1)|(g>1)|(b>1)] = -9999
        g_rescale[(r>1)|(g>1)|(b>1)] = -9999
        b_rescale[(r>1)|(g>1)|(b>1)] = -9999
        rgb_rescale[(r>1)|(g>1)|(b>1)] = -9999

        r_rescale[cloud_mask == -999.] = 128.
        g_rescale[cloud_mask == -999.] = 128.
        b_rescale[cloud_mask == -999.] = 128.

        rgba_array = np.asarray([r_rescale, g_rescale, b_rescale])

        name = outputFolder + '/%s.jp2'%outputFilename

        _save_rgb(rgba_array, rgb_scale, name, projectionRef, geotransform)

        #gdal.Translate(outputFolder +'/%s.png'%outputFilename, outputFolder + '/%s.jp2'%outputFilename, \
        #               format = 'PNG', widthPct=25, heightPct=25, resampleAlg=gdal.GRA_Bilinear ).FlushCache()

def cal_mosaic(sentinel2_directory, cloud_threshold):

    file_subdirectory = sentinel2_directory

    tbd_directory = file_subdirectory + '/tbd'  # temporal directory, to be deleted in the end.
    fig_directory = file_subdirectory + '/Figures'
    product_directory = file_subdirectory + '/albedo'
    if not os.path.exists(product_directory):
        os.mkdir(product_directory)

    # define a list of tuples containing the vrt filename and the subdataset pattern for each band
    bands = [('dhr_band02', 'sub_dhr_band02_*.tiff'),
             ('dhr_band03', 'sub_dhr_band03_*.tiff'),
             ('dhr_band04', 'sub_dhr_band04_*.tiff'),
             ('dhr_band8A', 'sub_dhr_band8A_*.tiff'),
             ('dhr_band11', 'sub_dhr_band11_*.tiff'),
             ('dhr_band12', 'sub_dhr_band12_*.tiff'),
             ('dhr_bandNIR', 'sub_dhr_bandNIR_*.tiff'),
             ('dhr_bandSW', 'sub_dhr_bandSW_*.tiff'),
             ('dhr_bandVIS', 'sub_dhr_bandVIS_*.tiff'),
             ('bhr_band02', 'sub_bhr_band02_*.tiff'),
             ('bhr_band03', 'sub_bhr_band03_*.tiff'),
             ('bhr_band04', 'sub_bhr_band04_*.tiff'),
             ('bhr_band8A', 'sub_bhr_band8A_*.tiff'),
             ('bhr_band11', 'sub_bhr_band11_*.tiff'),
             ('bhr_band12', 'sub_bhr_band12_*.tiff'),
             ('bhr_bandNIR', 'sub_bhr_bandNIR_*.tiff'),
             ('bhr_bandSW', 'sub_bhr_bandSW_*.tiff'),
             ('bhr_bandVIS', 'sub_bhr_bandVIS_*.tiff')]

    # loop through the list of tuples and build the vrt file for each band
    # for band in bands:
    #     vrt_filename = f"{tbd_directory}/merge_{band[0]}.vrt"
    #     subdataset_pattern = f"{tbd_directory}/{band[1]}"
    #     command = f"gdalbuildvrt {vrt_filename} {subdataset_pattern}"
    #     os.system(command)

    for file in os.listdir(file_subdirectory):
        if file.endswith("cloud_confidence.tif"):
            s2_mask_data = gdal.Open('%s/%s' % (file_subdirectory, file))
            s2_cols = s2_mask_data.RasterXSize
            s2_rows = s2_mask_data.RasterYSize
            cloud_mask = s2_mask_data.GetRasterBand(1).ReadAsArray(0, 0, s2_cols, s2_rows)

    plt.figure(figsize=(10, 10))
    plt.imshow(cloud_mask, cmap='rainbow')
    plt.colorbar(label='Cloud Confidence', shrink=0.5)
    plt.title('Cloud Confidence Map - Nairobi')
    plt.savefig('%s/cloud_confidence.png' % product_directory)

    cm_threshold = 5.  # cloud confidence threshold
    cm = np.zeros((cloud_mask.shape))
    cm[cloud_mask > cm_threshold] = 1.
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='rainbow')
    plt.colorbar(label='Cloud Confidence', shrink=0.5)
    plt.title('Cloud Mask - Nairobi')
    plt.savefig('%s/cm.png' % product_directory)
    quit()
    s2_bands = ['02', '03', '04', '8A', 'VIS', 'NIR', 'SW', '11', '12']

    for i in range(len(s2_bands)):

        merged_data  = gdal.Open('%s/merge_dhr_band%s.vrt'%(tbd_directory,s2_bands[i]))
        band_data = merged_data.GetRasterBand(1)

        cols = merged_data.RasterXSize
        rows = merged_data.RasterYSize
        band_data  = band_data.ReadAsArray(0, 0, cols, rows)

        band_unc_rel = np.load(tbd_directory + '/unc_relative_B%s.npy'%s2_bands[i])
        band_unc = band_data * band_unc_rel

        print('Mean band %s dhr is: %s -------'%(s2_bands[i], np.nanmean(band_data[band_data>0])))
        print('Mean band %s dhr uncertainty is: %s -------'%(s2_bands[i], np.nanmean(band_unc[band_unc>0])))

        band_data[cm>0.] = np.nan

        for file in os.listdir(file_subdirectory):
            if file.endswith("B02.tif"):
                src = gdal.Open('%s/%s'%(file_subdirectory, file))
                projectionRef10 = src.GetProjectionRef()
                geotransform10  = src.GetGeoTransform()

        projectionRef = src.GetProjectionRef()
        geotransform  = src.GetGeoTransform()

        dhr_name = product_directory + '/B%s_UCL_dhr.jp2'%(s2_bands[i])
        dhr_unc_name = product_directory + '/B%s_UCL_dhr-unc.jp2'%(s2_bands[i])

        _save_band(band_data, dhr_name, projectionRef, geotransform)
        _save_band(band_unc, dhr_unc_name, projectionRef, geotransform)

        fig, ax = plt.subplots(figsize=(16,16))
        cmap = plt.cm.jet
        cmap.set_bad('grey')
        plt.imshow(band_data,cmap=cmap,vmin=0.,vmax=0.6)
        cbar = plt.colorbar(shrink=0.5, extend = 'both')
        cbar.set_label('DHR', fontsize=30)
        cbar.ax.tick_params(labelsize=30)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        plt.title('DHR Band %s'%(s2_bands[i]), fontsize=34)
        plt.xlabel('Pixels', fontsize = 34)
        plt.ylabel('Pixels', fontsize = 34)
        plt.tight_layout()
        plt.savefig('%s/merged_DHR_band%s.png'%(product_directory,s2_bands[i]))
        plt.close()

    dhr_band02  = gdal.Open('%s/merge_dhr_band02.vrt'%tbd_directory)
    dhr_band03  = gdal.Open('%s/merge_dhr_band03.vrt'%tbd_directory)
    dhr_band04  = gdal.Open('%s/merge_dhr_band04.vrt'%tbd_directory)

    dhr_band02_data = dhr_band02.GetRasterBand(1)
    dhr_band03_data = dhr_band03.GetRasterBand(1)
    dhr_band04_data = dhr_band04.GetRasterBand(1)

    dhr_band02_data  = dhr_band02_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)
    dhr_band02_data[cm > 0.] = np.nan
    dhr_band03_data  = dhr_band03_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)
    dhr_band03_data[cm > 0.] = np.nan
    dhr_band04_data  = dhr_band04_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)
    dhr_band04_data[cm > 0.] = np.nan

    _compose_rgb(dhr_band04_data,dhr_band03_data,dhr_band02_data,cloud_mask_10m,product_directory,
                 'UCL_dhr_rgb', projectionRef10, geotransform10)

    for i in range(len(s2_bands)):

        merged_data  = gdal.Open('%s/merge_bhr_band%s.vrt'%(tbd_directory,s2_bands[i]))
        band_data = merged_data.GetRasterBand(1)

        cols = merged_data.RasterXSize
        rows = merged_data.RasterYSize
        band_data  = band_data.ReadAsArray(0, 0, cols, rows)

        band_unc_rel = np.load(tbd_directory + '/unc_relative_B%s.npy'%s2_bands[i])
        band_unc = band_data * band_unc_rel

        print('Mean band %s bhr is: %s -------'%(s2_bands[i], np.nanmean(band_data[band_unc>0])))
        print('Mean band %s bhr uncertainty is: %s -------'%(s2_bands[i], np.nanmean(band_unc[band_unc>0])))


        band_data[cm>0.] = np.nan
        for file in os.listdir(level2_dir):
            if file.endswith("B02.tif"):
                src = gdal.Open('%s/%s'%(level2_dir, file))

        projectionRef = src.GetProjectionRef()
        geotransform  = src.GetGeoTransform()

        bhr_name = product_directory + '/B%s_UCL_bhr.jp2'%(s2_bands[i])
        bhr_unc_name = product_directory + '/B%s_UCL_bhr-unc.jp2'%(s2_bands[i])

        _save_band(band_data, bhr_name, projectionRef, geotransform)
        _save_band(band_unc, bhr_unc_name, projectionRef, geotransform)

        fig, ax = plt.subplots(figsize=(16,16))
        cmap = plt.cm.jet
        cmap.set_bad('grey')
        plt.imshow(band_data,cmap=cmap,vmin=0.,vmax=0.6)
        cbar = plt.colorbar(shrink=0.5, extend = 'both')
        cbar.set_label('BHR', fontsize=30)
        cbar.ax.tick_params(labelsize=30)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        plt.title('BHR Band %s'%(s2_bands[i]), fontsize=34)
        plt.xlabel('Pixels', fontsize = 34)
        plt.ylabel('Pixels', fontsize = 34)
        plt.tight_layout()
        plt.savefig('%s/merged_BHR_band%s.png'%(product_directory,s2_bands[i]))
        plt.close()

    bhr_band02  = gdal.Open('%s/merge_bhr_band02.vrt'%tbd_directory)
    bhr_band03  = gdal.Open('%s/merge_bhr_band03.vrt'%tbd_directory)
    bhr_band04  = gdal.Open('%s/merge_bhr_band04.vrt'%tbd_directory)

    bhr_band02_data = bhr_band02.GetRasterBand(1)
    bhr_band02_data[cm > 0.] = np.nan
    bhr_band03_data = bhr_band03.GetRasterBand(1)
    bhr_band03_data[cm > 0.] = np.nan
    bhr_band04_data = bhr_band04.GetRasterBand(1)
    bhr_band04_data[cm > 0.] = np.nan

    bhr_band02_data  = bhr_band02_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)
    bhr_band03_data  = bhr_band03_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)
    bhr_band04_data  = bhr_band04_data.ReadAsArray(0, 0, s2_cols_10m, s2_rows_10m)

    _compose_rgb(bhr_band04_data,bhr_band03_data,bhr_band02_data,cloud_mask_10m,product_directory,'UCL_bhr_rgb', projectionRef10, geotransform10)

