# Created by Dr. Rui Song at 29/01/2022
# Email: rui.song@ucl.ac.uk

import matplotlib.pyplot as plt
import numpy as np
import gdal
import glob
import os

def add_cloud_mask(data_dir, cloud_threshold):

    ####################################################################################################
    granule_dir = data_dir + '/GRANULE'
    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            level2_dir = granule_dir + '/%s/IMG_DATA'%file

    ####################################################################################################
    # extract geo-coordinates
    for file in os.listdir(level2_dir):
        if file.endswith("B12_sur.tif"):
            src = gdal.Open('%s/%s'%(level2_dir, file))
            ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
            lrx = ulx + (src.RasterXSize * xres)
            lry = uly + (src.RasterYSize * yres)

    ####################################################################################################
    #reproject data to 20-m resolution
    if not os.path.exists('%s/20m'%level2_dir):
        os.mkdir('%s/20m'%level2_dir)

    if glob.glob('%s/20m/*_sur.tiff'%level2_dir):
        print('--- Sentinel-2 surface reflectancce data is already resampled to 20m.')
    else:
        print('--- Start to resample Sentinel-2 spectral surface reflectancce data to 20-m resolution.')
        for toa_file in os.listdir(level2_dir):
            if toa_file.endswith("B05_sur.tif") | toa_file.endswith("B06_sur.tif") | toa_file.endswith("B07_sur.tif") | \
                    toa_file.endswith("B8A_sur.tif") | toa_file.endswith("B11_sur.tif") | toa_file.endswith("B12_sur.tif"):
                command = "gdal_translate -of GTiff %s/%s %s/20m/%s.tiff \n" % (level2_dir, toa_file, level2_dir, toa_file[0:-4])
                os.system(command)
            if toa_file.endswith("B01_sur.tif") | toa_file.endswith("B02_sur.tif") | toa_file.endswith("B03_sur.tif") | \
                    toa_file.endswith("B04_sur.tif") | toa_file.endswith("B08_sur.tif") |toa_file.endswith("B09_sur.tif") | \
                    toa_file.endswith("B10_sur.tif"):
                command = "gdalwarp -tr 20 20 -te %s %s %s %s -r average -overwrite %s/%s %s/20m/%s.tiff \n" % \
                          (ulx, lry, lrx, uly, level2_dir, toa_file, level2_dir, toa_file[0:-4])
                os.system(command)

    ####################################################################################################
    #load 20-m data from band 2,3,4,8A,11,12 for endmember processing.
    for file in os.listdir('%s/20m/'%level2_dir):
        if file.endswith("B02_sur.tiff"):
            band02 = gdal.Open('%s/20m/%s'%(level2_dir, file))
        if file.endswith("B03_sur.tiff"):
            band03 = gdal.Open('%s/20m/%s'%(level2_dir, file))
        if file.endswith("B04_sur.tiff"):
            band04 = gdal.Open('%s/20m/%s'%(level2_dir, file))
        if file.endswith("B8A_sur.tiff"):
            band8A = gdal.Open('%s/20m/%s'%(level2_dir, file))
        if file.endswith("B11_sur.tiff"):
            band11 = gdal.Open('%s/20m/%s'%(level2_dir, file))
        if file.endswith("B12_sur.tiff"):
            band12 = gdal.Open('%s/20m/%s'%(level2_dir, file))

    ####################################################################################################
    # extract spectral surface reflectance data

    s2_20m_cols = band02.RasterXSize
    s2_20m_rows = band02.RasterYSize

    s2_scaling_factor = 1.e4

    boa_band02 = band02.GetRasterBand(1)
    boa_band03 = band03.GetRasterBand(1)
    boa_band04 = band04.GetRasterBand(1)
    boa_band8A = band8A.GetRasterBand(1)
    boa_band11 = band11.GetRasterBand(1)
    boa_band12 = band12.GetRasterBand(1)

    boa_band02 = boa_band02.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor
    boa_band03 = boa_band03.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor
    boa_band04 = boa_band04.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor
    boa_band8A = boa_band8A.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor
    boa_band11 = boa_band11.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor
    boa_band12 = boa_band12.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows) / s2_scaling_factor

    sensing_cover_rate = np.size(boa_band02[boa_band02>0.])/np.size(boa_band02)

    #cloud_mask = np.ones((boa_band02.shape))
    mask2 = np.load('%s/CLOUD_MASK/mask2.npy'%data_dir)
    mask2 = np.argmax(mask2,axis=-1)
    cloud_mask = np.zeros((mask2.shape))
    cloud_mask[mask2>=cloud_threshold] = 1

    cloud_cover_rate = 1-np.size(boa_band02[(boa_band02>0.) & (mask2<cloud_threshold)])/np.size(boa_band02)
    fig_dir = data_dir + '/Figures/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    #####################################################
    # plot cloud mask
    fig, ax = plt.subplots(figsize=(20,18))
    plt.imshow(cloud_mask,cmap='jet',vmin=0.,vmax=1.)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/cloud_mask.png'%fig_dir)

    geotransform_20m = band02.GetGeoTransform()
    proj_20m = band02.GetProjection()

    boa_band02_masked = np.copy(boa_band03)
    boa_band03_masked = np.copy(boa_band03)
    boa_band04_masked = np.copy(boa_band04)
    boa_band8A_masked = np.copy(boa_band8A)
    boa_band11_masked = np.copy(boa_band11)
    boa_band12_masked = np.copy(boa_band12)

    boa_band02_masked[cloud_mask==1] = -999.
    boa_band03_masked[cloud_mask==1] = -999.
    boa_band04_masked[cloud_mask==1] = -999.
    boa_band8A_masked[cloud_mask==1] = -999.
    boa_band11_masked[cloud_mask==1] = -999.
    boa_band12_masked[cloud_mask==1] = -999.

    ######################################################
    # save masked surface refelctance data

    tbd_dir = data_dir + '/tbd'
    if not os.path.exists(tbd_dir):
        os.mkdir(tbd_dir)

    nx, ny = boa_band02_masked.shape
    outputFileName =  tbd_dir + '/boa_band02_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band02_masked)
    dst_ds.FlushCache()
    dst_ds = None

    outputFileName =  tbd_dir + '/boa_band03_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band03_masked)
    dst_ds.FlushCache()
    dst_ds = None

    outputFileName =  tbd_dir + '/boa_band04_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band04_masked)
    dst_ds.FlushCache()
    dst_ds = None

    outputFileName =  tbd_dir + '/boa_band8A_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band8A_masked)
    dst_ds.FlushCache()
    dst_ds = None

    outputFileName =  tbd_dir + '/boa_band11_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band11_masked)
    dst_ds.FlushCache()
    dst_ds = None

    outputFileName =  tbd_dir + '/boa_band12_masked.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(boa_band12_masked)
    dst_ds.FlushCache()
    dst_ds = None

    ####################################################################################################
    #save png for 20-m cloud-masked data from band 2,3,4,8A,11,12.
    for file in os.listdir(tbd_dir):
        if file.endswith("02_masked.tiff"):
            masked_band02 = gdal.Open('%s/%s'%(tbd_dir, file))
        if file.endswith("03_masked.tiff"):
            masked_band03 = gdal.Open('%s/%s'%(tbd_dir, file))
        if file.endswith("04_masked.tiff"):
            masked_band04 = gdal.Open('%s/%s'%(tbd_dir, file))
        if file.endswith("8A_masked.tiff"):
            masked_band8A = gdal.Open('%s/%s'%(tbd_dir, file))
        if file.endswith("11_masked.tiff"):
            masked_band11 = gdal.Open('%s/%s'%(tbd_dir, file))
        if file.endswith("12_masked.tiff"):
            masked_band12 = gdal.Open('%s/%s'%(tbd_dir, file))

    masked_boa_band02 = masked_band02.GetRasterBand(1)
    masked_boa_band03 = masked_band03.GetRasterBand(1)
    masked_boa_band04 = masked_band04.GetRasterBand(1)
    masked_boa_band8A = masked_band8A.GetRasterBand(1)
    masked_boa_band11 = masked_band11.GetRasterBand(1)
    masked_boa_band12 = masked_band12.GetRasterBand(1)

    masked_boa_band02 = masked_boa_band02.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)
    masked_boa_band03 = masked_boa_band03.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)
    masked_boa_band04 = masked_boa_band04.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)
    masked_boa_band8A = masked_boa_band8A.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)
    masked_boa_band11 = masked_boa_band11.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)
    masked_boa_band12 = masked_boa_band12.ReadAsArray(0, 0, s2_20m_cols, s2_20m_rows)

    #####################################################
    # plot masked boa
    masked_boa_band02[masked_boa_band02 < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band02,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band02.png'%fig_dir)

    #####################################################
    # plot masked boa
    masked_boa_band03[masked_boa_band03 < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band03,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band03.png'%fig_dir)

    #####################################################
    # plot masked boa
    masked_boa_band04[masked_boa_band04 < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band04,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band04.png'%fig_dir)

    #####################################################
    # plot masked boa
    masked_boa_band8A[masked_boa_band8A < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band8A,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band8A.png'%fig_dir)

    #####################################################
    # plot masked boa
    masked_boa_band11[masked_boa_band11 < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band11,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band11.png'%fig_dir)

    #####################################################
    # plot masked boa
    masked_boa_band12[masked_boa_band12 < 0.] = np.nan
    fig, ax = plt.subplots(figsize=(20,18))
    cmap = plt.cm.jet
    cmap.set_bad('grey')
    plt.imshow(masked_boa_band12,cmap=cmap,vmin=0.,vmax=.2)
    cbar = plt.colorbar(shrink=0.5, extend = 'both')
    cbar.set_label('Masked BOA', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Pixels', fontsize = 26)
    plt.ylabel('Pixels', fontsize = 26)
    plt.savefig('%s/masked_boa_band12.png'%fig_dir)

    return sensing_cover_rate, cloud_cover_rate