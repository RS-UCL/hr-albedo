# Created by Dr. Rui Song at 28/01/2022
# Email: rui.song@ucl.ac.uk

import matplotlib.pyplot as plt
import numpy as np
import gdal
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

def cal_mosaic(sentinel2_file, cloud_threshold):

    tbd_directory = sentinel2_file + '/tbd'  # temporal directory, to be deleted in the end.
    fig_directory = sentinel2_file + '/Figures'  # temporal directory, to be deleted in the end.
    product_directory = sentinel2_file + '/albedo'
    granule_dir = sentinel2_file + '/GRANULE'

    if not os.path.exists(product_directory):
        os.mkdir(product_directory)

    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            level2_dir = granule_dir + '/%s/IMG_DATA'%file

    # build vrt format for individual albedo subdatasets
    command = "gdalbuildvrt %s/merge_dhr_band02.vrt %s/sub_dhr_band02_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_band03.vrt %s/sub_dhr_band03_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_band04.vrt %s/sub_dhr_band04_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_band8A.vrt %s/sub_dhr_band8A_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_band11.vrt %s/sub_dhr_band11_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_band12.vrt %s/sub_dhr_band12_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_bandNIR.vrt %s/sub_dhr_bandNIR_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_bandSW.vrt %s/sub_dhr_bandSW_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_dhr_bandVIS.vrt %s/sub_dhr_bandVIS_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band02.vrt %s/sub_bhr_band02_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band03.vrt %s/sub_bhr_band03_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band04.vrt %s/sub_bhr_band04_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band8A.vrt %s/sub_bhr_band8A_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band11.vrt %s/sub_bhr_band11_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_band12.vrt %s/sub_bhr_band12_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_bandNIR.vrt %s/sub_bhr_bandNIR_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_bandSW.vrt %s/sub_bhr_bandSW_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    command = "gdalbuildvrt %s/merge_bhr_bandVIS.vrt %s/sub_bhr_bandVIS_*.tiff"%(tbd_directory,tbd_directory)
    os.system(command)

    s2_bands = ['02','03','04','8A','VIS','NIR','SW','11','12']

    for file in os.listdir(tbd_directory):
        if file.endswith("band02_masked.tiff"):
            s2_band02_masked = gdal.Open('%s/%s'%(tbd_directory, file))

    # load cloud mask (here Deeplabv3+ cloud mask is being used)
    mask2 = np.load('%s/CLOUD_MASK/mask2.npy'%sentinel2_file)
    mask2 = np.argmax(mask2,axis=-1)
    cloud_mask = np.zeros((mask2.shape))
    cloud_mask[mask2>= cloud_threshold] = -999.

    geotransform_20m = s2_band02_masked.GetGeoTransform()
    proj_20m = s2_band02_masked.GetProjection()

    nx, ny = cloud_mask.shape
    outputFileName =  tbd_directory + '/cloud_mask_20m.tiff'
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform_20m)
    dst_ds.SetProjection(proj_20m)
    dst_ds.GetRasterBand(1).WriteArray(cloud_mask)
    dst_ds.FlushCache()
    dst_ds = None

    cloud_mask_20m_file = tbd_directory + '/cloud_mask_20m.tiff'

    ####################################################################################################
    granule_dir = sentinel2_file + '/GRANULE'
    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            level2_dir = granule_dir + '/%s/IMG_DATA'%file

    ####################################################################################################
    # extract geo-coordinates
    for file in os.listdir(level2_dir):
        if file.endswith("B02_sur.tif"):
            src = gdal.Open('%s/%s'%(level2_dir, file))
            ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
            lrx = ulx + (src.RasterXSize * xres)
            lry = uly + (src.RasterYSize * yres)

    command = "gdalwarp -tr 10 10 -te %s %s %s %s -r average -overwrite %s %s/cloud_mask_10m.tiff \n" % (ulx, lry, lrx, uly, cloud_mask_20m_file, tbd_directory)
    os.system(command)

    cloud_mask_10m_data = gdal.Open('%s/cloud_mask_10m.tiff'%tbd_directory)
    cloud_mask_10m = cloud_mask_10m_data.GetRasterBand(1)
    cloud_mask_10m_cols = cloud_mask_10m_data.RasterXSize
    cloud_mask_10m_rows = cloud_mask_10m_data.RasterYSize
    cloud_mask_10m = cloud_mask_10m.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)
    cloud_mask_10m[cloud_mask_10m<0] = -999.

    for i in range(len(s2_bands)):

        merged_data  = gdal.Open('%s/merge_dhr_band%s.vrt'%(tbd_directory,s2_bands[i]))
        band_data = merged_data.GetRasterBand(1)

        cols = merged_data.RasterXSize
        rows = merged_data.RasterYSize
        band_data  = band_data.ReadAsArray(0, 0, cols, rows)
        print(cols,rows)
        band_unc_rel = np.load(tbd_directory + '/unc_relative_B%s.npy'%s2_bands[i])
        band_unc = band_data * band_unc_rel

        print('Mean band %s dhr is: %s -------'%(s2_bands[i], np.mean(band_data)))
        print('Mean band %s dhr uncertainty is: %s -------'%(s2_bands[i], np.mean(band_unc[band_unc>0])))

        if (s2_bands[i] =='02') | (s2_bands[i] =='03') | (s2_bands[i] =='04'):
            # dhr for 10-m bands: band-02, band-03 and band-04
            band_data[cloud_mask_10m==-999.] = np.nan
            for file in os.listdir(level2_dir):
                if file.endswith("B02.jp2"):
                    src = gdal.Open('%s/%s'%(level2_dir, file))

                    projectionRef10 = src.GetProjectionRef()
                    geotransform10  = src.GetGeoTransform()

        else:
            # dhr for other 20-m bands
            band_data[cloud_mask==-999.] = np.nan
            for file in os.listdir(level2_dir):
                if file.endswith("B8A.jp2"):
                    src = gdal.Open('%s/%s'%(level2_dir, file))

                    projectionRef20 = src.GetProjectionRef()
                    geotransform20  = src.GetGeoTransform()

        for file in os.listdir(level2_dir):
            if file.endswith("B8A.jp2"):
                L1C_filename = file[0:-8]

        projectionRef = src.GetProjectionRef()
        geotransform  = src.GetGeoTransform()

        dhr_name = product_directory + '/%sB%s_UCL_dhr.jp2'%(L1C_filename,s2_bands[i])
        dhr_unc_name = product_directory + '/%sB%s_UCL_dhr-unc.jp2'%(L1C_filename,s2_bands[i])

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

    dhr_band02_data  = dhr_band02_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)
    dhr_band03_data  = dhr_band03_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)
    dhr_band04_data  = dhr_band04_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)

    _compose_rgb(dhr_band04_data,dhr_band03_data,dhr_band02_data,cloud_mask_10m,product_directory,
                 '%s_UCL_dhr_rgb'%L1C_filename, projectionRef10, geotransform10)

    for i in range(len(s2_bands)):

        merged_data  = gdal.Open('%s/merge_bhr_band%s.vrt'%(tbd_directory,s2_bands[i]))
        band_data = merged_data.GetRasterBand(1)

        cols = merged_data.RasterXSize
        rows = merged_data.RasterYSize
        band_data  = band_data.ReadAsArray(0, 0, cols, rows)

        band_unc_rel = np.load(tbd_directory + '/unc_relative_B%s.npy'%s2_bands[i])
        band_unc = band_data * band_unc_rel

        print('Mean band %s bhr is: %s -------'%(s2_bands[i], np.mean(band_data)))
        print('Mean band %s bhr uncertainty is: %s -------'%(s2_bands[i], np.mean(band_unc[band_unc>0])))

        if (s2_bands[i] =='02') | (s2_bands[i] =='03') | (s2_bands[i] =='04'):
            # bhr for 10-m bands: band-02, band-03 and band-04
            band_data[cloud_mask_10m==-999.] = np.nan
            for file in os.listdir(level2_dir):
                if file.endswith("B02.jp2"):
                    src = gdal.Open('%s/%s'%(level2_dir, file))

        else:
            # bhr for other 20-m bands
            band_data[cloud_mask==-999.] = np.nan
            for file in os.listdir(level2_dir):
                if file.endswith("B8A.jp2"):
                    src = gdal.Open('%s/%s'%(level2_dir, file))

        projectionRef = src.GetProjectionRef()
        geotransform  = src.GetGeoTransform()

        for file in os.listdir(level2_dir):
            if file.endswith("B8A.jp2"):
                L1C_filename = file[0:-8]

        bhr_name = product_directory + '/%sB%s_UCL_bhr.jp2'%(L1C_filename,s2_bands[i])
        bhr_unc_name = product_directory + '/%sB%s_UCL_bhr-unc.jp2'%(L1C_filename,s2_bands[i])

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
    bhr_band03_data = bhr_band03.GetRasterBand(1)
    bhr_band04_data = bhr_band04.GetRasterBand(1)

    bhr_band02_data  = bhr_band02_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)
    bhr_band03_data  = bhr_band03_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)
    bhr_band04_data  = bhr_band04_data.ReadAsArray(0, 0, cloud_mask_10m_cols, cloud_mask_10m_rows)

    _compose_rgb(bhr_band04_data,bhr_band03_data,bhr_band02_data,cloud_mask_10m,product_directory,'%s_UCL_bhr_rgb'%L1C_filename, projectionRef10, geotransform10)

