# Created by Dr. Rui Song at 31/01/2022
# Email: rui.song@ucl.ac.uk

# This script shows the method provided b Roy (2017) in Sentinel-2 NBAR from using MODIS BRDF.
# https://www.sciencedirect.com/science/article/pii/S0034425717302791

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib as mpl
import numpy as np
import gdal
import glob
import os

#################################
def readcdict(lutfile, minV, maxV):
    if (lutfile!=None):
        red=[]
        green=[]
        blue=[]
        lut=open(lutfile, 'r')
        i=0
        for line in lut:
            tab=line.split(',')
            tab[0]=tab[0].strip()
            tab[1]=tab[1].strip()
            tab[2]=tab[2].strip()
            val= minV+((i/255.0)*(maxV-minV))
            red.insert(i,(val , np.float32(tab[0])/255.0, np.float32(tab[0])/255.0))
            green.insert(i,(val, np.float32(tab[1])/255.0, np.float32(tab[1])/255.0))
            blue.insert(i,(val, float(tab[2])/255.0, np.float32(tab[2])/255.0))
            i= i+1
    return {'red':red, 'green':green, 'blue':blue}
##############################

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

def cal_Sentinel2_NBAR(Sentinel2_file, MODIS_brdf_file, Sentinel2_band_ID):

    """

    :param Sentinel2_file: location of Sentinel-2 L1C data.
    :param MODIS_brdf_file: location of MODIS BRDF prior (or daily BRDF).
    :param band_ID: Sentinel-2 band-ID
    :return: Sentinlel-2 spectral NBAR
    """

    granule_dir = Sentinel2_file + '/GRANULE'
    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            IMG_dir = granule_dir + '/%s/IMG_DATA' % file

    # locate and extract S2 brf data.
    sentinel_brf_data = glob.glob('%s/*%s_sur.tif'%(IMG_dir, Sentinel2_band_ID))[0]
    s2_brf = gdal.Open('%s' % sentinel_brf_data)

    # get geo-reference and proj
    s2_geotransform = s2_brf.GetGeoTransform()
    s2_proj = s2_brf.GetProjection()

    # get number of cols and rows
    s2_cols = s2_brf.RasterXSize
    s2_rows = s2_brf.RasterYSize

    s2_brf_array = s2_brf.GetRasterBand(1).ReadAsArray(0, 0, s2_cols, s2_rows) / 1.e4

    s2_ymax = s2_geotransform[3]
    s2_ymin = s2_geotransform[3] + s2_geotransform[5] * s2_brf.RasterYSize
    s2_xmin = s2_geotransform[0]
    s2_xmax = s2_geotransform[0] + s2_geotransform[1] * s2_brf.RasterXSize

    s2_x_resolution = s2_geotransform[1]
    s2_y_resolution = s2_geotransform[5]

    output_directory = './NBAR'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # get solar and instrument angles
    for file in os.listdir(granule_dir):
        if file.startswith('L1C'):
            angular_dir = granule_dir + '/%s/ANG_DATA' % file

    sensor_angle_file = angular_dir + '/VAA_VZA_B%s.tif'%Sentinel2_band_ID
    solar_angle_file = angular_dir + '/SAA_SZA.tif'

    os.system('gdalwarp -srcnodata 32767 -dstnodata 32767 -tr %s %s -te %s %s %s %s -overwrite %s ./NBAR/VAA_VZA_cropped.tif'
        % (s2_x_resolution, s2_y_resolution, s2_xmin, s2_ymin, s2_xmax, s2_ymax, sensor_angle_file))

    os.system('gdalwarp -srcnodata 32767 -dstnodata 32767 -tr %s %s -te %s %s %s %s -overwrite %s ./NBAR/SAA_SZA_cropped.tif'
        % (s2_x_resolution, s2_y_resolution, s2_xmin, s2_ymin, s2_xmax, s2_ymax, solar_angle_file))

    solar_angle_data = gdal.Open('./NBAR/SAA_SZA_cropped.tif')
    saa_angle = solar_angle_data.GetRasterBand(1).ReadAsArray() / 100.
    sza_angle = solar_angle_data.GetRasterBand(2).ReadAsArray() / 100.

    sensor_angle_data = gdal.Open('./NBAR/VAA_VZA_cropped.tif')
    vaa_angle = sensor_angle_data.GetRasterBand(1).ReadAsArray() / 100.
    vza_angle = sensor_angle_data.GetRasterBand(2).ReadAsArray() / 100.

    vza_nadir = np.zeros((vza_angle.shape))
    vaa_nadir = np.zeros((vza_angle.shape))

    phi = (saa_angle - vaa_angle) % 180.
    phi_nadir = (saa_angle - vaa_nadir) % 180.

    brdf1_val = brdf_f1(sza_angle, vza_angle, phi)
    brdf2_val = brdf_f2(sza_angle, vza_angle, phi)

    brdf1_nadir = brdf_f1(sza_angle, vza_nadir, phi_nadir)
    brdf2_nadir = brdf_f2(sza_angle, vza_nadir, phi_nadir)

    # get corresponding MODIS data
    if Sentinel2_band_ID == '02':
        modis_band_id = 'Band3'
        # sentinel-2 band-02 equals MODIS band-03
        modis_brdf_data = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band3' % MODIS_brdf_file

    os.system('gdalwarp -srcnodata 32767 -dstnodata 32767 -tr %s %s -te %s %s %s %s -overwrite %s ./NBAR/modis_brdf_cropped.tif'
              % (s2_x_resolution, s2_y_resolution, s2_xmin, s2_ymin, s2_xmax, s2_ymax, modis_brdf_data))

    mcd_dataset = gdal.Open("./NBAR/modis_brdf_cropped.tif")

    mcd_k0 = mcd_dataset.GetRasterBand(1)
    mcd_k1 = mcd_dataset.GetRasterBand(2)
    mcd_k2 = mcd_dataset.GetRasterBand(3)

    mcd_k0 = mcd_k0.ReadAsArray() / 1.e3
    mcd_k1 = mcd_k1.ReadAsArray() / 1.e3
    mcd_k2 = mcd_k2.ReadAsArray() / 1.e3

    brf_array = mcd_k0 + mcd_k1 * brdf2_val + mcd_k2 * brdf1_val
    brf_nadir = mcd_k0 + mcd_k1 * brdf2_nadir + mcd_k2 * brdf1_nadir

    correction_factor = brf_nadir / brf_array
    s2_NBAR = s2_brf_array * correction_factor

    # set color lookup table
    lut_color = './CLUT/lut_colors.txt'
    cdict = readcdict(lut_color, 0, 1)
    my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

    fig, ax = plt.subplots()
    im = ax.imshow(s2_brf_array, cmap=my_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(im)
    plt.axis('off')
    plt.savefig('./NBAR/sentinel_brf_band%s.png' % Sentinel2_band_ID, dpi=400)
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(s2_NBAR, cmap=my_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(im)
    plt.axis('off')
    plt.savefig('./NBAR/sentinel_NBAR_band%s.png' % Sentinel2_band_ID, dpi=400)
    plt.close




Sentinel2_file = '/gws/nopw/j04/qa4ecv_vol2/HR_Alebdo/WheatBelt/30UXD/30UXD/' \
                 'S2A_MSIL1C_20200725T110631_N0209_R137_T30UXD_20200725T114244.SAFE'
MODIS_brdf_file = '/gws/nopw/j04/qa4ecv_vol2/HR_Alebdo/WheatBelt/30UXD/30UXD/' \
                  'S2A_MSIL1C_20200725T110631_N0209_R137_T30UXD_20200725T114244.SAFE/' \
                  'MCD43/MCD43A1.A2020207.h17v03.006.2020216042659.hdf'

cal_Sentinel2_NBAR(Sentinel2_file, MODIS_brdf_file, '02')