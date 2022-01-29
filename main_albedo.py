# Created by Dr. Rui Song at 18/01/2022
# Email: rui.song@ucl.ac.uk

from apply_cloud_mask import *
from apply_inversion import *
from cal_endmember import *
from apply_mosaic import*
import datetime
import glob
import yaml


class LoadConfig:

    def __init__(self, configuration):
        with open(configuration) as yaml_config_file:
            self.config = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
            self.sentinel2_arg = self.config['sentinel2']
            self.modis_arg = self.config['MODIS']
            self.eea_arg = self.config['EEA']

            self.__get_sentinel2_attr__()
            self.__get_modis_attr__()
            self.__get_EEA_attr__()

    def __get_sentinel2_attr__(self):
        sentinel2_filename = self.sentinel2_arg['filename']
        sentinel2_directory = self.sentinel2_arg['directory']
        sentinel2_tile = self.sentinel2_arg['tile']
        return sentinel2_filename, sentinel2_directory, sentinel2_tile

    def __get_modis_attr__(self):
        modis_tile = self.modis_arg['tile']
        return modis_tile

    def __get_EEA_attr__(self):
        sample_interval = self.eea_arg['SampleInterval']
        patch_size = self.eea_arg['PatchSize']
        patch_overlap = self.eea_arg['Overlap']
        return sample_interval, patch_size, patch_overlap


def get_modis_jasmin(modis_tile, sentinel2_directory, sentinel2_filename):
    # extract year, month, day from Sentinel-2 file name
    year = sentinel2_filename[11:15]
    month = sentinel2_filename[15:17]
    day = sentinel2_filename[17:19]

    # convert datetime to day of year
    datetime_str = datetime.datetime.strptime('%s-%s-%s' % (year, month, day), '%Y-%m-%d')
    datetime_str = datetime_str.timetuple()
    doy = datetime_str.tm_yday
    doy = str(doy)

    # extract MCD43 data from SIAC intermediate results
    try:
        mcd43a1_file = glob.glob(sentinel2_directory + '/%s/MCD43/*%s%s*%s*.hdf' %
                                 (sentinel2_filename, year, doy, modis_tile))[0]
        print('-----------> MCD43A1 data found from SIAC intermediate results')
    except:
        print('MCD43A1 data not found from SIAC intermediate results')

    return mcd43a1_file


HrAlbedo_config = LoadConfig('./config.yaml')

(sentinel2_filename, sentinel2_directory, sentinel2_tile) = HrAlbedo_config.__get_sentinel2_attr__()
modis_tile = HrAlbedo_config.__get_modis_attr__()
(sample_interval, patch_size, patch_overlap) = HrAlbedo_config.__get_EEA_attr__()

print('-----------> Processing Sentinel-2 tile for: %s\n' % sentinel2_tile)

# get modis MCD43A1 data for the same day of year
mcd43a1_file = get_modis_jasmin(modis_tile, sentinel2_directory, sentinel2_filename)

# apply cloud mask
add_cloud_mask(sentinel2_directory+'/%s'%sentinel2_filename, 0.95)
# start calculating the endmembers
cal_endmember(sentinel2_directory+'/%s'%sentinel2_filename, mcd43a1_file, sample_interval)
# start retrieval process
apply_inversion(sentinel2_directory+'/%s'%sentinel2_filename, mcd43a1_file, patch_size, patch_overlap)
# add uncertainties to albedo
apply_uncertainty(sentinel2_directory+'/%s'%sentinel2_filename)
# albedo subdataset mosaic
cal_mosaic(sentinel2_directory+'/%s'%sentinel2_filename, 0.95)