#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_albedo_mosaic.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        14/03/2023 10:31

class LoadConfig:

    def __init__(self, configuration):
        with open(configuration) as yaml_config_file:
            self.config = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
            self.sentinel2_arg = self.config['sentinel2']
            # self.modis_arg = self.config['MODIS']
            # self.eea_arg = self.config['EEA']

            self.__get_sentinel2_attr__()
            # self.__get_modis_attr__()
            # self.__get_EEA_attr__()

    def __get_sentinel2_attr__(self):
        sentinel2_directory = self.sentinel2_arg['directory']
        return sentinel2_directory

    def __get_modis_attr__(self):
        modis_tile = self.modis_arg['tile']
        return modis_tile

    def __get_EEA_attr__(self):
        sample_interval = self.eea_arg['SampleInterval']
        patch_size = self.eea_arg['PatchSize']
        patch_overlap = self.eea_arg['Overlap']
        return sample_interval, patch_size, patch_overlap

def get_modis_jasmin(modis_tile, sentinel2_directory):

    # Split the path into a list of substrings using the "/" delimiter
    path_parts = sentinel2_directory.split("/")

    # Loop through the path parts and find the one that starts with "S2GM"
    s2gm_part = ""
    for part in path_parts:
        if part.startswith("S2GM_"):
            s2gm_part = part
            break

    # extract year, month, day from Sentinel-2 file name
    year_start = s2gm_part[9:13]
    month_start = s2gm_part[13:15]
    day_start = s2gm_part[15:17]

    # convert datetime to day of year
    datetime_start_str = datetime.datetime.strptime('%s-%s-%s' % (year_start, month_start, day_start), '%Y-%m-%d')
    print(datetime_start_str)
    quit()
    datetime_start_str = datetime_start_str.timetuple()
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

from cal_endmember_mosaic import *
from find_mosaic_mcd43 import *
import yaml

S2GM_config = LoadConfig('./config_mosaic.yaml')
sentinel2_directory = S2GM_config.__get_sentinel2_attr__()
modis_tile = find_mcd43(sentinel2_directory)

# get modis MCD43A1 data for the same day of year
mcd43a1_file = get_modis_jasmin(modis_tile, sentinel2_directory)

# start calculating the endmembers
cal_endmember(sentinel2_directory+'/%s'%sentinel2_filename, mcd43a1_file, sample_interval)