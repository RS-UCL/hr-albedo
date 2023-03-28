#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_albedo_val.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        24/03/2023 18:33

class LoadConfig:
    # test
    def __init__(self, configuration):
        with open(configuration) as yaml_config_file:
            self.config = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
            self.sentinel2_arg = self.config['sentinel2']
            # self.modis_arg = self.config['MODIS']
            self.eea_arg = self.config['EEA']

            self.__get_sentinel2_attr__()
            # self.__get_modis_attr__()
            self.__get_EEA_attr__()

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
    year_start = int(s2gm_part[9:13])
    month_start = int(s2gm_part[13:15])
    day_start = int(s2gm_part[15:17])

    year_end = int(s2gm_part[18:22])
    month_end = int(s2gm_part[22:24])
    day_end = int(s2gm_part[24:26])

    # convert datetime to day of year
    datetime_start = datetime.datetime(year_start, month_start, day_start)
    datetime_end = datetime.datetime(year_end, month_end, day_end)

    # Calculate the time delta between the two dates
    delta = datetime_end - datetime_start
    # Calculate the half delta
    half_delta = delta / 2
    # Calculate the middle datetime
    middle_datetime = datetime_start + half_delta
    # Format the middle datetime as a string in the format "YYYY-MM-DD"
    middle_datetime_str = middle_datetime.strftime("%Y-%m-%d")
    # Extract the day of year from the middle datetime string
    doy = datetime.datetime.strptime(middle_datetime_str, "%Y-%m-%d").strftime("%j")

    # extract MCD43 data from SIAC intermediate results
    try:
        mcd43a1_file = glob.glob(sentinel2_directory + '/MCD43/*%s*%s*.hdf' %
                                 (doy, modis_tile))[0]
        print('-----------> MCD43A1 data found from SIAC intermediate results')
    except:
        print('MCD43A1 data not found from SIAC intermediate results, please download *%s*%s*.hdf to %s/MCD43/' %(doy, modis_tile, sentinel2_directory))

    return mcd43a1_file

from apply_inversion_val import *
from cal_endmember_val import *
import datetime
import glob
import yaml

S2GM_config = LoadConfig('./config/config_val.yaml')
sentinel2_directory = S2GM_config.__get_sentinel2_attr__()
(sample_interval, patch_size, patch_overlap) = S2GM_config.__get_EEA_attr__()

# start calculating the endmembers
cal_endmember(sentinel2_directory)
# start retrieval process
apply_inversion(sentinel2_directory, patch_size, patch_overlap)

# add uncertainties to albedo
apply_uncertainty(sentinel2_directory)
