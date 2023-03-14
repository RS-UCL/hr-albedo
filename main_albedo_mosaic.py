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

from find_mosaic_mcd43 import *
S2GM_config = LoadConfig('./config_mosiac.yaml')
sentinel2_directory = S2GM_config.__get_sentinel2_attr__()
modis_tile = find_mcd43(sentinel2_directory)
print(modis_tile)