#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_albedo_mosaic_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        13/04/2023 17:55

from apply_inversion_mosaic_v2 import *
from cal_endmember_mosaic_v2 import *
from apply_mosaic_mosaic_v2 import *
from preprocess_kernels import *
import datetime
import glob
import yaml
import sys

# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('./albedo_process.log')  # change the path to the log file as needed
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# python main_albedo_mosaic_v2.py ./data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0/ 30 1000 100

# sentinel2_directory = sys.argv[1]
# sample_interval = int(sys.argv[2])
# patch_size = int(sys.argv[3])
# patch_overlap = int(sys.argv[4])

sentinel2_directory = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0'
prior_dir = '/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/Nairobi_M10_February_2022_UTM_CM_SIAC/S2GM_M10_20220201_20220228_Nairobi_STD_v2.0.1/tile_0/VIIRS_prior/'
index_file = sentinel2_directory + '/Nairobi_validation_source_index_resampling_mode_ovr_none.tif'

patch_size = 1000
patch_overlap = 100

######## start preprocessing the kernels
preprocess_kernels(prior_dir, index_file, logger)
######## start calculating the endmembers
cal_endmember(sentinel2_directory)
######## start retrieval process
apply_inversion(sentinel2_directory, patch_size, patch_overlap)
######## add uncertainties to albedo
apply_uncertainty(sentinel2_directory)
######## mosaic the albedo from subpatches
cal_mosaic(sentinel2_directory, 0.95)
