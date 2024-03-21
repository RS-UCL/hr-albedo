#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_albedo_multiprocessing.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        10/07/2023 17:10

from apply_inversion_multiprocessing import *
from cal_endmember_mosaic_v2 import *
from apply_mosaic_mosaic_v2 import *
from preprocess_kernels import *
import datetime
import time
import glob
import yaml
import sys


def list_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('albedo_process_multi.log')

# Set level for handlers
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.DEBUG) # This is the change, it will log all messages to file

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

# python main_albedo_mosaic_v2.py ./data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0/ 30 1000 100

# sentinel2_directory = sys.argv[1]
# sample_interval = int(sys.argv[2])
# patch_size = int(sys.argv[3])
# patch_overlap = int(sys.argv[4])

sentinel2_directory = sys.argv[1]
prior_dir = sentinel2_directory+'/VIIRS_prior/'
index_file = sentinel2_directory + '/validation_source_index_500.tif'

patch_size = 500
patch_overlap = 50

# List all files before Albedo procedure
BRF_files = list_files(os.path.dirname(sentinel2_directory))

# Start the timer
start_time = time.time()
######## start preprocessing the kernels
preprocess_kernels(prior_dir, index_file, logger)
preprocessing_time = time.time() - start_time
print('preprocessing time: ', preprocessing_time)
######## start calculating the endmembers
cal_endmember(sentinel2_directory)
endmember_time = time.time() - preprocessing_time - start_time
######## start retrieval process
failed_retrievals = apply_inversion(sentinel2_directory, patch_size, patch_overlap)

if len(failed_retrievals) == 9:
    # List all Albedo files
    All_files = list_files(os.path.dirname(sentinel2_directory))
    Albedo_files = [file for file in All_files if file not in BRF_files]

    # Remove every new file
    new_dirs = set([os.path.dirname(new_file) for new_file in Albedo_files])
    for new_file in Albedo_files:
        os.remove(new_file)
    for dir in new_dirs:
        if not "VIIRS_prior" in dir:
            os.rmdir(dir)

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time
    print('preprocessing time: ', preprocessing_time)
    print('endmember time: ', endmember_time)
    print('total time: ', total_time)

else:
    inversion_time = time.time() - preprocessing_time - endmember_time - start_time
    ######## add uncertainties to albedo
    apply_uncertainty(sentinel2_directory)
    uncertainty_time = time.time() - preprocessing_time - endmember_time - inversion_time - start_time
    ######## mosaic the albedo from subpatches
    cal_mosaic(sentinel2_directory, 0.95)
    mosaic_time = time.time() - preprocessing_time - endmember_time - inversion_time - uncertainty_time - start_time

    # List all Albedo files
    All_files = list_files(os.path.dirname(sentinel2_directory))
    Albedo_files = [file for file in All_files if file not in BRF_files]

    # Remove files related to failed bands
    new_dirs = set([os.path.dirname(new_file) for new_file in Albedo_files])
    all_failed_bands = failed_retrievals.copy()
    for band in failed_retrievals:
        all_failed_bands.append("band"+band[1:])

    for new_file in Albedo_files:
        if any(substring in new_file for substring in all_failed_bands):
            os.remove(new_file)

    # Remove RGB bands
    rgb_bands = [file for file in Albedo_files if "rgb" in file]
    if "B02" in failed_retrievals or "B03" in failed_retrievals or "B04" in failed_retrievals:
        for rgb_band in rgb_bands:
            os.remove(rgb_band)

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time
    print('preprocessing time: ', preprocessing_time)
    print('endmember time: ', endmember_time)
    print('inversion time: ', inversion_time)
    print('uncertainty time: ', uncertainty_time)
    print('mosaic time: ', mosaic_time)
    print('total time: ', total_time)
