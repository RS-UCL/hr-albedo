#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_albedo_masaic_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        13/04/2023 17:55

from cal_endmember_mosaic_v2 import *
import datetime
import glob
import yaml
import sys

# python main_albedo_masaic_v2.py ./data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0/ 30 1000 100

sentinel2_directory = sys.argv[1]
sample_interval = int(sys.argv[2])
patch_size = int(sys.argv[3])
patch_overlap = int(sys.argv[4])

######## start calculating the endmembers
cal_endmember(sentinel2_directory)
