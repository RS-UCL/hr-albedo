#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    find_mosaic_mcd43.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        12/03/2023 18:36

import numpy as np
from osgeo import gdal, osr

def pixel2coord(x, y, geotransform):
    """Returns global coordinates from pixel x, y coords"""
    xp = geotransform[0] + geotransform[1] * x + geotransform[2] * y
    yp = geotransform[3] + geotransform[4] * x + geotransform[5] * y
    return xp, yp

def coord2latlon(x, y, projection):
    """Returns lat, long from projected coordinates"""
    # Define the projection and reference system used by the dataset
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    # Define the projection and reference system for lat-lon coordinates
    srs_latlon = srs.CloneGeogCS()
    # Create a transformation function between the two coordinate systems
    transform = osr.CoordinateTransformation(srs, srs_latlon)
    # Transform the projected coordinates to lat-lon
    lon, lat, z = transform.TransformPoint(x, y)
    return lat, lon

def find_mcd43(s2_mosaic_band):

    # Open the GeoTIFF file
    dataset = gdal.Open(s2_mosaic_band)

    # Get the projection and geotransform information of the dataset
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    # Loop through each pixel and convert its coordinates to lat-lon
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows).astype(np.float)

    for x in range(0, cols + 1, 1000):
        for y in range(0, rows + 1, 1000):
            xp, yp = pixel2coord(x, y, geotransform)
            lat, lon = coord2latlon(xp, yp, projection)
            print(lat, lon, data[y][x])

if __name__ == '__main__':

    find_mcd43('/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0/B8A.tif')
