#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    find_mosaic_mcd43.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        12/03/2023 18:36

import math
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
    lat, lon, z = transform.TransformPoint(x, y)
    return lat, lon


def get_modis_tile(lat, lon):
    """
    Given a latitude and longitude, returns the MODIS sinusoidal projection tile number in string format (hXXvYY)
    """
    # Define the constants needed for the MODIS sinusoidal projection
    R = 6371007.181  # Earth's radius in meters
    MODIS_TILE_WIDTH = 1111950.51966667  # MODIS tile width in meters

    # Convert the latitude and longitude to radians
    lat_rad = lat * (3.141592653589793 / 180)
    lon_rad = lon * (3.141592653589793 / 180)

    # Calculate the x and y coordinates in the MODIS sinusoidal projection
    x = R * lon_rad * math.cos(lat_rad)
    y = R * lat_rad

    # Calculate the tile number
    tile_h = int((x + R * math.pi) / (MODIS_TILE_WIDTH))
    tile_v = int((R * math.pi / 2 - y) / (MODIS_TILE_WIDTH))
    tile_number = "h{:02d}v{:02d}".format(tile_h, tile_v)

    return tile_number


def find_mcd43(s2_mosaic_band_location):

    # Open the GeoTIFF file
    dataset = gdal.Open(s2_mosaic_band_location + '/B8A.tif')

    # Get the projection and geotransform information of the dataset
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    # Loop through each pixel and convert its coordinates to lat-lon
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows).astype(float)

    tile_all = []
    for x in range(0, cols + 1, 5000):
        for y in range(0, rows + 1, 5000):
            xp, yp = pixel2coord(x, y, geotransform)
            lat, lon = coord2latlon(xp, yp, projection)
            tile_number = get_modis_tile(lat, lon)
            tile_all.append(tile_number)

    most_common_tile = max(set(tile_all), key=tile_all.count)


if __name__ == '__main__':

    find_mcd43('/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0')
