#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    find_mosaic_mcd43.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        12/03/2023 18:36

import numpy as np
from osgeo import gdal

def find_mcd43(s2_mosaic_band):

    # Open the GeoTIFF file
    ds = gdal.Open(s2_mosaic_band)

    # Get the geotransform information (affine transformation matrix)
    geotransform = ds.GetGeoTransform()

    # Get the raster band
    band = ds.GetRasterBand(1)

    # Get the array of raster data
    data = band.ReadAsArray()

    # Get the spatial reference of the UTM projection
    utm_sr = osr.SpatialReference()
    utm_sr.ImportFromWkt(ds.GetProjection())

    # Create a spatial reference for the WGS84 coordinate system (latitude and longitude)
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.SetWellKnownGeogCS('WGS84')

    # Create a transformation object to convert from UTM to WGS84
    transform = osr.CoordinateTransformation(utm_sr, wgs84_sr)

    # Calculate the x and y coordinates of each pixel in UTM
    rows, cols = data.shape
    x_coords = np.arange(cols) * geotransform[1] + geotransform[0]
    y_coords = np.arange(rows) * geotransform[5] + geotransform[3]

    # Convert the UTM coordinates to latitude and longitude
    lon, lat, _ = transform.TransformPoints(np.vstack((x_coords, y_coords)).T).T

    # Create the 2D grid of longitude and latitude values
    lon = lon.reshape((rows, cols))
    lat = lat.reshape((rows, cols))

    # Print the shape of the latitude and longitude grids
    print('Latitude grid shape:', lat.shape)
    print('Longitude grid shape:', lon.shape)

if main == '__main__':

    find_mcd43('/gws/nopw/j04/qa4ecv_vol3/S2GM/hr-albedo/data/S2GM_T10_20220711_20220720_s2gm-LaCrau-v2_STD_v2.0.1/tile_0/B8A.tif')
