# Created by Dr. Rui Song at 03/02/2022
# Email: rui.song@ucl.ac.uk

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import gdal
import os

input_directory = '/gws/nopw/j04/qa4ecv_vol3/HR_Albedo/SouthSudan/36PVT/36PVT/S2A_MSIL1C_20201118T081231_N0209_R078_T36PVT_20201118T093043_UCL.SAFE'
save_directory = '/gws/nopw/j04/qa4ecv_vol2/HR_Alebdo/eumetsat_fig/SouthSudan/'

#################################
def readcdict(lutfile, minV, maxV):
    if (lutfile!=None):
        red=[]
        green=[]
        blue=[]
        lut=open(lutfile, 'r')
        i=0
        for line in lut:
            tab=line.split(',')
            tab[0]=tab[0].strip()
            tab[1]=tab[1].strip()
            tab[2]=tab[2].strip()
            val= minV+((i/255.0)*(maxV-minV))
            red.insert(i,(val , np.float32(tab[0])/255.0, np.float32(tab[0])/255.0))
            green.insert(i,(val, np.float32(tab[1])/255.0, np.float32(tab[1])/255.0))
            blue.insert(i,(val, float(tab[2])/255.0, np.float32(tab[2])/255.0))
            i= i+1
    return {'red':red, 'green':green, 'blue':blue}
##############################

# set color lookup table
lut_color = './CLUT/lut_colors.txt'
cdict = readcdict(lut_color, 0, 1)
my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

for file in os.listdir(input_directory):
    if file.endswith('_UCL_bhr.jp2'):

        s2_brf = gdal.Open('%s/%s' % (input_directory, file))
        s2_cols = s2_brf.RasterXSize
        s2_rows = s2_brf.RasterYSize
        s2_brf_array = s2_brf.GetRasterBand(1).ReadAsArray(0, 0, s2_cols, s2_rows) / 1.e4

        fig, ax = plt.subplots()
        im = ax.imshow(s2_brf_array, cmap=my_cmap, vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        plt.axis('off')
        plt.title('%s' %file[-15:-12])
        plt.savefig('%s/%s.png' % (save_directory, file[0:-4]), dpi=400)
        plt.close()