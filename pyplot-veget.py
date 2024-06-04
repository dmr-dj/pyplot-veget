#!/usr/bin/python
# vim: set fileencoding=utf-8 :

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib import colors
import os
import netCDF4 as n4



# Utilities functions ...
# =======================

# from http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_closest(A,target) :
    import numpy as np
    return (np.abs(A-target)).argmin()
#enddef find_closest


def map_dataint(var_map, lons, lats, title, legend_label, colorlist,
                out_file=False, labels=None, extent = [-14, 50, 33, 72] ):

    """Generates a map of output data from `get_var` function plotted as categories.

    Keyword arguments:
    var_plot -- a masked array of the data. This must be 2-dimensions (lat, lon).
    lons -- a masked array of longitude values.
    lats -- a masked array of latitude values.
    title -- figure title for the map.
    legen_label -- label for the map's colorbar.
    colorlist -- a color list to create a colormap from (if None, use linear viridis segmented)
    out_file -- optional filepath to save the output image (default False).
    labels -- a series of discrete labels matching the numbers of colors to be used
    extent -- limits the extent in lat/lon of the figure (default values are Europe)
    """

    crs = ccrs.PlateCarree()
    min_bounds = round(np.min(var_map))
    max_bounds = round(np.max(var_map))
    nbs_bounds = round(max_bounds - min_bounds) +1

    fix_bounds = np.linspace(min_bounds, max_bounds, nbs_bounds)
    cmap = colors.ListedColormap(colorlist[min_bounds:max_bounds+1])

    fig, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=crs))
    ax.set_title(title)
    ax.set_global()
    ax.set_extent(extent)
    ax.gridlines()

    print(min_bounds, max_bounds, nbs_bounds)
    mesh = ax.pcolormesh(lons, lats, var_map, cmap=cmap, transform=crs,
                          vmin=min_bounds-0.5, vmax=max_bounds+0.5)

    cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.61, label=legend_label,ticks=fix_bounds)

    if labels != None:
        cbar.ax.set_yticklabels(labels[round(min_bounds):round(max_bounds)+1])

    ax.gridlines()
    ax.coastlines()
    fig.show()

    if out_file:
        fig.savefig(out_file)


#enddef

def map_dataflt(var_map, lons, lats, title, legend_label, cmap='viridis',
                out_file=False, extent = [-14, 50, 33, 72], masklmt=0.0 ):

    """Generates a map of output data from `get_var` function plotted as categories.

    Keyword arguments:
    var_plot -- a masked array of the data. This must be 2-dimensions (lat, lon).
    lons -- a masked array of longitude values.
    lats -- a masked array of latitude values.
    title -- figure title for the map.
    legen_label -- label for the map's colorbar.
    colorlist -- a color list to create a colormap from (if None, use linear viridis segmented)
    out_file -- optional filepath to save the output image (default False).
    labels -- a series of discrete labels matching the numbers of colors to be used
    extent -- limits the extent in lat/lon of the figure (default values are Europe)
    """

    crs = ccrs.PlateCarree()
    min_bounds = np.min(var_map)
    max_bounds = np.max(var_map)

    fig, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=crs))
    ax.set_title(title)
    ax.set_global()
    ax.set_extent(extent)
    ax.gridlines()

    var_tomap = np.ma.masked_less_equal(var_map,masklmt)

    mesh = ax.pcolormesh(lons, lats, var_tomap, cmap=cmap, transform=crs,
                          vmin=min_bounds-0.5, vmax=max_bounds+0.5)

    cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.61, label=legend_label)

    ax.gridlines()
    ax.coastlines()
    fig.show()

    if out_file:
        fig.savefig(out_file)


#enddef

def plot_barsInLON_int(colorlist,llat,llon1,llon2,lats_array,lons_array, data_array,pft_dict,show='False', Mean='False', title=""):
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(10,5))
  if Mean == 'True':
    for j in range(len(colorlist)):
      idx_lat = find_closest(lats_array[0,:],llat)
      idx_lon1 = find_closest(lons_array[:,0],llon1)
      idx_lon2 = find_closest(lons_array[:,0],llon2)
      data_mean = np.ma.mean(data_array[idx_lon1:idx_lon2,idx_lat,:],axis=0) * 100
      ax.bar(""+str(llon1)+":"+str(llon2),data_mean[j],bottom=np.ma.sum(data_mean[:j], axis = -1), label=pft_dict[j], color=colorlist[j])
    #data_mean = data_array[:,idx_lat,:]
    #ax.bar(lons_array[:,0],data_mean[:,j].filled(0),bottom=np.sum(data_mean[:,:j], axis = -1).filled(0), label=pft_dict[j], color=color)
  #endfor
  else:
    legend = 0
    for addlon in range(llon2-llon1+1):
      idx_lat = find_closest(lats_array[0,:],llat)
      idx_lon = find_closest(lons_array[:,0],llon1+addlon)
      # data_mean = np.ma.masked_where(data_array[idx_lon,idx_lat,:] <= 0,data_array[idx_lon,idx_lat,:])
      data_mean = data_array[idx_lon,idx_lat,:]
      for j in range(len(colorlist)):
        print(llon1, llon2, addlon,np.ma.sum(data_mean[:], axis = -1))
        if np.ma.sum(data_mean[:], axis = -1) > 0:
          if legend == 0:
            ax.bar(""+str(llon1+addlon),data_mean[j],bottom=np.ma.sum(data_mean[:j], axis = -1), label=pft_dict[j], color=colorlist[j])
            if j == len(colorlist) - 1:
               legend = 1
            #endif
          else:
            ax.bar(""+str(llon1+addlon),data_mean[j],bottom=np.ma.sum(data_mean[:j], axis = -1), color=colorlist[j])
          #endif
        #endif
    #data_mean = data_array[:,idx_lat,:]
    #ax.bar(lons_array[:,0],data_mean[:,j].filled(0),bottom=np.sum(data_mean[:,:j], axis = -1).filled(0), label=pft_dict[j], color=color)
  #endfor

  #enddif

  if show == 'True':
     ax.set_title(title)
     ax.legend()
     plt.show()
  #enddif
#enddef




# Define the correspondance between PFT names and colors

pft_color_dict = {
    "TeNEg" : "yellowgreen",
    "Med."  : "olive",
    "TeBSg" : "lime",
    "BNEg"  : "darkgreen",
    "BNSg"  : "darkred",
    "BBSg"  : "lightskyblue",
    "C3"    : "bisque",
    "C4"    : "gold",
    "HPFT"  : "gold",
    "DES"   : "yellow"
}


plot_type="SEIB"
# SEIB needs a directory as input
# path_data = "/home/acclimate/ibertrix/out_6k_EGU/out_npppft"
# path_data = "/home/acclimate/ibertrix/out_EWEMBI_egu/out_npppft"
path_data = "/home/acclimate/ibertrix/out_EWEMBI_BBSg/out_npppft"
path_data = "/home/acclimate/ibertrix/out_6k_BBSg/out_npppft"
path_data = "/home/acclimate/ibertrix/out_8k5/out_npppft"

# plot_type="reveals"
# file_toPLOT = "/home/acclimate/ibertrix/python-pour-Veget/pollens_onlyTree_TW1_sansCalluna.csv"
# file_toPLOT = "/home/acclimate/ibertrix/python-pour-Veget/pollensHerbaceousTW1.csv"
# file_toPLOT = "/home/acclimate/ibertrix/python-pour-Veget/pollensHumanPFTTW1.csv"

if plot_type == "SEIB":


  SEIB_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","C4", "DES"]
  # SEIB_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","C4"]

  n_pft=len(SEIB_pfts)
  pft_dict=SEIB_pfts[0:n_pft]


  grid_spacing="0.25" # 0.25


  n_lats = 156
  n_lons = 256

  step_per_degree=4
  lat_init=33
  lon_init=-14

  mask = pd.read_csv('/home/acclimate/ibertrix/SEIB-EU/params/landmask_'+grid_spacing+'deg.txt',header=None)
  landmask = mask.values[72:228,664:920]

  # Set lat/lon according to grid definition

  data_array = np.zeros((n_lons,n_lats))
  lons_array = np.zeros((n_lons,n_lats))
  lats_array = np.zeros((n_lons,n_lats))

  for j in range(n_lons):
    for i in range(n_lats):
      lats_array[j,i] = lat_init+i*1./step_per_degree
      lons_array[j,i] = lon_init+j*1./step_per_degree
    #endfor
  #endfor



  list_fichs=["07", "08", "09", "10", "13", "14", "15", "16"]
  data_array_nm = np.zeros((n_lons,n_lats,len(list_fichs)))-1.0


  for n_um in list_fichs:
    fich = ""+path_data+n_um+".txt"
    data = pd.read_csv(fich,header=None)
    data_array_nm[:,:,list_fichs.index(n_um)] = data.values[:,0:-1].T[:,::-1]
  #endfor

  data_array = np.ma.masked_less(data_array_nm,0)

  sum_array = np.sum(data_array,axis=-1)
  for i in range(data_array.shape[-1]):
    data_array[:,:,i] = data_array[:,:,i] / sum_array
  #endfor


  data_to_Plot_value = data_array.argmax(axis=-1)

  if SEIB_pfts[-1] == "DES":

    seuil_desert = 0.03
    data_to_Plot_value = np.ma.where(sum_array < seuil_desert,8,data_to_Plot_value)
    pft_dict_noDES = pft_dict[:-1]


  else:
    pft_dict_noDES = pft_dict

  #fi

  data_toPlot = np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_to_Plot_value,-1),0)

  titleforPlot=path_data
  data_forBars = np.zeros(data_array.shape)
  for i in range(data_forBars.shape[-1]):
    data_forBars[:,:,i] = np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_array[:,:,i],-1),0)
  #endfor

elif plot_type == "reveals":

  REVEALS_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","HPFT"]

  n_pft=len(REVEALS_pfts)# -1 # -1 to omit the HPFT ...
  pft_dict=REVEALS_pfts[0:n_pft+1]

  n_lats = 50
  n_lons = 100

  step_per_degree=1
  lat_init=33.5
  lon_init=-14.5

#  n_lats = 77
#  n_lons = 117

#  step_per_degree=2
#  lat_init=33.5
#  lon_init=-10.5
  # Set lat/lon according to grid definition

  data_array = np.zeros((n_lons,n_lats))
  lons_array = np.zeros((n_lons,n_lats))
  lats_array = np.zeros((n_lons,n_lats))

  for j in range(n_lons):
    for i in range(n_lats):
      lats_array[j,i] = lat_init+i*1./step_per_degree
      lons_array[j,i] = lon_init+j*1./step_per_degree
    #endfor
  #endfor

  dataPLOT = pd.read_csv(file_toPLOT)
  grid_toPLOT = np.zeros((n_lons,n_lats,n_pft+1)) -1.0

  for indx in range(dataPLOT.shape[0]):
    indx_lat = find_closest(lats_array[0,:],dataPLOT.LatDD.values[indx])
    indx_lon = find_closest(lons_array[:,0],dataPLOT.LonDD.values[indx])
    grid_toPLOT[indx_lon, indx_lat,:] = dataPLOT.values[indx,3:3+n_pft+1]
  #endfor

  data_toPlot = np.ma.where(grid_toPLOT[:,:,-1] < 100.0,np.ma.masked, grid_toPLOT[:,:,0:n_pft].argmax(axis=-1))
  titleforPlot = file_toPLOT

#endif

map_dataint(data_toPlot,lons_array,lats_array,titleforPlot,"PFT name", colorlist=[pft_color_dict[pft] for pft in pft_dict], labels=pft_dict)
# map_dataflt(grid_toPLOT[:,:,5], lons_array,lats_array,titleforPlot,"%"+str(pft_dict[5]), cmap="gist_earth", masklmt=5.0)

#llat = 61.44
#llon1 = 24
#llon2 = 30

llat = 68.5
llon1 = 13
llon2 = 29

plot_barsInLON_int([pft_color_dict[pft] for pft in pft_dict_noDES],llat,llon1,llon2,lats_array,lons_array, data_forBars,pft_dict_noDES,show='True', title=titleforPlot)


# The End of All Things (op. cit.)
