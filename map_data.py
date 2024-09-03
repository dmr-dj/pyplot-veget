#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__     = "Didier M. Roche, Isabeau Bertrix, Mathis Voisin, Jean-Yves Peterschmitt"
__copyright__  = "Copyright 2024, HIVE project"
__credits__    = ["Jean-Yves Peterschmitt"]
__license__    = "Apache-2.0"
__version__    = "0.3"
__maintainer__ = "Didier M. Roche"
__email__      = "didier.roche@lsce.ipsl.fr"
__status__     = "Development"

# This file is part of the pyplot-veget software and contains plotting routines

# Data handling
import numpy as np

# For plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mcolors

# For randomization
import random

# Got this one from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
#enddef discrete_cmap

def gen_list_colors(N):
    return [ random.choice(list(mcolors.CSS4_COLORS.keys())) for i in range(N) ]
#enddef gen_list_colors

def map_dataintPFT(var_map, lons, lats, title, legend_label, colorlist,
                out_file=False, labels=None, extent = [-14, 50, 33, 72] ):

    """Generates a map of output data from `get_var` function plotted as categories.

    Keyword arguments:
    var_plot -- a masked array of the data. This must be 2-dimensions (lat, lon).
    lons -- a masked array of longitude values.
    lats -- a masked array of latitude values.
    title -- figure title for the map.
    legend_label -- label for the map's colorbar.
    colorlist -- a color list to create a colormap from (if None, use linear viridis segmented)
    out_file -- optional filepath to save the output image (default False).
    labels -- a series of discrete labels matching the numbers of colors to be used
    extent -- limits the extent in lat/lon of the figure (default values are Europe)
    """

    crs = ccrs.PlateCarree()
    # ~ min_bounds = round(np.min(var_map))
    # ~ max_bounds = round(np.max(var_map))
    # ~ nbs_bounds = round(max_bounds - min_bounds) +1

    # We only keep the values that are expressed on the map (and non-masked)
    pft_list = np.unique(var_map.compressed()).tolist()
    nb_pft = len(pft_list)

    # Colorbar version jyp
    # ====================

    # Prepare the 'listed colormap' and the associated 'norm'
    cmap_col_list = []
    cmap_bounds_list = []
    pft_cb_labels_list = []

    for pft_value in pft_list:
      # ~ current_pft = pft_dic[pft_value]
      # ~ cmap_col_list.append(current_pft[0])
      cmap_col_list.append(colorlist[pft_value])
      cmap_bounds_list.append(pft_value - 0.5)
      # ~ pft_cb_labels_list.append(current_pft[1])
      pft_cb_labels_list.append(labels[pft_value])
    #endfor

    # We need one more bound than we have PFTs
    cmap_bounds_list.append(pft_value + 0.5)

    # Prepare a nice list of tickmarks to use on the colorbar, taking into
    # account the fact that the intervals may not all have the same
    # length, but we want the tickmarks in the MIDDLE of the intervals!
    pft_cb_ticks_list = []
    for idx in range(nb_pft):
        pft_cb_ticks_list.append((cmap_bounds_list[idx] + cmap_bounds_list[idx+1])/2.)
    #endfor

    # Prepare a colormap, and its associated 'norm'
    pft_cmap = mpl.colors.ListedColormap(cmap_col_list)

    # Use the following if you want to specify a color where we have
    # missing/masked values
    # pft_cmap.set_bad('0.7')
    # pft_cmap.set_bad('aliceblue')

    pft_norm = mpl.colors.BoundaryNorm(boundaries=cmap_bounds_list,
                                   ncolors=nb_pft)


    # End colorbar version jyp
    # ========================


    # ~ fix_bounds = np.linspace(min_bounds, max_bounds, nbs_bounds)
    # ~ cmap = mpl.colors.ListedColormap(colorlist[min_bounds:max_bounds+1])

    fig, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=crs))
    ax.set_title(title)
    ax.set_global()
    ax.set_extent(extent)
    ax.gridlines()

    # ~ v_print(V_WARN,"Bounds in map_dataint: ",min_bounds, max_bounds, nbs_bounds)
    # ~ mesh = ax.pcolormesh(lons, lats, var_map, cmap=cmap, transform=crs,
                          # ~ vmin=min_bounds-0.5, vmax=max_bounds+0.5)
    mesh = ax.pcolormesh(lons, lats, var_map,
                         cmap=pft_cmap,
                         norm=pft_norm,
                         transform=crs)

    # ~ cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.61,
                        # ~ label=legend_label,ticks=fix_bounds)
    cbar = plt.colorbar(mesh,
                       orientation='vertical',
                       shrink=0.61,
                       label=legend_label,
                       boundaries=cmap_bounds_list,
                       drawedges=True,
                       ticks=pft_cb_ticks_list,
                       spacing='uniform'
                       )
    cbar.set_ticklabels(pft_cb_labels_list)

    # ~ if labels != None:
        # ~ cbar.ax.set_yticklabels(labels[round(min_bounds):round(max_bounds)+1])

    ax.gridlines()
    ax.coastlines()
    fig.show()

    if out_file:
        fig.savefig(out_file)


#enddef map_dataintPFT

def map_dataint(var_map, lons, lats, title, legend_label, colorlist=None,
                out_file=False, labels=None, extent = [-14, 50, 33, 72], cmap='viridis' ):

    """Generates a map of output data from `get_var` function plotted as categories.
    Keyword arguments:
    var_plot -- a masked array of the data. This must be 2-dimensions (lat, lon).
    lons -- a masked array of longitude values.
    lats -- a masked array of latitude values.
    title -- figure title for the map.
    legend_label -- label for the map's colorbar.
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

    if colorlist != None:
      if len(colorlist)<nbs_bounds:
         #raise ValueError( "Colorlist provided is too short for the number of values to plot = "+str(nbs_bounds) )
         colorlist = gen_list_colors(nbs_bounds)
       #endif
      cmap = mpl.colors.ListedColormap(colorlist[0:nbs_bounds])
    elif cmap:
      cmap = discrete_cmap(nbs_bounds,base_cmap=cmap)
    #endif

    cmap.set_under(color="grey")

    fig, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=crs))
    ax.set_title(title)
    ax.set_global()
    ax.set_extent(extent)
    ax.gridlines()

    # ~ v_print(V_WARN,"Bounds in map_dataint: ",min_bounds, max_bounds, nbs_bounds)
    mesh = ax.pcolormesh(lons, lats, var_map, cmap=cmap, transform=crs,
                          vmin=min_bounds-0.5, vmax=max_bounds+0.5)

    cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.61,
                        label=legend_label,ticks=fix_bounds,extend='min')


    if labels != None:
        cbar.ax.set_yticklabels(labels[round(min_bounds):round(max_bounds)+1])

    ax.gridlines()
    ax.coastlines()
    fig.show()

    if out_file:
        fig.savefig(out_file)

#enddef map_dataint

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

    cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.61,
                        label=legend_label)

    ax.gridlines()
    ax.coastlines()
    fig.show()

    if out_file:
        fig.savefig(out_file)


#enddef

def plot_barsInLON_int(colorlist,llat,llon1,llon2,lats_array,lons_array
                      ,data_array,pft_dict,show='False', Mean='False'
                      , title=""):

  fig, ax = plt.subplots(figsize=(10,5))

  if Mean == 'True':
    for j in range(len(colorlist)):
      idx_lat = find_closest(lats_array[0,:],llat)
      idx_lon1 = find_closest(lons_array[:,0],llon1)
      idx_lon2 = find_closest(lons_array[:,0],llon2)
      data_mean = np.ma.mean(data_array[idx_lon1:idx_lon2,idx_lat,:],axis=0) * 100
      ax.bar(""+str(llon1)+":"+str(llon2),data_mean[j]
            ,bottom=np.ma.sum(data_mean[:j], axis = -1)
            ,label=pft_dict[j],color=colorlist[j])

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
        # ~ print(llon1, llon2, addlon,np.ma.sum(data_mean[:], axis = -1))
        if np.ma.sum(data_mean[:], axis = -1) > 0:
          if legend == 0:
            ax.bar(""+str(llon1+addlon),data_mean[j]
                  ,bottom=np.ma.sum(data_mean[:j], axis = -1)
                  ,label=pft_dict[j],color=colorlist[j])
            if j == len(colorlist) - 1:
               legend = 1
            #endif
          else:
            ax.bar(""+str(llon1+addlon),data_mean[j]
                  ,bottom=np.ma.sum(data_mean[:j], axis = -1)
                  , color=colorlist[j])
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

# The End of All Things (op. cit.)
