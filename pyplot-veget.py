#!/usr/bin/python
# vim: set fileencoding=utf-8 :

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib import colors
import os
import netCDF4 as n4

import argparse

# Generic definitions
# ===================

class plottypes :
  SEIB_plt = "SEIB"
  MLRout_plt = "MLRout"
  REVEALS_plt = "reveals"

known_plottypes = [plottypes.SEIB_plt, plottypes.MLRout_plt, plottypes.REVEALS_plt]

grid_choices  = { plottypes.SEIB_plt : "0.25", plottypes.MLRout_plt : "0.25", plottypes.REVEALS_plt : "0.5" }

# Utilities functions ...
# =======================


def check_python_version(limit_minor=10) :
  import sys
  if sys.version_info.minor <= limit_minor:
    raise RuntimeError( "Python version should be >= 3."+str(limit_minor) ) 
  #endif
#enddef


def parse_args() -> argparse.Namespace:
	
   parser = argparse.ArgumentParser()
   parser.add_argument("-p", '--plot_type', dest='str_plot_type', type=str, help='Type of plot that is needed [SEIB|reveals|MLRout]', required=True)
   parser.add_argument("-w", '--write_out', dest='wrt_out_filen', type=str, help='File name to be used for writing the data out', required=False)
   parser.add_argument("-i", '--input_fil', dest='input_dataset', type=str, help='File name or directory name to be used for input data and specified with the plot_type', required=True)
   parser.add_argument('-d', '--desert', dest='desert_flg', action='store_true')  # on/off flag
   args = parser.parse_args()
   
   return args
   
#enddef parse_args

# from http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_closest(A,target) :
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



def load_grid_latlon_EU (grid_spacing="0.25", lndmask=True):
	
    match grid_spacing:
        case "0.25":

          n_lats = 156
          n_lons = 256

          step_per_degree=4
          lat_init=33
          lon_init=-14
        
        case "0.5":
          
          n_lats = 77
          n_lons = 117

          step_per_degree=2
          lat_init=33.5
          lon_init=-10.5
          
        case "1.0":

          n_lats = 50
          n_lons = 100

          step_per_degree=1
          lat_init=33.5
          lon_init=-14.5
			
        case _: 
          pass
		
    #endmatch

    # Set lat/lon according to grid definition

    lons_array = np.zeros((n_lons,n_lats))
    lats_array = np.zeros((n_lons,n_lats))

    for j in range(n_lons):
      for i in range(n_lats):
        lats_array[j,i] = lat_init+i*1./step_per_degree
        lons_array[j,i] = lon_init+j*1./step_per_degree
      #endfor
    #endfor
	
    if lndmask :
      mask = pd.read_csv('inputdata/SEIB-EU/landmask_'+grid_spacing+'deg.txt',header=None)
      landmask = mask.values[72:228,664:920]	# Why are these integers there? Should not be OK with non 0.25 grid I think 2024-08-20
      return n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree, landmask
    else:
      return n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree
    #endif
    
#enddef load_grid_latlon_EU


def check_input_dataset( input_dataset, plot_type ):

  match plot_type :
      case plottypes.SEIB_plt:
      # SEIB needs a directory as input
      # ~ path_data = "test-data/out_6k_TC/out_npppft"
        if os.path.isdir(os.path.dirname(input_dataset)):
           return input_dataset
        #endif
      # REVEALS type need a csv file as input / MLRout as well
      case plottypes.REVEALS_plt | plottypes.MLRout_plt :
        if os.path.isfile(input_dataset):
           return input_dataset
        #endif
      case _:
        return None
      #       
  #endmatch        
      
#enddef check_input_dataset


def read_input_dataset( path_dataset, plot_type, pft_dict, data_map ):

  if plot_type == plottypes.SEIB_plt:
    
    # the typical output of SEIB used is a list of out_npppft[NN].txt file
    # Here the list of fichs defines the PFT numbers that will match the above list of PFTs
    list_fichs=["07", "08", "09", "10", "13", "14", "15", "16"]
    # ~ data_array_nm = np.zeros((n_lons,n_lats,len(list_fichs)))-1.0
    data_array_nm = np.zeros(data_map.shape+(len(list_fichs),))-1.0
    
    # SEIB input is a directory with a list of files, reading up the thing in one big table
    for n_um in list_fichs:
      fich = ""+path_dataset+n_um+".txt"
      data = pd.read_csv(fich,header=None)
      data_array_nm[:,:,list_fichs.index(n_um)] = data.values[:,0:-1].T[:,::-1]
    #endfor


    # here data_array_nm contains the lons, lats, pft_typ, npp values

    # Masking negative npp values if any
    data_array = np.ma.masked_less(data_array_nm,0)

    # Creating an array of proportion of NPP (relative to the total sum)
    sum_array = np.sum(data_array,axis=-1)
    for i in range(data_array.shape[-1]):
      data_array[:,:,i] = data_array[:,:,i] / sum_array
    #endfor

    # here data_array contains the lons, lats, pft_typ, of % of NPP

    # Retreiving the PFT number of maximum NPP production
    data_to_Plot_value = data_array.argmax(axis=-1)
    
    # data_to_Plot_value contains the PFT number with maximum NPP production
    
    # ~ # Sepcifying the data that needs to be written out
    # ~ data_to_wrt_valueS = data_array

    # Specific case handling when the artificial desert is set (absence of npp)
    if pft_dict[-1] == "DES":
      seuil_desert = 0.01
      # Keeping only values above the desertic threshold, if below set as PFT 8 (DESERT)
      data_to_Plot_value = np.ma.where(sum_array < seuil_desert,8,data_to_Plot_value)
      pft_dict_noDES = pft_dict[:-1]
    else: # no desert is required
      pft_dict_noDES = pft_dict
    #fi

    # Masking correctly the data with the landmask
    data_toPlot = np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_to_Plot_value,-1),0)

    # ~ # Preparing the data for the bar plotting if needed
    # ~ titleforPlot=path_dataset
    # ~ data_forBars = np.zeros(data_array.shape)
    # ~ for i in range(data_forBars.shape[-1]):
      # ~ data_forBars[:,:,i] = np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_array[:,:,i],-1),0)
    # ~ #endfor
    
    return data_toPlot

  elif plot_type == plottypes.REVEALS_plt:
	  
    dataPLOT = pd.read_csv(path_dataset)
    n_pft = len(pft_dict)
    # ~ grid_toPLOT = np.zeros((n_lons,n_lats,n_pft+1)) -1.0
    grid_toPLOT = np.zeros(data_map.shape+(n_pft+1,))-1.0
    
    for indx in range(dataPLOT.shape[0]):
      indx_lat = find_closest(lats_array[0,:],dataPLOT.LatDD.values[indx])
      indx_lon = find_closest(lons_array[:,0],dataPLOT.LonDD.values[indx])
      grid_toPLOT[indx_lon, indx_lat,:] = dataPLOT.values[indx,3:3+n_pft+1]
    #endfor

    data_toPlot = np.ma.where(grid_toPLOT[:,:,-1] < 100.0,np.ma.masked, grid_toPLOT[:,:,0:n_pft].argmax(axis=-1))
	  	  
    return data_toPlot
  #endif

#enddef read_input_dataset



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



# ----- MAIN PROGRAM -----


if __name__ == '__main__':

  check_python_version()

  got_args = parse_args()
  
  plot_type = got_args.str_plot_type
  
  if not plot_type in known_plottypes:
     raise RuntimeError("Unknown plot type, known are :", " ; ".join(str(e) for e in known_plottypes))
  #fi 


  # PFTs definition ....
  # ====================

  pft_list = []

  if plot_type == plottypes.SEIB_plt:

    # with or without desert
    if got_args.desert_flg :
      SEIB_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","C4", "DES"]
    else:
      SEIB_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","C4"]
    #endif
    
    pft_list = SEIB_pfts
    
  elif plot_type == plottypes.REVEALS_plt :
    REVEALS_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","HPFT"]
    # ~ REVEALS_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg"]

    pft_list = REVEALS_pfts

    n_pft=len(REVEALS_pfts)
    pft_dict=REVEALS_pfts[0:n_pft]


  n_pft=len(pft_list)
  pft_dict=pft_list[0:n_pft] # should be removed ....

  # READING DATASET
  # ======================
      
  path_dataset = got_args.input_dataset

  try:
      writeout_file=got_args.wrt_out_filen # is there a file to writeout?
  except:
	  pass
  #endtry

  # Loading grid depending on plot_type
  n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree, landmask = load_grid_latlon_EU(grid_spacing=grid_choices[plot_type])
  
  
  data_array = np.zeros((n_lons,n_lats)) # Base format for the whole thing: a lat,lon placeholder

  
  # Check the input data format, depending on plot type
  print(path_dataset, check_input_dataset( path_dataset, plot_type ))

  if check_input_dataset( path_dataset, plot_type ) == path_dataset:
	  print("Reading input dataset ...")
	  
	  # read_input_dataset returns:
	  # the max npp pft if SEIB is chosen
	  
	  data_toPlot = read_input_dataset( path_dataset, plot_type, pft_list, data_array ) 


  if plot_type == "SEIB":

    # ~ # If needed, write out the output file for data_to_wrt_valueS
    # ~ if writeout_file != None :
      # ~ f_to_write = open(writeout_file, 'w')
      # ~ for i in range(data_array.shape[0]):
        # ~ for j in range(data_array.shape[1]):
           # ~ if not np.ma.is_masked(data_to_wrt_valueS[i,j,0]):
             # ~ f_to_write.write(', '.join([str(lons_array[i,j]), str(lats_array[i,j]), ', '.join(str(x) for x in data_to_wrt_valueS[i,j,:])]))
             # ~ f_to_write.write('\n')
           # ~ # 
        # ~ #endfor
      # ~ #endfor
      # ~ f_to_write.close()
    # ~ #endif

    pass


  elif plot_type == "reveals":

    # ~ REVEALS_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg","C3","HPFT"]
    # ~ REVEALS_pfts=["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg"]

    # ~ n_pft=len(REVEALS_pfts)# -1 # -1 to omit the HPFT ...
    # ~ pft_dict=REVEALS_pfts[0:n_pft+1]

    # ~ n_lats = 50
    # ~ n_lons = 100

    # ~ step_per_degree=1
    # ~ lat_init=33.5
    # ~ lon_init=-14.5

  # ~ #  n_lats = 77
  # ~ #  n_lons = 117

  # ~ #  step_per_degree=2
  # ~ #  lat_init=33.5
  # ~ #  lon_init=-10.5
  # ~ # Set lat/lon according to grid definition

    # ~ data_array = np.zeros((n_lons,n_lats))
    # ~ lons_array = np.zeros((n_lons,n_lats))
    # ~ lats_array = np.zeros((n_lons,n_lats))

    # ~ for j in range(n_lons):
      # ~ for i in range(n_lats):
        # ~ lats_array[j,i] = lat_init+i*1./step_per_degree
        # ~ lons_array[j,i] = lon_init+j*1./step_per_degree
      # ~ #endfor
    # ~ #endfor

    pass


  elif plot_type == "MLRout":

    data_array = np.zeros((n_lons,n_lats))
    data_toPLOT = pd.read_csv(file_toPLOT)

    # Assuming that lat and lon are called as such
    lats=data_toPLOT.lat
    lons=data_toPLOT.lon
    
    # Following line depends on the data column name, needs to be updated
    data_brutto=data_toPLOT.kappa # this is a one dimensional array of lat, lon points

    for i in range(len(data_toPLOT.lat)):
      lat_ici=data_toPLOT.lat[i]
      lat_index=round((lat_ici-lat_init)*step_per_degree)-1
      lon_ici=data_toPLOT.lon[i]
      lon_index=round((lon_ici-lon_init)*step_per_degree)-1
      data_array[lon_index,lat_index] = data_brutto[i]
    #end for

    # case of MLRout what has been created is data_array(lon, lat)
    # Can be plotted with a simple map_dataflt below
  #endif


  map_dataint(data_toPlot,lons_array,lats_array,path_dataset,"PFT name", colorlist=[pft_color_dict[pft] for pft in pft_dict], labels=pft_dict)

  # ~ map_dataflt(grid_toPLOT[:,:,5], lons_array,lats_array,titleforPlot,"%"+str(pft_dict[5]), cmap="gist_earth", masklmt=5.0)

  # ~ map_dataflt(np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_array,-1),0), lons_array,lats_array,os.path.basename(file_toPLOT),"[1]", cmap="BrBG", masklmt=-5.0)


  #llat = 61.44
  #llon1 = 24
  #llon2 = 30

  # ~ llat = 68.5
  # ~ llon1 = 13
  # ~ llon2 = 29

  # ~ plot_barsInLON_int([pft_color_dict[pft] for pft in pft_dict_noDES],llat,llon1,llon2,lats_array,lons_array, data_forBars,pft_dict_noDES,show='True', title=titleforPlot)

#endif main

# The End of All Things (op. cit.)
