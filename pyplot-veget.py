#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__     = "Didier M. Roche, Isabeau Bertrix, Mathis Voisin"
__copyright__  = "Copyright 2024, HIVE project"
__credits__    = ["Jean-Yves Peterschmitt"]
__license__    = "Apache-2.0"
__version__    = "0.7"
__maintainer__ = "Didier M. Roche"
__email__      = "didier.roche@lsce.ipsl.fr"
__status__     = "Development"

# Base os stuff
import os
import argparse

# Data handling
import numpy as np
import netCDF4 as n4
import pandas as pd
import ast

from map_data import *

# Generic definitions
# ===================

# VERBOSITY LEVELS

V_INFO = 1
V_WARN = 2
V_ERRO = 3

V_dict = { V_INFO : "INFO: " , V_WARN : "WARN: ", V_ERRO : "SOFT ERROR: "}

# Known types for models ...

class inputtypes :
  SEIB_plt     = "SEIB"
  MLRout_plt   = "MLRout"
  REVEALS_plt  = "reveals"
  ORCHIDEE_plt = "ORCHIDEE"
  CARAIB_plt   = "CARAIB"
#endclass

known_inputtypes = [
        inputtypes.SEIB_plt,
        inputtypes.MLRout_plt,
        inputtypes.REVEALS_plt,
        inputtypes.ORCHIDEE_plt,
        inputtypes.CARAIB_plt
        ]

grid_choices  = {
        inputtypes.SEIB_plt     : "0.25",
        inputtypes.MLRout_plt   : "0.25",
        inputtypes.REVEALS_plt  : "1.0",
        inputtypes.ORCHIDEE_plt : "0.25",
        inputtypes.CARAIB_plt   : "0.25"
        }

class data_geoEurope :

    def __init__(self, geodata, lons, lats, path, pftdict, inputtype):
        self.geodata = geodata
        self.lons = lons
        self.lats = lats
        self.path = path
        self.pftdict = pftdict
        self.inputtype = inputtype
    #enddef

    def add_lndmsk(self, lndmsk):
        self.lndmsk=lndmsk
    #enddef

#endclass

# Define the correspondance between PFT names and colors

with open('inputdata/PFT_color_scheme.txt') as f:
    data = f.read()
#endwith

PFTs_color_NAMES = ast.literal_eval(data)



PFT_list_SEIB =     ["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg"
                    ,"C3","C4"]
PFT_list_reveals =  ["TeNEg","Med.","TeBSg","BNEg","BNSg","BBSg"
                    ,"C3"]#, "HPFT"]
PFT_list_ORCHIDEE = ["solnu", "TrEg","TrSg","TeNEg","TeBEg","TeBSg"
                    ,"BNEg","BBSg","BNSg","TeC3","TrC3","BC4"]
PFT_list_MLRout = None

PFT_list_CARAIB = ["C3h","C3d","C4","BSg artic shrubs",                   # 4
                   "BSg B/Te cold shrubs","BSg Te warm shrubs",           # 6
                   "BEg B/Te cold shrubs","BEg Te warm shrubs",           # 8
                   "BEg xeric shrubs","Subdesertic shrubs",               # 10
                   "Tr Shrubs","NEg B/Te cold trees","NEg Te cool trees", # 13
                   "NEg sMed trees","NEg mMed trees","NEg subTr trees",   # 16
                   "NSg B/Te cold trees","NSg subTr swamp trees",         # 18
                   "BEg mMed trees","BEg tMed trees","BEg subTr trees",   # 21
                   "BSg B/Te cold trees","BSg Te cool trees",             # 23
                   "BSg Te warm trees","TrBRg","TrBEg"                    # 26
                   ]

PFT_list_choices = {
        inputtypes.SEIB_plt     : PFT_list_SEIB,
        inputtypes.ORCHIDEE_plt : PFT_list_ORCHIDEE,
        inputtypes.MLRout_plt   : PFT_list_MLRout,
        inputtypes.REVEALS_plt  : PFT_list_reveals,
        inputtypes.CARAIB_plt   : PFT_list_CARAIB
        }

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
   parser.add_argument("-i", '--input_type', dest='input_type',
                       type=str, nargs=2, action='append',
                       help='Combined input type: <inputtype> <filename>'
                       ,required=True)
   parser.add_argument("-w", '--write_out', dest='wrt_out_filen',
                       type=str,
                       help='File name to be used for writing the data out'
                       ,required=False)
   parser.add_argument('-s', '--substract', dest='substract_flg',
                       action='store_true',
                       help='If set, attempt the difference between the two first dataset with a weight matrix'
                       ,required=False)  # on/off flag
   parser.add_argument('-d', '--desert', dest='desert_flg', action='store_true',
                       help='Add an auto-computed desert pseudo-PFT based on low NPP points'
                       ,required=False)  # on/off flag
   parser.add_argument('-v', '--verbosity', action="count",
                       help="increase output verbosity (e.g., -vv is more than -v)")
   args = parser.parse_args()

   return args

#enddef parse_args

# from http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_closest(A,target) :
    return (np.abs(A-target)).argmin()
#enddef find_closest


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

          lndmask = False

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
      landmask = mask.values[72:228,664:920]    # Why are these integers there? Should not be OK with non 0.25 grid I think 2024-08-20
      return n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree, landmask
    else:
      return n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree, None
    #endif

#enddef load_grid_latlon_EU


def check_input_dataset( input_dataset, plot_type ):

  match plot_type :

      case inputtypes.SEIB_plt:
      # SEIB needs a directory and many subfiles as input, testing directory presence
        if os.path.isdir(os.path.dirname(input_dataset)):
           return input_dataset
        #endif
      #endcase

      # REVEALS type need a csv file as input / MLRout as well
      case inputtypes.REVEALS_plt | inputtypes.MLRout_plt | inputtypes.ORCHIDEE_plt | inputtypes.CARAIB_plt :
        if os.path.isfile(input_dataset):
           return input_dataset
        #endif
      #endcase

      case _:
        return None
      #endcase

  #endmatch

#enddef check_input_dataset


def read_input_dataset( path_dataset, plot_type, pft_dict, data_map ):

  if plot_type == inputtypes.SEIB_plt:

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

  elif plot_type == inputtypes.REVEALS_plt:

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

  elif plot_type == inputtypes.MLRout_plt:

    data_array = np.zeros(data_map.shape)
    data_toPLOT = pd.read_csv(path_dataset)

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

    return data_array

  elif plot_type == inputtypes.CARAIB_plt:

    dataPLOT = pd.read_csv(path_dataset)
    n_pft = len(pft_dict)
    grid_toPLOT = np.ma.zeros(data_map.shape+(n_pft,))-1.0

    for indx in range(dataPLOT.shape[0]):
        gotten_lon = dataPLOT.LonDD.values[indx]
        if gotten_lon >= 180.0:
            gotten_lon -= 360.0
        #endif
        indx_lat = find_closest(lats_array[0,:],dataPLOT.LatDD.values[indx])
        indx_lon = find_closest(lons_array[:,0],gotten_lon)
        grid_toPLOT[indx_lon, indx_lat,:] = dataPLOT.values[indx,2:]
    #endfor

    # ~ masked_data_PLOT = np.ma.where(grid_toPLOT < 0, np.ma.masked, grid_toPLOT)

    data_toPlot = np.ma.where(grid_toPLOT[:,:,-1] < 0,np.ma.masked, grid_toPLOT[:,:,0:n_pft].argmax(axis=-1))
    # ~ data_toPlot = masked_data_PLOT[:,:,0:n_pft].argmax(axis=-1)


    return data_toPlot

  elif plot_type == inputtypes.ORCHIDEE_plt:

    dst = n4.Dataset(path_dataset)
    n_pft = len(dst.variables['veget'][:]) # this variable contains an axis with values 0:12, so 13 PFTs
    vegfrac = dst.variables['vegetfrac'] # Vegetation Fraction, time, nvegtyp, lat,lon
    vegfrc = vegfrac[-1,:,:,:]  # Taking last time step
    # ~ print(" :: ", len(pft_list))
    remap_fracveg = np.zeros(data_map.shape+(n_pft,))
    for i in range(n_pft):
      remap_fracveg[:,:,i] = vegfrc[i,::-1,:].T
    #endfor

    var_to_PLOT = np.ma.masked_less_equal(remap_fracveg.argmax(axis=-1),0)

    return var_to_PLOT

  #endif

#enddef read_input_dataset


# Non Contiguous is dataset2 ...
def compare_PFT_weights_NC(dataset1, dataset2, PFT_12_weights):

    # Local variables
    sum_distance = 0
    count_points = 0
    datasetout = np.ma.zeros((dataset2.geodata.shape),np.int32)

    # dataset2 is the non continuous one, hence looping on it
    for i in range(dataset2.geodata.shape[0]):
        for j in range(dataset2.geodata.shape[1]):
            if not dataset2.geodata.mask[i,j]:
              llat = dataset2.lats[i,j]
              llon = dataset2.lons[i,j]

              idx_lat = find_closest(dataset1.lats[0,:],llat)
              idx_lon = find_closest(dataset1.lons[:,0],llon)
              if not dataset1.geodata.mask[idx_lon,idx_lat]:
                try:
                  weigth_value = PFT_12_weights[dataset1.geodata[idx_lon,idx_lat],dataset2.geodata[i,j]]
                  sum_distance += int(weigth_value)
                  datasetout[i,j] = int(weigth_value)
                  count_points += 1
                except:
                  datasetout[i,j] = -99
                  pass
                #endtry
            else:
                datasetout[i,j] = -99
            #endif
        #endfor
    #endfor

    datasetout = np.ma.masked_where(datasetout<0, datasetout)
    return sum_distance, datasetout, np.max(PFT_12_weights)*int(count_points)
#enddef


def get_PFT_weights(data01,data02):

    std_path = "inputdata/poids_PFTs_"
    ext_file = ".csv"

    dataplottype01 = data01.inputtype
    dataplottype02 = data02.inputtype

    file01 = std_path+dataplottype01+"_"+dataplottype02+ext_file
    file02 = std_path+dataplottype02+"_"+dataplottype01+ext_file

    exists01 = os.path.isfile(file01)
    exists02 = os.path.isfile(file02)

    PFT_list01 = data01.pftdict
    PFT_list02 = data02.pftdict

    if exists01 | exists02:
       if exists01:
           PFT_weights = pd.read_csv(file01)
       else:
           PFT_weights = pd.read_csv(file02)
       #endif
       PFT_weights_read = PFT_weights.values[:,1:] # 1: to suppress the labelling column ...
    else:
        # No weights file / break / raise exception?
        PFT_weights_read = None
    #endif

    v_print(V_INFO,"Check consistency ::")
    v_print(V_INFO,"Dataset 01 :: "+dataplottype01)
    v_print(V_INFO,"PFTs nb & list:", len(PFT_list01))
    v_print(V_INFO,PFT_list01)
    v_print(V_INFO,"===================")
    v_print(V_INFO,"Dataset 02 :: "+dataplottype02)
    v_print(V_INFO,"PFTs nb & list:", len(PFT_list02))
    v_print(V_INFO,PFT_list02)
    v_print(V_INFO,"===================")
    v_print(V_INFO,"Weights table:", PFT_weights_read.shape)

    if ( not PFT_weights_read.shape[0] == len(PFT_list01) ) or ( not PFT_weights_read.shape[1] == len(PFT_list02) ):
        raise IndexError('PFT shapes in base list and weights table do not conform')
    #endif

    return PFT_weights_read

#enddef get_PFT_weights


# ----- MAIN PROGRAM -----


if __name__ == '__main__':

  check_python_version()

  got_args = parse_args()

  # Use the verbosity print code adapted from: https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
  if got_args.verbosity:
      def _v_print(*verb_args):
          if verb_args[0] > (3 - got_args.verbosity):
              print(V_dict[verb_args[0]] + " ".join(str(e) for e in verb_args[1:])) # V_dict above is defining the type of message level
  else:
      _v_print = lambda *a: None  # do-nothing function

  global v_print
  v_print = _v_print
  # <- End verbosity code ...

  # Looping over the series of inputs (in the form of input_type, path_dataset)

  full_data_list = []

  for nb_if in range(len(got_args.input_type)):

    plot_type = got_args.input_type[nb_if][0]
    path_dataset = got_args.input_type[nb_if][1]

    if not plot_type in known_inputtypes:
       raise RuntimeError("Unknown plot type, known are :", " ; ".join(str(e) for e in known_inputtypes))
    #fi

    pft_list = PFT_list_choices[plot_type]

    if got_args.desert_flg :
      pft_list.append("DES")
    #endif
    if not pft_list is None:
      n_pft=len(pft_list)
      pft_dict=pft_list[0:n_pft] # should be removed ....
    #endif

    # READING DATASET
    # ======================

    try:
        writeout_file=got_args.wrt_out_filen # is there a file to writeout?
    except:
        pass
    #endtry

    # Loading grid depending on plot_type
    n_lats, n_lons, lats_array, lons_array, lat_init, lon_init, step_per_degree, landmask = load_grid_latlon_EU(grid_spacing=grid_choices[plot_type])

    data_array = np.zeros((n_lons,n_lats)) # Base format for the whole thing: a lat,lon placeholder


    # Check the input data format, depending on plot type
    v_print(V_WARN,"Check input data format: ", path_dataset, check_input_dataset( path_dataset, plot_type ))

    if check_input_dataset( path_dataset, plot_type ) == path_dataset:
      v_print(V_INFO,"Reading input dataset ...")

        # read_input_dataset returns:
        # the max npp pft if SEIB is chosen

      data_toPlot = read_input_dataset( path_dataset, plot_type, pft_list, data_array )
    #endif

    # Add the data thus obtained in the data list containing all datasets ...
    full_data_list.append(data_geoEurope(data_toPlot, lons_array, lats_array, path_dataset, pft_list, plot_type))
    if not landmask is None:
      full_data_list[-1].add_lndmsk(landmask)
    #endif

  #end_for

  for nb_data in range(len(full_data_list)):

    to_plot = full_data_list[nb_data]

    # ~ pft_color_dict = PFT_color_choices[to_plot.inputtype]
    pft_color_dict = PFTs_color_NAMES

    if to_plot.inputtype == inputtypes.MLRout_plt:
      map_dataflt(np.ma.masked_less(np.ma.where(to_plot.lndmsk.T[:,::-1]>0,to_plot.geodata,-1),0)
                 ,to_plot.lons,to_plot.lats,os.path.basename(to_plot.path),"[1]"
                 , cmap="BrBG", masklmt=-5.0
                 )
    elif to_plot.inputtype == inputtypes.CARAIB_plt:

      map_dataint(np.ma.masked_less(np.ma.where(to_plot.lndmsk.T[:,::-1]>0,to_plot.geodata,-1),0)
                 ,to_plot.lons,to_plot.lats,to_plot.path
                 ,"PFT name", colorlist=[pft_color_dict[pft] for pft in to_plot.pftdict]
                 , labels=to_plot.pftdict
                 )
    else:
      map_dataint(to_plot.geodata,to_plot.lons,to_plot.lats,to_plot.path
                 ,"PFT name", colorlist=[pft_color_dict[pft] for pft in to_plot.pftdict]
                 , labels=to_plot.pftdict
                 )
    #endif

  # endfor


  # Section to compute the distance between the two datasets ...

  if got_args.substract_flg:

    distance_color_dict={0:'lime',1:"darkorange",2:"darkred",3:"indigo"}

    dataset_01 = full_data_list[0]
    dataset_02 = full_data_list[1]

    v_print(V_INFO,"Trying to calculate the distance from ",dataset_01.inputtype, " to ", dataset_02.inputtype)

    match dataset_02.inputtype :

        case inputtypes.SEIB_plt | inputtypes.ORCHIDEE_plt :
            PFT_weights_toUse = get_PFT_weights(dataset_01, dataset_02)
            distance_value, distance_map, distance_max = compare_PFT_weights_NC(dataset_01, dataset_02, PFT_weights_toUse)
        #endcase

        case inputtypes.REVEALS_plt :
            PFT_weights_toUse = get_PFT_weights(dataset_01, dataset_02)
            distance_value, distance_map, distance_max = compare_PFT_weights_NC(dataset_01, dataset_02, PFT_weights_toUse)
        #endcase

        case _:
            distance_value, distance_map, distance_max = None, None, None
        #endcase

    #endmatch

    # ~ distance_value, distance_map, distance_max = compare_PFT_weights_NC(full_data_list[0], full_data_list[1], PFT_weights_SEIB_reveals)
    map_dataint(distance_map,full_data_list[1].lons,full_data_list[1].lats
               ,""+str(distance_value)+"/"+str(distance_max),"Distance value [1]"
               ,colorlist=[distance_color_dict[values] for values in distance_color_dict]
               )
  #endif substract

  input("Press Any Key to close program")


#endif main


# The End of All Things (op. cit.)




# Old bits to be kept for now ...


  # ~ map_dataflt(grid_toPLOT[:,:,5], lons_array,lats_array,titleforPlot,"%"+str(pft_dict[5]), cmap="gist_earth", masklmt=5.0)

  # ~ map_dataflt(np.ma.masked_less(np.ma.where(landmask.T[:,::-1]>0,data_array,-1),0), lons_array,lats_array,os.path.basename(file_toPLOT),"[1]", cmap="BrBG", masklmt=-5.0)


  #llat = 61.44
  #llon1 = 24
  #llon2 = 30

  # ~ llat = 68.5
  # ~ llon1 = 13
  # ~ llon2 = 29

  # ~ plot_barsInLON_int([pft_color_dict[pft] for pft in pft_dict_noDES],llat,llon1,llon2,lats_array,lons_array, data_forBars,pft_dict_noDES,show='True', title=titleforPlot)





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
