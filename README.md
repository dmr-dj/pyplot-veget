# pyplot-veget
A series of python script to plot vegetation related fields

## Helper text as preliminary documentation
```
usage: pyplot-veget.py [-h] -i INPUT_TYPE INPUT_TYPE [-w WRT_OUT_FILEN] [-s] [-d] [-l LIMIT_NPP_VAL] [-v] [-m MEAN_VALUE]

options:
  -h, --help            show this help message and exit
  -i INPUT_TYPE INPUT_TYPE, --input_type INPUT_TYPE INPUT_TYPE
                        Combined input type: <inputtype> <filename>
  -w WRT_OUT_FILEN, --write_out WRT_OUT_FILEN
                        File name to be used for writing the data out
  -s, --substract       If set, attempt the difference between the two first dataset with a weight matrix
  -d, --desert          Add an auto-computed desert pseudo-PFT based on low NPP points
  -l LIMIT_NPP_VAL, --limit_npp LIMIT_NPP_VAL
                        A value to decipher between dominant and non-dominant values of npp for PFTs
  -v, --verbosity       increase output verbosity (e.g., -vv is more than -v)
  -m MEAN_VALUE, --mean_yrs MEAN_VALUE
                        Do the mean over the mean_value years of the file
```

## Full test case:

```ruby
python pyplot-veget.py -i SEIB test-data/out_6k_TC/out_npppft -i MLRout test-data/MLRF/K8.csv -i reveals test-data/pollensHumanPFTTW14.csv -i ORCHIDEE test-data/ewembi_dgvm_2D-last.nc
```
