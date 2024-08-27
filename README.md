# pyplot-veget
A series of python script to plot vegetation related fields

## Helper text as preliminary documentation
```
usage: pyplot-veget.py [-h] -i INPUT_TYPE INPUT_TYPE [-w WRT_OUT_FILEN] [-s] [-d]

options:
  -h, --help            show this help message and exit
  -i INPUT_TYPE INPUT_TYPE, --input_type INPUT_TYPE INPUT_TYPE
                        Combined input type: <inputtype> <filename>
  -w WRT_OUT_FILEN, --write_out WRT_OUT_FILEN
                        File name to be used for writing the data out
  -s, --substract       If set, attempt the difference between the two first dataset with a weight matrix
  -d, --desert          Add an auto-computed desert pseudo-PFT based on low NPP points
```

## Full test case:

```ruby
python pyplot-veget.py -i SEIB test-data/out_6k_TC/out_npppft -i MLRout test-data/MLRF/K8.csv -i reveals test-data/pollensHumanPFTTW14.csv -i ORCHIDEE test-data/ewembi_dgvm_2D-last.nc
```
