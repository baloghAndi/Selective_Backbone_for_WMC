#!/bin/bash

folder="./input/Dataset_preproc/"
for i in $folder*.cnf; do
    if [[ $i = *"temphybrid_wmcdynamic_22percent_medium4_partialSB"* ]] ; then
      echo $i
      foldername="${i%%"temphybrid_wmcdynamic_p_22percent_medium4_partialSBs"*}"".cnf"
      filename="${foldername##*/}"
      timeout 3600 python3 compile_cnf.py $i

     fi
    #fi
done
