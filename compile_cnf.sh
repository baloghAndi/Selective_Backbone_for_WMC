#!/bin/bash

folder="./input/Dataset_preproc/"
for i in $folder*.cnf; do
    if [[ $i = *"temphybrid_wmcdynamic_22percent"* ]] ; then
      echo $i
      foldername="${i%%"temphybrid_wmcdynamic_22percent"*}"".cnf"
      filename="${foldername##*/}"
      if [[  $3 = "1" ]]  ; then
      	      if  [[   $i = *"/03_"* ]] ||  [[   $i = *"/04_"* ]] || [[   $i = *"/05_"* ]] ||  [[   $i = *"/06_"* ]]  ||  [[   $i = *"/07_"* ]] ||  [[   $i = *"/08_"* ]] ||  [[   $i = *"/09_"* ]] ;  then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	          fi
	          if [[  $3 = "2" ]]  ; then
      	      if [[   $i = *"/10_"* ]] || [[   $i = *"/11_"* ]] || [[   $i = *"/12_"* ]]  ; then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	          fi
	          if [[  $3 = "3" ]]  ; then
      	      if [[   $i = *"/13_"* ]] || [[   $i = *"/14_"* ]] ; then
	             timeout 3600 python3 compile_cnf.py $i $1
	            fi
	          fi
	          if [[  $3 = "4" ]]  ; then
      	      if [[   $i = *"/16_"* ]] || [[   $i = *"/15_"* ]] ; then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	      fi

      fi
    #fi
done
