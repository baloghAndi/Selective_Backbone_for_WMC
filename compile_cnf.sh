#!/bin/bash

folder="./input/Dataset_preproc/"
for i in $folder*.cnf; do
    if [[ $i = *"temphybrid_wmcdynamic_partialSB"* ]] ; then
      echo $i
      foldername="${i%%"temphybrid_wmcdynamic_partialSB"*}"".cnf"
      filename="${foldername##*/}"
      if [[  $3 = "1" ]]  ; then
      	      if  [[   $i = *"/03_"* ]] ||  [[   $i = *"/05_"* ]] || [[   $i = *"/06_"* ]] ||  [[   $i = *"/16_"* ]]  ;  then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	 fi
	if [[  $3 = "2" ]]  ; then
      	     if [[   $i = *"/07_"* ]]  ; then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	          fi
	 if [[  $3 = "3" ]]  ; then
      	      if [[   $i = *"/13_"* ]]  ; then
	             timeout 3600 python3 compile_cnf.py $i $1
	            fi
	  fi
	  if [[  $3 = "4" ]]  ; then
      	      if [[   $i = *"/15_"* ]]  ; then
	              timeout 3600 python3 compile_cnf.py $i $1
	            fi
	      fi

     fi
    #fi
done
