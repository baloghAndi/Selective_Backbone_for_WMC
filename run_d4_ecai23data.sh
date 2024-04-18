#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
      	  for i in $d*.cnf; do
      	    if [[  $i = *"temp"* ]] ; then
      	      continue
      	    fi
            timeout 1800 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3 $4
     done
  done
done
