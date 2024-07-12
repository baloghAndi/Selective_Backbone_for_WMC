#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
      	  for i in $d*.cnf; do
      	    if [[  $i = *"temp"* ]] ; then
      	      continue
      	    fi
	    if [[ $i = *uts_k5* ]] ; then
            	timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2
	    fi
	    echo done $i
     done
  done
done
