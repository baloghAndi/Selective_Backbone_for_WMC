#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
          for i in $d*.cnf; do
            if [[  $i = *"temp"* ]] ; then
              continue
            fi
            if [[  $3 = "1" ]]  ; then
              if   [[   $i = *"/05_"* ]] ||  [[   $i = *"/06_"* ]]  ||  [[   $i = *"/07_"* ]] ||  [[   $i = *"/08_"* ]]  ;  then
                      echo 1 $i
                      timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
              fi
            fi
            if [[  $3 = "2" ]]  ; then
              if  [[   $i = *"/09_"* ]]  ; then
                     echo 2 $i
                      timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
              fi
            fi
            if [[  $3 = "3" ]]  ; then
              if [[   $i = *"/10_"* ]] ||; then
                      echo 3 $i
                      timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
              fi
            fi
            if [[  $3 = "4" ]]  ; then
              if  [[   $i = *"/11_"* ]]  || [[   $i = *"/12"* ]]  || [[   $i = *"/13_"* ]]  || [[   $i = *"/14_"* ]]  || [[   $i = *"/15_"* ]] ||  [[   $i = *"/16_"* ]]  ; then
                      echo 4 $i
                      timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
              fi
            fi
          #echo "done" $i
     done
  done
done