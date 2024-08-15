#!/bin/bash


seed=1234
if [[  $1 = "1" ]]  ; then
      timeout 21600 python3 greedy_selective_backboneD4.py "./input/Dataset_preproc/" './input/Dataset_preproc/04_iscas89_s1494.bench.cnf' 'hybrid_wmc' 'dynamic' $1
elif [[  $1 = "2" ]]  ; then
      timeout 21600 python3 greedy_selective_backboneD4.py "./input/Dataset_preproc/" './input/Dataset_preproc/04_iscas89_s820.bench.cnf' 'hybrid_wmc' 'dynamic' $1
elif [[  $1 = "3" ]]  ; then
      timeout 21600 python3 greedy_selective_backboneD4.py "./input/Dataset_preproc/" './input/Dataset_preproc/04_iscas89_s832.bench.cnf' 'hybrid_wmc' 'dynamic' $1
elif [[  $1 = "4" ]]  ; then
      timeout 21600 python3 greedy_selective_backboneD4.py "./input/Dataset_preproc/" './input/Dataset_preproc/04_iscas89_s953.bench.cnf' 'hybrid_wmc' 'dynamic' $1
elif [[  $1 = "5" ]]  ; then
      timeout 21600 python3 greedy_selective_backboneD4.py "./input/Dataset_preproc/" './input/Dataset_preproc/05_iscas93_s967.bench.cnf' 'hybrid_wmc' 'dynamic' $1
else
	echo "wrong"
fi
