#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
      	  for i in $d*.cnf; do
      	    if [[  $i = *"temp"* ]] ; then
      	      continue
      	    fi
      	    if [[  $3 == "1" ]]  ; then
		if [[ $i == *"/03_"* ]] || [[ $i == *"/04_"* ]] || [[ $i == *"/05_"* ]] || [[ $i == *"/06_"* ]] || [[ $i == *"/07_"* ]] ; then
			echo "start1" $i
			timeout 7200 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3

		fi
	     fi
	    if [[  $3 == "2" ]]  ; then
      	      if [[   $i == *"/08_"* ]] || [[   $i == *"/09_"* ]] || [[   $i == *"/10_"* ]]  ; then
	             echo "start2" $i
		     timeout 7200 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            fi
	         fi
	          if [[  $3 == "3" ]]  ; then
      	      if [[   $i == *"/11_"* ]] || [[ $i == *"/12_"* ]] || [[ $i == *"/13_"* ]] || [[ $i == *"/14_"* ]]  ; then
	              echo "start3" $i
		     timeout 7200 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            fi
	          fi
	          if [[  $3 == "4" ]]  ; then
      	      if [[   $i == *"/15_"* ]] || [[ $i == *"/16_"* ]]  ; then
	              echo "start4" $i
		      timeout 7200 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            fi
	         
	    #else 
		#   echo "no part" 
		   # timeout 7200 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 ""
	   fi
	    #echo done $i
     done
  done
done
