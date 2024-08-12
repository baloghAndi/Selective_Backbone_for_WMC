#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
      	  for i in $d*.cnf; do
      	    if [[  $i = *"temp"* ]] ; then
      	      continue
	          fi
      	    if  [[  $3 = "1" ]]  ; then
			  
      		      if [[   $i = *"/01_"* ]] || [[   $i = *"/02_"* ]] ||  [[   $i = *"/03_"* ]] || [[   $i = *"/04_"* ]] ||  [[   $i = *"/05_"* ]]; then
	        	      timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	     			echo "running" $i      
	     		 fi
	    	
	    	 elif [[  $3 = "2" ]]  ; then
      	      		if [[   $i = *"/06_"* ]] || [[   $i = *"/07_"* ]] || [[   $i = *"/08_"* ]]  || [[   $i = *"/09_"* ]] || [[   $i = *"/10_"* ]] || [[   $i = *"/14_"* ]]  ; then
	              		timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            	fi
	          
	          elif [[  $3 = "3" ]]  ; then
      	     		 if [[   $i = *"/11_"* ]]  || [[   $i = *"/12_"* ]] || [[   $i = *"/13_"* ]]; then
	              		timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            	fi
	          
	          elif [[  $3 = "4" ]]  ; then
      	     		 if [[   $i = *"/14_"* ]] || [[   $i = *"/15_"* ]] || [[   $i = *"/16_"* ]] ; then
	             	 	timeout 3600 python3 greedy_selective_backboneD4.py $d $i $alg_type $2 $3
	            	fi
	    fi

	    echo done $i
     done
  done
done
