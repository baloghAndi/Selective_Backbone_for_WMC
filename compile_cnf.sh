#!/bin/bash

#time ./d4 -dDNNF ./input/Dataset_preproc/03_iscas85_c880.isc.cnf -wFile=./input/Dataset_preproc/03_iscas85_c880.isc_w3.w -mc
ecai23=('01_istance_K3_N15_M45_01.cnf'    '01_istance_K3_N15_M45_02.cnf'    '01_istance_K3_N15_M45_03.cnf'
              '01_istance_K3_N15_M45_04.cnf'    '01_istance_K3_N15_M45_05.cnf'    '01_istance_K3_N15_M45_06.cnf'
              '01_istance_K3_N15_M45_07.cnf'    '01_istance_K3_N15_M45_08.cnf'    '01_istance_K3_N15_M45_09.cnf'
              '01_istance_K3_N15_M45_10.cnf'    '02_instance_K3_N30_M90_01.cnf'
              '02_instance_K3_N30_M90_02.cnf'    '02_instance_K3_N30_M90_03.cnf'
              '02_instance_K3_N30_M90_04.cnf'    '02_instance_K3_N30_M90_05.cnf'
              '02_instance_K3_N30_M90_06.cnf'    '02_instance_K3_N30_M90_07.cnf'
              '02_instance_K3_N30_M90_08.cnf'    '02_instance_K3_N30_M90_09.cnf'
              '02_instance_K3_N30_M90_10.cnf'    '04_iscas89_s400_bench.cnf'    '04_iscas89_s420_1_bench.cnf'
              '04_iscas89_s444_bench.cnf'
              '04_iscas89_s526_bench.cnf'    '04_iscas89_s526n_bench.cnf'    '05_iscas93_s344_bench.cnf'
              '05_iscas93_s499_bench.cnf'    '06_iscas99_b01.cnf'    '06_iscas99_b02.cnf'    '06_iscas99_b03.cnf'
              '06_iscas99_b06.cnf'
              '06_iscas99_b08.cnf'    '06_iscas99_b09.cnf'    '06_iscas99_b10.cnf'    "07_blocks_right_2_p_t1.cnf"
              "07_blocks_right_2_p_t1.cnf"    "07_blocks_right_2_p_t2.cnf"    "07_blocks_right_2_p_t3.cnf"
              "07_blocks_right_2_p_t4.cnf"    "07_blocks_right_2_p_t5.cnf"    "07_blocks_right_3_p_t1.cnf"
              "07_blocks_right_3_p_t2.cnf"    "07_blocks_right_4_p_t1.cnf"    "08_bomb_b10_t5_p_t1.cnf"
              "08_bomb_b5_t1_p_t1.cnf"    "08_bomb_b5_t1_p_t2.cnf"    "08_bomb_b5_t1_p_t3.cnf"    "08_bomb_b5_t1_p_t4.cnf"
              "08_bomb_b5_t1_p_t5.cnf"    "08_bomb_b5_t5_p_t1.cnf"    "08_bomb_b5_t5_p_t2.cnf"    "09_coins_p01_p_t1.cnf"
              "09_coins_p02_p_t1.cnf"    "09_coins_p03_p_t1.cnf"    "09_coins_p04_p_t1.cnf"    "09_coins_p05_p_t1.cnf"
              "09_coins_p05_p_t2.cnf"    "09_coins_p10_p_t1.cnf"    "10_comm_p01_p_t1.cnf"    "10_comm_p01_p_t2.cnf"
              "10_comm_p02_p_t1.cnf"    "10_comm_p03_p_t1.cnf"    "11_emptyroom_d12_g6_p_t1.cnf"
              "11_emptyroom_d12_g6_p_t2.cnf"    "11_emptyroom_d16_g8_p_t1.cnf"    "11_emptyroom_d16_g8_p_t2.cnf"
              "11_emptyroom_d20_g10_corners_p_t1.cnf"    "11_emptyroom_d24_g12_p_t1.cnf"
              "11_emptyroom_d28_g14_corners_p_t1.cnf"    "11_emptyroom_d4_g2_p_t10.cnf"    "11_emptyroom_d4_g2_p_t1.cnf"
              "11_emptyroom_d4_g2_p_t2.cnf"    "11_emptyroom_d4_g2_p_t3.cnf"    "11_emptyroom_d4_g2_p_t4.cnf"
              "11_emptyroom_d4_g2_p_t5.cnf"    "11_emptyroom_d4_g2_p_t6.cnf"    "11_emptyroom_d4_g2_p_t7.cnf"
              "11_emptyroom_d4_g2_p_t8.cnf"    "11_emptyroom_d4_g2_p_t9.cnf"    "11_emptyroom_d8_g4_p_t1.cnf"
              "11_emptyroom_d8_g4_p_t2.cnf"    "11_emptyroom_d8_g4_p_t3.cnf"    "11_emptyroom_d8_g4_p_t4.cnf"
              "12_flip_1_p_t10.cnf"    "12_flip_1_p_t1.cnf"    "12_flip_1_p_t2.cnf"    "12_flip_1_p_t3.cnf"
              "12_flip_1_p_t4.cnf"    "12_flip_1_p_t5.cnf"    "12_flip_1_p_t6.cnf"    "12_flip_1_p_t7.cnf"
              "12_flip_1_p_t8.cnf"    "12_flip_1_p_t9.cnf"    "12_flip_no_action_1_p_t10.cnf"
              "12_flip_no_action_1_p_t1.cnf"    "12_flip_no_action_1_p_t2.cnf"    "12_flip_no_action_1_p_t3.cnf"
              "12_flip_no_action_1_p_t4.cnf"    "12_flip_no_action_1_p_t5.cnf"    "12_flip_no_action_1_p_t6.cnf"
              "12_flip_no_action_1_p_t7.cnf"    "12_flip_no_action_1_p_t8.cnf"    "12_flip_no_action_1_p_t9.cnf"
              "13_ring2_r6_p_t1.cnf"    "13_ring2_r6_p_t2.cnf"    "13_ring2_r6_p_t3.cnf"    "13_ring2_r8_p_t1.cnf"
              "13_ring2_r8_p_t2.cnf"    "13_ring2_r8_p_t3.cnf"    "13_ring_3_p_t1.cnf"    "13_ring_3_p_t2.cnf"
              "13_ring_3_p_t3.cnf"    "13_ring_3_p_t4.cnf"    "13_ring_4_p_t1.cnf"    "13_ring_4_p_t2.cnf"
              "13_ring_4_p_t3.cnf"    "13_ring_5_p_t1.cnf"    "13_ring_5_p_t2.cnf"    "13_ring_5_p_t3.cnf"
              "14_safe_safe_10_p_t10.cnf"    "14_safe_safe_10_p_t1.cnf"    "14_safe_safe_10_p_t2.cnf"
              "14_safe_safe_10_p_t3.cnf"    "14_safe_safe_10_p_t4.cnf"    "14_safe_safe_10_p_t5.cnf"
              "14_safe_safe_10_p_t6.cnf"    "14_safe_safe_10_p_t7.cnf"    "14_safe_safe_10_p_t8.cnf"
              "14_safe_safe_10_p_t9.cnf"    "14_safe_safe_30_p_t1.cnf"    "14_safe_safe_30_p_t2.cnf"
              "14_safe_safe_30_p_t3.cnf"    "14_safe_safe_30_p_t4.cnf"    "14_safe_safe_30_p_t5.cnf"
              "14_safe_safe_30_p_t6.cnf"    "14_safe_safe_5_p_t10.cnf"    "14_safe_safe_5_p_t1.cnf"
              "14_safe_safe_5_p_t2.cnf"    "14_safe_safe_5_p_t3.cnf"    "14_safe_safe_5_p_t4.cnf"
              "14_safe_safe_5_p_t5.cnf"    "14_safe_safe_5_p_t6.cnf"    "14_safe_safe_5_p_t7.cnf"
              "14_safe_safe_5_p_t8.cnf"    "14_safe_safe_5_p_t9.cnf"    "15_sort_num_s_3_p_t10.cnf"
              "15_sort_num_s_3_p_t1.cnf"    "15_sort_num_s_3_p_t2.cnf"    "15_sort_num_s_3_p_t3.cnf"
              "15_sort_num_s_3_p_t4.cnf"    "15_sort_num_s_3_p_t5.cnf"    "15_sort_num_s_3_p_t6.cnf"
              "15_sort_num_s_3_p_t7.cnf"    "15_sort_num_s_3_p_t8.cnf"    "15_sort_num_s_3_p_t9.cnf"
              "15_sort_num_s_4_p_t1.cnf"    "16_uts_k1_p_t10.cnf"    "16_uts_k1_p_t1.cnf"    "16_uts_k1_p_t2.cnf"
              "16_uts_k1_p_t3.cnf"    "16_uts_k1_p_t4.cnf"    "16_uts_k1_p_t5.cnf"    "16_uts_k1_p_t6.cnf"
              "16_uts_k1_p_t7.cnf"    "16_uts_k1_p_t8.cnf"    "16_uts_k1_p_t9.cnf"    "16_uts_k2_p_t1.cnf"
              "16_uts_k2_p_t2.cnf"    "16_uts_k3_p_t1.cnf")

folder="./input/Dataset_preproc/"
for i in $folder*.cnf; do
    if [[ $i = *"temphybrid_wmcdynamic"* ]] ; then
      echo $i
      foldername="${i%%"_temphybrid_wmcdynamic"*}"".cnf"
      filename="${foldername##*/}"
      if  [[ "$ecai23" =~ (" "|^)$filename(" "|$) ]] ; then
        echo $filename
      else
             if [[  $1 == "1" ]]  ; then
	            	if [[ $i == *"/03_"* ]] || [[ $i == *"/04_"* ]] || [[ $i == *"/05_"* ]] || [[ $i == *"/06_"* ]] || [[ $i == *"/07_"* ]] ; then
		            	echo "start1" $i
			           timeout 3600 python3 compile_cnf.py $i $1

		            fi
	           fi
	          if [[  $1 == "2" ]]  ; then
      	      if [[   $i == *"/08_"* ]] || [[   $i == *"/09_"* ]]   ; then
	             echo "start2" $i
		             timeout 3600 python3 compile_cnf.py $i $1
	            fi
	         fi
	          if [[  $1 == "3" ]]  ; then
      	      if [[   $i == *"/10_"* ]] || [[   $i == *"/11_"* ]] || [[ $i == *"/12_"* ]] || [[ $i == *"/13_"* ]] || [[ $i == *"/14_"* ]]  ; then
	              echo "start3" $i
		                 timeout 3600 python3 compile_cnf.py $i $1
	            fi
	          fi
	          if [[  $1 == "4" ]]  ; then
      	      if [[   $i == *"/15_"* ]] || [[ $i == *"/16_"* ]]  ; then
	              echo "start4" $i
		             timeout 3600 python3 compile_cnf.py $i $1
	            fi
        echo "compile" $filename
        fi

      fi
    fi
done