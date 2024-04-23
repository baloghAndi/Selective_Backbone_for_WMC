#!/bin/bash


seed=1234
  for alg_type in $1 ; do
     for d in "./input/Dataset_preproc/" ; do
        if [ $3 = "1" ] ; then
            files=("07_blocks_right_2_p_t1.cnf" "07_blocks_right_2_p_t2.cnf" "07_blocks_right_2_p_t3.cnf" "07_blocks_right_2_p_t4.cnf" "07_blocks_right_2_p_t5.cnf" "07_blocks_right_3_p_t1.cnf" "07_blocks_right_3_p_t2.cnf" "07_blocks_right_4_p_t1.cnf" "08_bomb_b10_t5_p_t1.cnf" "08_bomb_b5_t1_p_t1.cnf" "08_bomb_b5_t1_p_t2.cnf" "08_bomb_b5_t1_p_t3.cnf" "08_bomb_b5_t1_p_t4.cnf" "08_bomb_b5_t1_p_t5.cnf" "08_bomb_b5_t5_p_t1.cnf" "08_bomb_b5_t5_p_t2.cnf" "09_coins_p01_p_t1.cnf" "09_coins_p02_p_t1.cnf" "09_coins_p03_p_t1.cnf" "09_coins_p04_p_t1.cnf" "09_coins_p05_p_t1.cnf" "09_coins_p05_p_t2.cnf" "09_coins_p10_p_t1.cnf" )
        elif [ $3 = "2" ]  ; then
           files=("10_comm_p01_p_t1.cnf" "10_comm_p01_p_t2.cnf" "10_comm_p02_p_t1.cnf" "10_comm_p03_p_t1.cnf" "11_emptyroom_d12_g6_p_t1.cnf" "11_emptyroom_d12_g6_p_t2.cnf" "11_emptyroom_d16_g8_p_t1.cnf" "11_emptyroom_d16_g8_p_t2.cnf" "11_emptyroom_d20_g10_corners_p_t1.cnf" "11_emptyroom_d24_g12_p_t1.cnf" "11_emptyroom_d28_g14_corners_p_t1.cnf" "11_emptyroom_d4_g2_p_t10.cnf" "11_emptyroom_d4_g2_p_t1.cnf" "11_emptyroom_d4_g2_p_t2.cnf" "11_emptyroom_d4_g2_p_t3.cnf" "11_emptyroom_d4_g2_p_t4.cnf" "11_emptyroom_d4_g2_p_t5.cnf" "11_emptyroom_d4_g2_p_t6.cnf" "11_emptyroom_d4_g2_p_t7.cnf" "11_emptyroom_d4_g2_p_t8.cnf" "11_emptyroom_d4_g2_p_t9.cnf" "11_emptyroom_d8_g4_p_t1.cnf" "11_emptyroom_d8_g4_p_t2.cnf" "11_emptyroom_d8_g4_p_t3.cnf" "11_emptyroom_d8_g4_p_t4.cnf" )
        elif [ $3 = "3" ] ; then
                files=("12_flip_1_p_t10.cnf"  "12_flip_1_p_t1.cnf"  "12_flip_1_p_t2.cnf"  "12_flip_1_p_t3.cnf" "12_flip_1_p_t4.cnf"  "12_flip_1_p_t5.cnf"  "12_flip_1_p_t6.cnf"  "12_flip_1_p_t7.cnf" "12_flip_1_p_t8.cnf"  "12_flip_1_p_t9.cnf"  "12_flip_no_action_1_p_t10.cnf" "12_flip_no_action_1_p_t1.cnf"  "12_flip_no_action_1_p_t2.cnf"  "12_flip_no_action_1_p_t3.cnf" "12_flip_no_action_1_p_t4.cnf"  "12_flip_no_action_1_p_t5.cnf"  "12_flip_no_action_1_p_t6.cnf" "12_flip_no_action_1_p_t7.cnf" "12_flip_no_action_1_p_t8.cnf"  "12_flip_no_action_1_p_t9.cnf" "13_ring2_r6_p_t1.cnf"  "13_ring2_r6_p_t2.cnf"  "13_ring2_r6_p_t3.cnf"  "13_ring2_r8_p_t1.cnf" "13_ring2_r8_p_t2.cnf"  "13_ring2_r8_p_t3.cnf"  "13_ring_3_p_t1.cnf"  "13_ring_3_p_t2.cnf" "13_ring_3_p_t3.cnf"  "13_ring_3_p_t4.cnf"  "13_ring_4_p_t1.cnf"  "13_ring_4_p_t2.cnf" "13_ring_4_p_t3.cnf" "13_ring_5_p_t1.cnf" "13_ring_5_p_t2.cnf"  "13_ring_5_p_t3.cnf" "14_safe_safe_10_p_t10.cnf" "14_safe_safe_10_p_t1.cnf" "14_safe_safe_10_p_t2.cnf" "14_safe_safe_10_p_t3.cnf" "14_safe_safe_10_p_t4.cnf" "14_safe_safe_10_p_t5.cnf" "14_safe_safe_10_p_t6.cnf" "14_safe_safe_10_p_t7.cnf" "14_safe_safe_10_p_t8.cnf" "14_safe_safe_10_p_t9.cnf" "14_safe_safe_30_p_t1.cnf" "14_safe_safe_30_p_t2.cnf" "14_safe_safe_30_p_t3.cnf" "14_safe_safe_30_p_t4.cnf" "14_safe_safe_30_p_t5.cnf" "14_safe_safe_30_p_t6.cnf" "14_safe_safe_5_p_t10.cnf" )
        elif [ $3 = "4" ] ; then
            files=("14_safe_safe_5_p_t1.cnf" "14_safe_safe_5_p_t2.cnf" "14_safe_safe_5_p_t3.cnf" "14_safe_safe_5_p_t4.cnf" "14_safe_safe_5_p_t5.cnf" "14_safe_safe_5_p_t6.cnf" "14_safe_safe_5_p_t7.cnf" "14_safe_safe_5_p_t8.cnf" "14_safe_safe_5_p_t9.cnf" "15_sort_num_s_3_p_t10.cnf" "15_sort_num_s_3_p_t1.cnf" "15_sort_num_s_3_p_t2.cnf" "15_sort_num_s_3_p_t3.cnf" "15_sort_num_s_3_p_t4.cnf" "15_sort_num_s_3_p_t5.cnf" "15_sort_num_s_3_p_t6.cnf" "15_sort_num_s_3_p_t7.cnf" "15_sort_num_s_3_p_t8.cnf" "15_sort_num_s_3_p_t9.cnf" "15_sort_num_s_4_p_t1.cnf" "16_uts_k1_p_t10.cnf" "16_uts_k1_p_t1.cnf" "16_uts_k1_p_t2.cnf" "16_uts_k1_p_t3.cnf" "16_uts_k1_p_t4.cnf" "16_uts_k1_p_t5.cnf" "16_uts_k1_p_t6.cnf" "16_uts_k1_p_t7.cnf" "16_uts_k1_p_t8.cnf" "16_uts_k1_p_t9.cnf" "16_uts_k2_p_t1.cnf" "16_uts_k2_p_t2.cnf" "16_uts_k3_p_t1.cnf" )
        fi

        for i in "${files[@]}" ; do
          echo $i

          timeout 3600 python3 main_d4.py $d $d$i $alg_type $2 $3


        done
    done
done
