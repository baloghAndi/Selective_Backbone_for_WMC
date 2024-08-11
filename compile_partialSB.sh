#!/bin/bash
files=('./input/Dataset_preproc/15_sort_num_s_7_p_t1_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_6_p_t1_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_5_p_t2_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/16_uts_k2_p_t7_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring_5_p_t6_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring_5_p_t10_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_4_p_t9_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/06_iscas99_b04_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring2_r8_p_t9_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring2_r8_p_t8_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/07_blocks_right_6_p_t1_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring2_r8_p_t10_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_4_p_t8_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring2_r6_p_t9_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/05_iscas93_s1269.bench_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/13_ring2_r6_p_t10_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/07_blocks_right_5_p_t2_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_4_p_t7_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/03_iscas85_c1908.isc_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/03_iscas85_c1355.isc_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/07_blocks_right_3_p_t5_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/07_blocks_right_2_p_t8_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/07_blocks_right_2_p_t5_temphybrid_wmcdynamic.cnf'
'./input/Dataset_preproc/15_sort_num_s_7_p_t1_temphybrid_wmcdynamic_error2024-08-10-04:34:51.cnf')
for i in $files; do
 7200 python3 compile_cnf.py $i    #fi
done