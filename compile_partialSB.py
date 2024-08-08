
import subprocess
import math
import time
import sys
import csv

#compile instances that ran out of time or reached a conflict - compile intermediate cnf that was saved

if __name__ == "__main__":
    partial_sb = ['03_iscas85_c1355.isc.cnf', '03_iscas85_c1908.isc.cnf', '05_iscas93_s1269.bench.cnf', '06_iscas99_b04.cnf', '07_blocks_right_6_p_t1.cnf', '13_ring2_r6_p_t10.cnf', '13_ring2_r8_p_t10.cnf', '13_ring2_r8_p_t8.cnf', '13_ring2_r8_p_t9.cnf', '13_ring_5_p_t10.cnf', '15_sort_num_s_4_p_t7.cnf', '15_sort_num_s_4_p_t8.cnf', '15_sort_num_s_4_p_t9.cnf', '15_sort_num_s_5_p_t2.cnf', '15_sort_num_s_5_p_t3.cnf', '15_sort_num_s_5_p_t4.cnf', '15_sort_num_s_6_p_t1.cnf', '15_sort_num_s_6_p_t2.cnf', '15_sort_num_s_7_p_t1.cnf', '16_uts_k2_p_t7.cnf', '16_uts_k2_p_t8.cnf', '16_uts_k2_p_t9.cnf', '16_uts_k3_p_t3.cnf', '16_uts_k3_p_t4.cnf', '16_uts_k4_p_t2.cnf']

    all_start = time.perf_counter()
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
               "obj"]
    for cnf in partial_sb:

        cnf_file =  "./input/Dataset_preproc/"+cnf.replace(".cnf", "_temphybrid_wmcdynamic.cnf")
        print(cnf_file)
        #part = sys.argv[2]
        stats_file = "./22percent_compilations_medium2_partialSB.csv"

        f = open(stats_file, "a+")
        writer = csv.writer(f, delimiter=',')


        writer.writerow([cnf_file])
        weights_file ="./input/Dataset_preproc/"+cnf.replace(".cnf", "_w3.w")
        res = subprocess.run(["./d4", "-dDNNF", cnf_file, "-wFile="+weights_file ], stdout=subprocess.PIPE, text=True)
        output = res.stdout
        print(output)
        output = output.split("\n")
        nb_nodes = 0
        nb_edges = 0
        comp_time = 0
        for line in output:
            if "Number of nodes:" in line:
                nb_nodes = int(line.split(" ")[-1].strip())
            elif "Number of edges:" in line:
                nb_edges = int(line.split(" ")[-1].strip())
            elif line.startswith("s "):
                scaled_wmc = float(line.split(" ")[-1].strip())
                if math.isinf(scaled_wmc):
                    scaled_wmc = int(line.split(" ")[-1].strip().split(".")[0])
                wmc = scaled_wmc
            elif "Final time:" in line:
                comp_time = float(line.split(" ")[-1].strip())
        end = time.perf_counter()
        print(end-all_start)

        log_line = [0, -1, -1, -1, -1, -1, nb_edges, nb_nodes, comp_time, wmc, 0, 0]
        writer.writerow(log_line)
        f.flush()
        f.close()
