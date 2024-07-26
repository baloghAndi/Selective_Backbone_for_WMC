import subprocess
import math
import time
import sys
import csv

if __name__ == "__main__":
	all_start = time.perf_counter()
	columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
			   "obj"]

	cnf_file =  sys.argv[1]
	part = sys.argv[2]
	stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/8percent_compilations_part"+str(part)+".csv"

	f = open(stats_file, "a+")
	writer = csv.writer(f, delimiter=',')


	writer.writerow([cnf_file])
	weights_file = cnf_file.replace(".cnf", "_w3.w" ) #"./input/Dataset_preproc/03_iscas85_c880.isc_w3.w"
	weights_file = weights_file.replace("_temphybrid_wmcdynamic", "" ) #"./input/Dataset_preproc/03_iscas85_c880.isc_w3.w"
	res = subprocess.run(["./d4", "-dDNNF", cnf_file, "-wFile="+weights_file ], stdout=subprocess.PIPE, text=True)
	output = res.stdout
	# print(output)
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
