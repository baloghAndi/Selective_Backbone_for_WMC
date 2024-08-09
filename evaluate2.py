import csv
import math
import os
import re
import statistics
import time
from itertools import count

import matplotlib.colors as mcolors
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylatex as px
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from torch._C._return_types import sort

import CNFmodelBDD as _cnfBDD


class Logger:

    def __init__(self, filename, column_names, expr_data, out_folder, compile=False):
        print(os.getcwd())
        self.f = open(filename,"a+")
        self.writer = csv.writer(self.f, delimiter=',')
        self.progress_log =  open(filename.replace(".csv", ".txt"),"a+")
        self.column_names = column_names
        self.expr_data = expr_data
        self.out_folder = out_folder
        self.compile = compile

    def log_expr(self, expr_name):
        self.writer.writerow([expr_name])
        self.writer.writerow(self.column_names)
        if len(self.expr_data.data) > 0: #TODO why
            self.expr_data.all_expr_data[expr_name] = self.expr_data.data.copy()
        self.expr_data.exprs.append(expr_name)
        self.expr_data.data = []
        self.f.flush()
    def log(self, row):
        self.writer.writerow(row)
        self.expr_data.data.append(row)
        self.f.flush()

    def log_error(self, message):
        for m in message:
            self.writer.writerow(m)
        self.f.flush()
    def close(self):
        if len(self.expr_data.data) > 0:
            self.expr_data.all_expr_data[self.expr_data.exprs[-1]] = self.expr_data.data.copy()
        self.f.flush()
        self.f.close()

    def set_start_time(self, start):
        self.start_time = start

    def get_time_elapsed(self):
        return time.perf_counter()-self.start_time



class ExprData:

    def __init__(self, column_names):
        self.data = []
        self.all_expr_data = {}
        self.exprs = []
        self.full_expr_name = []
        self.column_names = column_names
        self.init_compilation_time = {}
        self.finish_time = {}
        self.nb_completed_assignments = {}

    def read_nocompile_stats_file(self, filename, full_expr_only=True, min_nb_expr=1, padding=True, filter_timeout=False, filter_conflict=False, cutoff={}):
        """
        Remove exprs that only have 1 solution -as in iniial mc is 1
        """
        # print(filename)
        self.data = []
        self.all_expr_data = {}
        self.exprs = []
        self.full_expr_name = []
        self.filename = filename
        print(filename)
        with (open(filename) as csvfile):
            reader = csv.reader(csvfile, delimiter=',')
            prev_line = []
            line_index = 0
            for line in reader:
                # print(line)
                if len(line) == 1 or ".cnf" in line[0]: #if first line or start of new expr
                    line_index = 0
                    save_expr_name = line[0]
                    if save_expr_name.count(".")>1:
                        save_expr_name = save_expr_name.replace(".", "_", save_expr_name.count(".")-1) #actually first . will always be ./input so should skipp that
                    # print("expr:",line)
                    if len(self.data) > 0: #next expr is starting, need to save current expr data
                        if self.exprs[-1] in self.all_expr_data:
                            print("duplicate expr: ",  self.exprs[-1])
                            exit(88)
                        if len(cutoff) > 0:
                            self.all_expr_data[self.exprs[-1]] = self.data.copy()[:int(cutoff[self.exprs[-1]])+1]
                            prev_line = self.all_expr_data[self.exprs[-1]][-1]
                            self.finish_time[self.exprs[-1]] = float(prev_line[self.column_names.index("time")])  # - self.init_compilation_time[self.exprs[-1]]
                            self.nb_completed_assignments[self.exprs[-1]] = prev_line[self.column_names.index("p")]
                        else:
                            self.all_expr_data[self.exprs[-1]] = self.data.copy()
                            self.finish_time[self.exprs[-1]] = float(prev_line[self.column_names.index("time")]) # - self.init_compilation_time[self.exprs[-1]]
                            self.nb_completed_assignments[self.exprs[-1]] = self.data[-1][self.column_names.index("p")]
                    if len(self.data) == 0 and len(self.exprs) > 0: #last expr finished
                        self.exprs.pop()
                        self.full_expr_name.pop()
                    self.full_expr_name.append(save_expr_name)
                    expr_short_name = save_expr_name.split("/")[-1]
                    self.exprs.append(expr_short_name) #add expr name - should only add if it has data
                    self.data = []
                elif self.column_names[0] in line:
                    continue
                else:
                    # print(line)
                    typed_line = []
                    # [int(x) if i != 1 else x for i,x in enumerate(line[:-1]) ]
                    for i,x in enumerate(line[:-1]): #why do we ignore last column? it might be the empty value after the end line separator
                        # print(x)
                        if i!=1:
                            if "[" in x:
                                typed_line.append(x)
                            # elif i ==3 or "." in x or i == self.column_names.index("MC") :
                            #     typed_line.append(float(x)) #read mc as float
                            else:
                                # print(type(x), float(x))
                                # typed_line.append(int(x))
                                # print(x)
                                typed_line.append(float(x))
                        else:
                            typed_line.append(x)
                    typed_line.append(float(line[-1]))
                    self.data.append(typed_line)
                    prev_line = line
                if line_index == 1:
                    self.init_compilation_time[self.exprs[-1]] = float(line[-1])
                    # print("init compilation ", self.exprs[-1])
                line_index += 1
            if len(self.data) > 0:
                if len(cutoff) > 0:
                    self.all_expr_data[self.exprs[-1]] = self.data.copy()[:int(cutoff[self.exprs[-1]])+1]
                    prev_line = self.all_expr_data[self.exprs[-1]][-1]
                    self.finish_time[self.exprs[-1]] = float(prev_line[self.column_names.index("time")])  # - self.init_compilation_time[self.exprs[-1]]
                    self.nb_completed_assignments[self.exprs[-1]] = prev_line[self.column_names.index("p")]
                else:
                    self.all_expr_data[self.exprs[-1]] = self.data.copy()
                    self.nb_completed_assignments[self.exprs[-1]] = self.data[-1][self.column_names.index("p")]
                    self.finish_time[self.exprs[-1]] = float(self.data[-1][self.column_names.index("time")])
            if len(self.data) == 0 and len(self.exprs) > 0:
                self.exprs.pop()
                self.full_expr_name.pop()

        print("@@@@@@@@@@@@@@@@@@@@@@ read stat file:", self.filename, len(self.full_expr_name))

        mc_index = self.column_names.index("MC")

            # print(len(self.all_expr_data[expr]))


        if padding:
            for expr in self.all_expr_data.keys():  # in case mc got to 0 make it look like expr finished -- why is this needed?
                last_row = self.all_expr_data[expr][-1]
                nb_vars_index = self.column_names.index("nb_vars")
                p_index = self.column_names.index("p")
                if last_row[nb_vars_index] == -1 or last_row[nb_vars_index] == "-1":
                    last_row[nb_vars_index] = self.get_nb_vars(expr)
                if last_row[p_index] != last_row[nb_vars_index]:# and last_row[mc_index] == 0: extra condition to onlu change id conflict was encounter
                    missing_rows = int(last_row[nb_vars_index] - last_row[p_index])
                    print(expr,missing_rows, "missing rows")
                    p = last_row[p_index]
                    add_row = last_row.copy()
                    for i in range(1, missing_rows + 1):
                        add_row[p_index] = i + p
                        add_row[mc_index:] = len(last_row[mc_index:]) * [0]

                        self.all_expr_data[expr].append(add_row)

    def get_line(self, line_index):
        lines = {}
        for e in self.all_expr_data.keys():
            expr_init = self.all_expr_data[e][line_index]
            lines[e] = expr_init
        return lines
    def read_stats_file(self, filename, full_expr_only=True, min_nb_expr=1, padding=True, filter_timeout=False, filter_conflict=False):
        """
        Remove exprs that only have 1 solution -as in iniial mc is 1
        """
        # print(filename)
        self.data = []
        self.all_expr_data = {}
        self.exprs = []
        self.full_expr_name = []
        self.filename = filename
        self.no_data_expr = []
        self.only_init_compilation = []
        expr_count = 0
        print(filename)
        with (open(filename) as csvfile):
            reader = csv.reader(csvfile, delimiter=',')
            prev_line = []
            line_index = 0
            for line in reader:
                # print(line)
                if len(line) == 1 or ".cnf" in line[0]: #if first line or start of new expr
                    expr_count +=1
                    line_index = 0
                    save_expr_name = line[0]
                    if save_expr_name.count(".")>1:
                        save_expr_name = save_expr_name.replace(".", "_", save_expr_name.count(".")-1) #actually first . will always be ./input so should skipp that
                    print("expr:",line)
                    if len(self.data) > 0: #next expr is starting, need to save current expr data
                        if self.exprs[-1] in self.all_expr_data:
                            print("duplicate expr: ",  self.exprs[-1])
                            exit(8)
                        self.all_expr_data[self.exprs[-1]] = self.data.copy()
                        self.finish_time[self.exprs[-1]] = float(prev_line[self.column_names.index("time")]) # - self.init_compilation_time[self.exprs[-1]]
                        self.nb_completed_assignments[self.exprs[-1]] = self.data[-1][self.column_names.index("p")]
                    if len(self.data) == 0 and len(self.exprs) > 0: #last expr finished
                        temp = self.exprs.pop()
                        self.full_expr_name.pop()
                        self.no_data_expr.append(temp)

                    self.full_expr_name.append(save_expr_name)
                    expr_short_name = save_expr_name.split("/")[-1]
                    self.exprs.append(expr_short_name) #add expr name - should only add if it has data
                    self.data = []
                elif self.column_names[0] in line:
                    continue
                else:
                    # print(line)
                    typed_line = []
                    # [int(x) if i != 1 else x for i,x in enumerate(line[:-1]) ]
                    for i,x in enumerate(line[:-1]): #why do we ignore last column? it might be the empty value after the end line separator
                        # print(x)
                        if i!=1:
                            # if "[" in x:
                            #     typed_line.append(x)
                            #     print("why? ", x)
                            #     exit(3)
                            # elif i ==3 or "." in x or i == self.column_names.index("MC") :
                            #     typed_line.append(float(x)) #read mc as float
                            # else:
                            x_val = float(x.strip())
                            if math.isinf(x_val):
                                x_val = np.float128(x)
                            typed_line.append(x_val)
                        else:
                            typed_line.append(x)
                    x_val = float(line[-1])
                    if math.isinf(x_val):
                        x_val = np.float128(line[-1])
                    typed_line.append(x_val)
                    self.data.append(typed_line)
                    prev_line = line
                if line_index == 1:
                    self.init_compilation_time[self.exprs[-1]] = float(line[-1])
                    # print("init compilation ", self.exprs[-1])
                line_index += 1
            if len(self.data) > 0:
                self.all_expr_data[self.exprs[-1]] = self.data.copy()
                self.nb_completed_assignments[self.exprs[-1]] = self.data[-1][self.column_names.index("p")]
                self.finish_time[self.exprs[-1]] = float(self.data[-1][self.column_names.index("time")])
            if len(self.data) == 0 and len(self.exprs) > 0: #remove expr if it has no data at all
                temp = self.exprs.pop()
                self.full_expr_name.pop()
                self.no_data_expr.append(temp)

        print("@@@@@@@@@@@@@@@@@@@@@@ read stat file:", self.filename, len(self.full_expr_name), len(self.all_expr_data), expr_count, len(self.no_data_expr))

        mc_index = self.column_names.index("MC")

            # print(len(self.all_expr_data[expr]))
        remove_expr = []
        #remove exprs that have only the initial compilation
        if min_nb_expr >= 1:
            for expr in self.all_expr_data:
                # print("data len:", len(self.all_expr_data[expr]))
                if len(self.all_expr_data[expr]) <= min_nb_expr:
                    if expr not in remove_expr:
                        remove_expr.append(expr)
                        if expr not in self.no_data_expr:
                            self.no_data_expr.append(expr)
                        print("remove: ", expr, " from ", self.filename)
        # if min_nb_expr  > 0:
        #     #remove exprs who only have less then 2 models, initial and another one
        #     mc_index = self.column_names.index("MC")
        #     for expr in self.all_expr_data.keys():
        #         if self.all_expr_data[expr][0][mc_index] <= 2 or self.all_expr_data[expr][0][mc_index] == "1.0":
        #             print("expr has 1 solution ", expr)
        #             if expr not in remove_expr:
        #                 remove_expr.append(expr)
        #                 print("remove: ", expr)
        # remove exprs that have inf as mc or wmc - if min_nb = -1
        wmc_index = self.column_names.index("WMC")
        for expr in self.all_expr_data.keys():
            # if math.isinf(self.all_expr_data[expr][0][mc_index]) or math.isinf(self.all_expr_data[expr][0][wmc_index]) :
            if np.isinf(self.all_expr_data[expr][0][mc_index]) or np.isinf(self.all_expr_data[expr][0][wmc_index]) :
                print("expr has inf mc or wmc ", expr, self.all_expr_data[expr][0][mc_index], self.all_expr_data[expr][0][wmc_index] )
                if expr not in remove_expr:
                    remove_expr.append(expr)
                    print("remove: ", expr)

        # remove exprs that did not finish
        if full_expr_only:
            for expr in self.all_expr_data.keys():
                last_row = self.all_expr_data[expr][-1]
                # if last_row[3] == -1 or last_row[3] == "-1":
                #     last_row[3] = self.get_nb_vars(expr)
                if last_row[0] != last_row[3]: #if p not eq to number of variables
                    if expr not in remove_expr:
                        remove_expr.append(expr)
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!! Remove unfinished expr: ", expr, filename, last_row[0], last_row[3])
                        # exit(8)
        #filter timeout expr:
        if filter_timeout:
            mc_index = self.column_names.index("MC")
            nb_vars_index = self.column_names.index("nb_vars")
            p_index = self.column_names.index("p")
            for expr in self.all_expr_data.keys():
                last_row = self.all_expr_data[expr][-1]
                if last_row[p_index] != last_row[nb_vars_index] and last_row[mc_index] != 0 :
                    print("timeout : ", expr)
                    if expr not in remove_expr:
                        remove_expr.append(expr)
        if filter_conflict:
            mc_index = self.column_names.index("MC")
            nb_vars_index = self.column_names.index("nb_vars")
            p_index = self.column_names.index("p")
            for expr in self.all_expr_data.keys():
                last_row = self.all_expr_data[expr][-1]
                if last_row[p_index] != last_row[nb_vars_index] and last_row[mc_index] == 0 :
                    print("conflict : ", expr)
                    if expr not in remove_expr:
                        remove_expr.append(expr)
        if len(remove_expr) > 0:
            self.remove_expr = remove_expr.copy()
            self.removed_expr_data = {}
            for ex in remove_expr:
                self.exprs.remove(ex)
                data = self.all_expr_data.pop(ex)
                self.removed_expr_data[ex] = data
            if len(remove_expr) > 0:
                print("---------------------------REMOVE EXPRS----------------------")
                print(self.filename, remove_expr)

                # exit(12345678)

        if padding:
            print("-------------------------padding--------------------------")
            for expr in self.all_expr_data.keys():  # in case mc got to 0 make it look like expr finished -- why is this needed?
                last_row = self.all_expr_data[expr][-1]
                nb_vars_index = self.column_names.index("nb_vars")
                p_index = self.column_names.index("p")
                if last_row[nb_vars_index] == -1 or last_row[nb_vars_index] == "-1":
                    last_row[nb_vars_index] = self.get_nb_vars(expr)
                if last_row[p_index] != last_row[nb_vars_index]:# and last_row[mc_index] == 0: extra condition to onlu change id conflict was encounter
                    missing_rows = int(last_row[nb_vars_index] - last_row[p_index])
                    print(expr,missing_rows, "missing rows")
                    p = last_row[p_index]
                    add_row = last_row.copy()
                    for i in range(1, missing_rows + 1):
                        add_row[p_index] = i + p
                        add_row[mc_index:] = len(last_row[mc_index:]) * [0]

                        self.all_expr_data[expr].append(add_row)

    def get_finishing_times(self):
        return self.finish_time

    def get_nb_vars(self, expr):
        if "n_vars" in self.column_names:
            n = int(self.all_expr_data[expr][0][self.column_names.index("n_vars")])
            if n >= 0:
                return int(n)
        if "nb_vars" in self.column_names:
            n = int(self.all_expr_data[expr][0][self.column_names.index("nb_vars")])
            if n >= 0:
                return int(n)
        expr_index = self.exprs.index(expr)
        expr_file = self.full_expr_name[expr_index]
        # print("stop", expr_file, len(self.full_expr_name), len(self.exprs))
        # print(self.filename, expr, expr_file)
        # exit(666)
        with open(expr_file, "r") as f:
            content = f.readlines()
            nb_vars = int(content[0].strip().split(" ")[2])
        return int(nb_vars)

    def get_metric_wrt_initial_per_expr(self, metric, obj):
        """
        Calculate ratio of each p with respect to the initial MC/BDD ratio
        :return:
        """
        # result = {e: [] for e in self.exprs}
        result = {}
        mc_index = self.column_names.index("MC")
        if "WMC" == obj:
            mc_index  = self.column_names.index("WMC")
        #actually could say : mc_index  = self.column_names.index(metric)
        # bdd_index = self.column_names.index("SDD size")
        size_index = self.column_names.index("edge_count")
        # bdd_index = self.column_names.index("dag_size")
        smallest_n = 600000
        for expr in self.exprs:
            curren_n = self.get_nb_vars(expr)
            # if curren_n >= 100:
            #     print("----------------------------skipped:", expr)
            #     continue
            if curren_n < smallest_n:
                smallest_n = curren_n
                # print(smallest_n, expr)

            ratios = []
            if metric == "ratio" or metric == "weighted_ratio":
                init_ratio = self.all_expr_data[expr][0][mc_index] / self.all_expr_data[expr][0][size_index]
            elif metric == "MC" or metric == "WMC":
                init_ratio = self.all_expr_data[expr][0][mc_index]
            elif metric == "BDD" or metric == "edge_count":
                init_ratio = self.all_expr_data[expr][0][size_index]
            ratios.append(1.0)
            # ratios.append(init_ratio)
            for i in range(1, len(self.all_expr_data[expr])):
                if metric == "ratio" or metric == "weighted_ratio":
                    if self.all_expr_data[expr][i][size_index] != 0:
                        r = self.all_expr_data[expr][i][mc_index] / self.all_expr_data[expr][i][size_index]
                    else:
                        r =0
                elif metric == "MC" or metric == "WMC":
                    r = self.all_expr_data[expr][i][mc_index]
                elif metric == "BDD" or metric == "edge_count":
                    r = self.all_expr_data[expr][i][size_index]
                ratios.append(r / init_ratio)
                # ratios.append(r)
            result[expr] = ratios.copy()
            # print(self.filename, expr, smallest_n)
        return result, smallest_n


def sample_data(data, smallest_n):
    n = len(data)
    # if n < smallest_n:
        # print("Less assignments then should be according to nb vars")
    # print(len(data),smallest_n)
    #return false in case there is less data then smallest n . smallest n is smallest nb variables, there can be less data in case exprs didn't finish
    return [data[int((i*n)/smallest_n)] for i in range(smallest_n)], n >= smallest_n


def get_best_variable_percentage(sample_size = 50):
    #given a set of experiments :
    # 1. read stats files - no padding - all should have the same nb iterations - if not eliminate it
    # 2. calculate adjusted ratio(AR) for each iteration - ratio with respect to initial compilation wmc and size
    # 3. nb_improvement = for each p - number of iteration - calculate how many instanes have AR > 1 ( improvement)
    # 4. percentage var = p that maximizes 3.nb_improvement(nb_p)
    # break ties for 4 with max lb_p ; lb_p =  min AR per each p
    FOLDER = "Dataset_preproc"
    nb_vars_data = {}
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4
    nb_exprs = 0
    all_expr_names_count = {}
    folder = "./results_aaai2/Dataset_preproc_hybrid_wmc/"
    type = "dynamic"
    nb_vars_data = {}
    stats_file = folder + "dataset_stats_" + type + ".csv" #-for init compilations
    expr_data = ExprData(columns)
    expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False, filter_conflict=False)
    init_wmc_per_expr, smallest_n = expr_data.get_metric_wrt_initial_per_expr(metric="WMC", obj="WMC")
    init_mc_per_expr, smallest_n = expr_data.get_metric_wrt_initial_per_expr(metric="MC", obj="MC")
    init_ratios_per_expr, smallest_n = expr_data.get_metric_wrt_initial_per_expr(metric="ratio", obj="WMC")
    f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/temp_ratio.csv", "w+")
    writer = csv.writer(f, delimiter=',')
    for e in init_ratios_per_expr:
        writer.writerow([e])
        writer.writerow([100*k for k in  init_ratios_per_expr[e]])
    f.flush()
    f.close()
    print("data set len, ", len(expr_data.all_expr_data))
    dont_consider = []
    conflict_exprs = []
    nb_vars = columns.index("nb_vars")
    for e in init_ratios_per_expr.keys():
        if expr_data.all_expr_data[e][0][nb_vars] < 50: # or len(init_ratios_per_expr[e]) < sample_size+1 :
            print(e, len(init_ratios_per_expr[e]), expr_data.all_expr_data[e][0][nb_vars])
            dont_consider.append(e)
        elif len(init_ratios_per_expr[e]) < sample_size + 1:
            conflict_exprs.append(e)
    # dont_consider = []
    print("=------------------dont_consider,", len(dont_consider))
    print(dont_consider)
    exit(9)
    print("conflict_exprs" ,len(conflict_exprs))
    for e in conflict_exprs:
        print(e, len(init_ratios_per_expr[e]))

    nb_compact_ars = [0 for i in  range(1, sample_size+1)]
    lbs = [100000 for i in  range(1, sample_size+1)]
    all_compact_ars = [[] for i in  range(1, sample_size+1)]
    all_ars = [[] for i in  range(1, sample_size+1)]
    # print("dont consider ", len(dont_consider) )
    count_exp = []
    for index in range(1, sample_size+1):
        for e in init_ratios_per_expr.keys():
            if e not in dont_consider:
                if e not in count_exp:
                    count_exp.append(e)
                if index < len(init_ratios_per_expr[e]):

                    current_ar = init_ratios_per_expr[e][index]
                    all_ars[index-1].append(current_ar)
                    # if current_ar >= 1 and init_wmc_per_expr[e][index] >= 0.5:
                    if current_ar >= 1.5 :
                        nb_compact_ars[index-1] += 1
                        all_compact_ars[index-1].append(current_ar)
                        if current_ar < lbs[index-1]:
                            lbs[index - 1] = current_ar
                else:
                    all_ars[index - 1].append(0)
    best_var_percentage = 0
    best_var_percentage_index = 0
    for i, ar in enumerate(nb_compact_ars):
        if ar > best_var_percentage:
            best_var_percentage = ar
            best_var_percentage_index = i
        elif ar == best_var_percentage:
            if lbs[i] > lbs[best_var_percentage_index]:
                best_var_percentage = ar
                best_var_percentage_index = i


    print("nb_compact_ars", len(nb_compact_ars), len(count_exp), len(dont_consider), len(init_ratios_per_expr))
    print(nb_compact_ars)
    print("best_var_percentage_index: ", best_var_percentage_index, best_var_percentage)
    print("lbs: ", lbs)
    print("all")
    # for ar in all_ars:
    #     print(ar)
    avgs = []
    medians = []
    maxes = []
    for adjusted_ratio_at_index in all_ars:
        avg = sum(adjusted_ratio_at_index) / len(adjusted_ratio_at_index)
        avgs.append(avg)
        median = statistics.median(adjusted_ratio_at_index)
        medians.append(median)
    print("max avg: ", max(avgs), np.argmax(avgs))
    print("max medians : ", max(medians), np.argmax(medians))

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [i for i in range(len(medians))]
    ax1.scatter(x, medians)
    ax1.plot(x, medians)
    plt.show()

    print(len(all_ars))
    print(len(dont_consider))
    # for e in count_exp:
    #     if 11 < len(init_ratios_per_expr[e]):
    #         print(e, round(100* init_wmc_per_expr[e][11], 26),round(100* init_mc_per_expr[e][11], 26) )

def write_inits():
    alg_types = [  "dynamic"]# , "static"]
    FOLDER = "Dataset_preproc"
    result_folder = "./results_aaai/"
    # expr_folders = [result_folder + FOLDER+"_wscore_estimate/" ]#,  result_folder + FOLDER+"_WMC/",result_folder + FOLDER + "_hybrid_wmc/"  ]
    expr_folders = [result_folder + FOLDER+"_WMC/" ]#,  result_folder + FOLDER+"_WMC/",result_folder + FOLDER + "_hybrid_wmc/"  ]
    columns = [ "p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4
    init_exprs = {}
    no_compiling_expr = []
    i = 0
    for folder in expr_folders:
        for type in alg_types:

            stats_file = folder + "dataset_stats_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False, filter_conflict=False)
            inits = expr_data.get_line(0)
            for e in inits.keys():
                if e not in init_exprs:
                    init_exprs[e] = inits[e]
                else:
                    print("duplicate/ ", e)
            # for e in expr_data.no_data_expr:
            #     if e not in no_compiling_expr:
            #         no_compiling_expr.append(e)
    # for e in init_exprs:
    #     print(e, [t[5] for t in init_exprs[e]])
        # print(e, variance([t[5] for t in init_exprs[e]]), variance([t[6] for t in init_exprs[e]]), len([t[5] for t in init_exprs[e]]))
    print(len(init_exprs), len(inits))
    print("no_data_expr: ",len(expr_data.no_data_expr))
    print(len(no_compiling_expr))
    print(no_compiling_expr)
    # f = open("./init_compilations.csv", "w+")
    # writer = csv.writer(f, delimiter=',')
    # for e in init_exprs:
    #     writer.writerow([e])
    #     writer.writerow(list(init_exprs[e]))
    # f.flush()
    # f.close()
    return init_exprs


def plot_percentage_experiments(percent=8):
    #look at exprs that have an init compilation, are present in the list of the dynamic_p_ stats file - and have a line for it -  and have a non zero value for compilation
    #read in init data

    f = open("./results_aaai2/" + "init_compilations.csv", "r")
    init_compilations = {}
    init_times = {}
    columns = [ "p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4

    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line2: break  # EOF
        data_row = []
        print(line1)
        print(line2)
        for x in line2.split(","):  # why do we ignore last column? it might be the empty value after the end line separator
            print(x)
            # x_val = float(x.strip('.%'))
            x_val = float(x)
            if math.isinf(x_val):
                x_val = np.float128(x)
            data_row.append(x_val)
        init_compilations[line1.strip()] = data_row
        # read compilation file
    c = 0
    for e in init_compilations:
        if e not in medium_instances:
            c+=1

    # f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/" + str(percent)+"percent_compilations.csv", "r")
    f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/22percent_compilations_medium_nofullsb.csv", "r")
    # f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/" + "8percent_compilations.csv", "r")
    percent_compilations = {}
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line2: break  # EOF
        data_row = []
        print(line1)
        print(line2)
        for x in line2.split(","):
            print(x, type(x))
            x_val = float(x.strip('"'))
            # x_val = float(x.strip())
            if math.isinf(x_val):
                x_val = np.float128(x.strip('"'))
            data_row.append(x_val)
        ename = line1.split(",")[0].split("_temphybrid")[0].split("/")[-1]+".cnf"
        if ename.count(".") > 1:
            ename = ename.strip("\n").replace(".", "_", ename.count(".") - 1)  # actually first . will always be ./input so should skipp tha
        percent_compilations[ename] = data_row
    #read csv from where we extract if expr shoul have a compilation
    percent_expr_data = ExprData(columns)
    stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "dataset_stats_p"+str(percent)+"_dynamic.csv"
    # stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "dataset_stats_p8_dynamic.csv"
    percent_expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=1, padding=False, filter_timeout=False,
                                      filter_conflict=False)
    lines = percent_expr_data.get_line(1)
    # print(len(lines))
    print(len(percent_expr_data.no_data_expr))
    print(len(percent_expr_data.all_expr_data))
    # print(len(expr_data.no_data_expr))
    # no_init_compilation = 0

    ars = {}
    y = []
    wmc_index = columns.index("WMC")
    size_index = columns.index("edge_count")
    time_index = columns.index("time")
    nb_vars_index = columns.index("nb_vars")
    nb_expr = 0
    sb_times = {}
    sb_compilation_times = {}
    init_ratios = {}
    count_compact = 0
    count_not_compact = 0
    no_init_compilation = 0
    compilation_zero = 0
    compilation_zero_expr = []
    plotted_expr_ratios = {}
    plotted_expr_nb_count = {}
    # for expr in lines.keys():
    for expr in percent_compilations:
        if expr in init_compilations:
            if expr in percent_compilations:
                if  expr in  medium_part2:
                    if percent_compilations[expr][wmc_index] > 0 and  percent_compilations[expr][size_index] > 0 :
                        print(expr, init_compilations[expr][wmc_index],  init_compilations[expr][size_index],  percent_compilations[expr][wmc_index] , percent_compilations[expr][size_index])
                        init_ratio = init_compilations[expr][wmc_index] /  init_compilations[expr][size_index]
                        current_ratio = percent_compilations[expr][wmc_index] /  percent_compilations[expr][size_index]
                        ar = current_ratio / init_ratio
                        if ar >= 1:
                            count_compact += 1
                        else:
                            print("not compact: ", ar)
                            count_not_compact += 1
                        init_ratios[expr] = init_ratio
                        # ars[expr] = ar
                        y.append(ar)
                        plotted_expr_ratios[expr] = ar
                        # plotted_expr_nb_count[expr] = percent_expr_data.all_expr_data[expr][0][nb_vars_index]

                        init_times[expr] = init_compilations[expr][time_index]
                        # sb_times[expr] = lines[expr][time_index]
                        # sb_compilation_times[expr] = percent_compilations[expr][time_index]
                        nb_expr += 1
            else:
                compilation_zero += 1
                compilation_zero_expr.append(expr)
        else:
            print(expr)
            no_init_compilation += 1

    # sorted_exprs = dict(sorted(plotted_expr_nb_count.items(), key=lambda kv: kv[1]))
    # instance_sizes = list(sorted_exprs.values())
    # y = []
    # for e in sorted_exprs.keys():
    #     y.append(plotted_expr_ratios[e])
    #
    print(statistics.median(y))
    print(len(y), len(medium_part2))

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [i for i in range(nb_expr)]
    ax1.scatter(x, y)
    ax1.plot(x, y)

    # ax1.scatter(instance_sizes, y)
    # ax1.plot(instance_sizes, y)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    # fig.tight_layout()
    # plt.yticks([i for i in range(300)])
    # plt.xticks(instance_sizes)
    # plt.ylim((float(6.931164793584654e-21), 280.37139126615557))
    plt.yscale("log")
    plt.grid()
    plt.show()
    # plt.savefig("./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "ratio_at_p"+str(percent)+"_log.png")
    exit(9)


    #plot time
    # init_y = []
    # sb_y = []
    # sb_comp_y = []
    # for e in sorted_exprs.keys():
    #     init_y.append(init_times[e])
    #     sb_y.append(sb_times[e])
    #     sb_comp_y.append(sb_compilation_times[e])
    # fig = plt.figure(figsize=(10, 7))
    # ax1 = fig.add_subplot(111)
    #----------

    # ax1.scatter(instance_sizes, init_y, color="r")
    # ax1.plot(instance_sizes, init_y, color = "r", label="init compilation time")
    #
    ax1.scatter(instance_sizes, sb_y, color="green")
    ax1.plot(instance_sizes, sb_y, color="green", label=str(percent)+" var percent SB")
    #
    # ax1.scatter(instance_sizes, sb_comp_y, color="cyan")
    # ax1.plot(instance_sizes, sb_comp_y, color="cyan", label=str(percent) + " var percent compilation")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    # fig.tight_layout()
    # plt.yticks([i for i in range(300)])
    # plt.xticks(instance_sizes)
    # plt.ylim((float(6.931164793584654e-21), 280.37139126615557))
    # plt.yscale("log")
    plt.grid()
    plt.show()
    # plt.savefig("./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "ratio_at_p"+str(percent)+"_log.png")

    # --------------------------s
    # print(y)
    print(init_times)
    print(sb_times)
    print(nb_expr)
    print("count_compact", count_compact)
    print("count_not_compact", count_not_compact)
    print("no_init_compilation", no_init_compilation)
    print(len(medium_instances))
    print(y)
    # count_medium = 0
    # count_large = 0
    # diff_count = 0
    # for expr in percent_expr_data.all_expr_data.keys():
    #     # print(expr, plotted_expr_nb_count[expr], plotted_expr_ratios[expr] )
    #     if expr in medium_instances:
    #         count_medium+=1
    #     elif expr in large_instances:
    #         count_large +=1
    #     else:
    #         diff_count += 1
    #         print(expr)
    # print("count_medium: ", count_medium, "count_large: ", count_large, len(medium_instances), len(large_instances), len(percent_expr_data.all_expr_data), diff_count)
    #
    # count = 0
    # no_sb = []
    # for expr in medium_instances:
    #     if expr not in plotted_expr_ratios:
    #         if expr in compilation_zero_expr:
    #             print("zero compilation ", expr)
    #         else:
    #             print(" ---------------------- not in plotted :", expr)
    #         count += 1
    #     if expr not in percent_compilations:
    #         print("no sb within an hour: ", expr)
    #         no_sb.append(expr)
    # print(count, len(medium_instances), len(plotted_expr_ratios), len(no_sb))
    # print(no_sb)

    # count = 0
    # for expr in large_instances:
    #     if expr not in plotted_expr_ratios:
    #         # print(" ---------------------- not in plotted :", expr)
    #         count += 1
    # print("large ", count)
    # print( len(medium_instances), len(sorted_exprs), len(percent_compilations))
    # print(min(y),max(y))
    # print("avg:", sum(y)/len(y) )
    # print("avg:", statistics.mean(y) )
    # print("compilation_zero:", compilation_zero)

def read_compilation_file(fname):
    f = open(fname, "r")
    # f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/" + "8percent_sscompilations.csv", "r")
    percent_compilations = {}
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line2: break  # EOF
        data_row = []
        # print(line1)
        # print(line2)
        for x in line2.split(","):
            # print(x, type(x))
            x_val = float(x.strip('"'))
            # x_val = float(x.strip())
            if math.isinf(x_val):
                x_val = np.float128(x.strip('"'))
            data_row.append(x_val)
        if "_temphybrid" in line1:
            ename = line1.split(",")[0].split("_temphybrid")[0].split("/")[-1] + ".cnf"
        else:
            ename = line1.split(",")[0].strip()
        if ename.count(".") > 1:
            ename = ename.strip("\n").replace(".", "_", ename.count(
                ".") - 1)  # actually first . will always be ./input so should skipp tha
        percent_compilations[ename] = data_row
    return  percent_compilations

def log_plot_percentage_experiment(percent=22):
    """
     the instances are sorted by adjusted ratio (so maybe a bar plot instead of having the number of variables as the X axis)
    - the Y axis starts at 10^-2
    """


    f = open("./results_aaai2/" + "init_compilations.csv", "r")
    init_compilations = {}
    init_times = {}
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
               "obj"]  # for d4

    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line2: break  # EOF
        data_row = []
        print(line1)
        print(line2)
        for x in line2.split(","):  # why do we ignore last column? it might be the empty value after the end line separator
            print(x)
            x_val = float(x)
            if math.isinf(x_val):
                x_val = np.float128(x)
            data_row.append(x_val)
        init_compilations[line1.strip()] = data_row
    # read compilation file
    c = 0


    # f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/" + str(percent)+"percent_compilations.csv", "r")
    f = "./results_aaai2/Dataset_preproc_hybrid_wmc/22percent_allmedium_compilations.csv"
    # f = "./results_aaai2/Dataset_preproc_hybrid_wmc/22percent_medium_compilations.csv"
    # f = "./results_aaai2/Dataset_preproc_hybrid_wmc/22percent_compilations_medium_nofullsb.csv"
    # f = open("./results_aaai2/Dataset_preproc_hybrid_wmc/" + "8percent_sscompilations.csv", "r")
    sb_compilations = read_compilation_file(f)
    print("sb_compilations ", len(sb_compilations))

    #
    # percent_expr_data = ExprData(columns)
    # # stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/" + "dataset_stats_p" + str(percent) + "_dynamic.csv"
    # stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/" + "dataset_stats_medium2_p_dynamic_p" + str(percent) + ".csv"
    # # stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "dataset_stats_p8_dynamic.csv"
    # percent_expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=1, padding=False,
    #                                   filter_timeout=False,
    #                                   filter_conflict=False)
    # lines = percent_expr_data.get_line(1)
    # print(len(lines))
    # print(len(percent_expr_data.no_data_expr))
    # print(len(percent_expr_data.all_expr_data))
    # print(len(expr_data.no_data_expr))
    # no_init_compilation = 0

    ars = {}
    y = []
    wmc_index = columns.index("WMC")
    size_index = columns.index("edge_count")
    time_index = columns.index("time")
    nb_vars_index = columns.index("nb_vars")
    nb_expr = 0
    sb_times = {}
    sb_compilation_times = {}
    init_ratios = {}
    count_compact = 0
    count_not_compact = 0
    no_init_compilation = 0
    compilation_zero = 0
    compilation_zero_expr = []
    plotted_expr_ratios = {}
    plotted_expr_nb_count = {}
    # for expr in lines.keys():
    for expr in medium_instances:
        if expr in init_compilations and expr in sb_compilations:
            if sb_compilations[expr][wmc_index] > 0 and sb_compilations[expr][size_index] > 0:
                init_ratio = init_compilations[expr][wmc_index] / init_compilations[expr][size_index]
                current_ratio = sb_compilations[expr][wmc_index] / sb_compilations[expr][size_index]
                ar = current_ratio / init_ratio
                if ar >= 1.5:
                    count_compact += 1
                else:
                    print("not compact: ", ar)
                    count_not_compact += 1
                init_ratios[expr] = init_ratio
                # ars[expr] = ar
                y.append(ar)
                plotted_expr_ratios[expr] = ar
                        # plotted_expr_nb_count[expr] = percent_expr_data.all_expr_data[expr][0][nb_vars_index]

                init_times[expr] = init_compilations[expr][time_index]
                # sb_times[expr] = lines[expr][time_index]
                # sb_compilation_times[expr] = percent_compilations[expr][time_index]
                nb_expr += 1
            else:
                compilation_zero += 1
                compilation_zero_expr.append(expr)
        else:
            print(expr, "no init and full sb")
            no_init_compilation += 1
    print("----------------no_init_compilation", no_init_compilation)
    sorted_exprs = dict(sorted(plotted_expr_ratios.items(), key=lambda kv: kv[1]))
        # instance_sizes = list(sorted_exprs.values())
    y = []
    for e in sorted_exprs.keys():
        y.append(plotted_expr_ratios[e])
        print(e, plotted_expr_ratios[e])


    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [i for i in range(nb_expr)]
    ax1.bar(x, y)
    # ax1.plot(x, y)

    # ax1.scatter(instance_sizes, y)
    # ax1.plot(instance_sizes, y)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    # fig.tight_layout()
    # plt.yticks([i for i in range(300)])
    # plt.xticks(instance_sizes)
    plt.ylim(0.01, max(y)+10 )
    plt.yscale("log")
    plt.grid()
    # plt.show()
    # plt.savefig("./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "ratio_at_p"+str(percent)+"_ordered_log.png")

    percent_expr_data = ExprData(columns)
    stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/" + "dataset_stats_medium2_p_dynamic_p22_details.csv"
    percent_expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=1, padding=False,
                                      filter_timeout=False,
                                      filter_conflict=False)
    selective_backbone_line = percent_expr_data.get_line(-1)
    time_index = columns.index("time")
    longest_time = 0
    for e in selective_backbone_line:
        if e in init_compilations and e in sb_compilations:
            print(e, selective_backbone_line[e][time_index])
            if float(selective_backbone_line[e][time_index]) > longest_time:
                    longest_time=float(selective_backbone_line[e][time_index])
    print("longest_time", longest_time)
    exit(9)

    percent_expr_data_m1 = ExprData(columns)
    stats_file_m1 = "./results_aaai2/Dataset_preproc_hybrid_wmc/" + "dataset_stats_p22_dynamic.csv"
    percent_expr_data_m1.read_stats_file(stats_file_m1, full_expr_only=False, min_nb_expr=1, padding=False,
                                         filter_timeout=False,
                                         filter_conflict=False)
    selective_backbone_line_m1 = percent_expr_data_m1.get_line(1)

    saved_expr = []
    filename = "./results_aaai2/Dataset_preproc_hybrid_wmc/ratio_at_p"+str(percent)+"_ordered_log.csv"
    f = open(filename, "w+")
    writer = csv.writer(f, delimiter=',')
    no_init = 0
    no_full_sb_count = 0
    expr_count = 0

    for e in medium_instances:
        if e.count(".") > 1:
            e = e.strip("\n").replace(".", "_", e.count(".") - 1)

        if (e in percent_expr_data_m1.remove_expr and e not in medium_part2) or e in percent_expr_data.remove_expr:
            writer.writerow([e])
            writer.writerow(columns)
            if e not in init_compilations:
                empty_row = [-1 for i in range(len(columns))]
                if e in percent_expr_data_m1.remove_expr:
                    empty_row[3] = percent_expr_data_m1.removed_expr_data[e][0][3]
                    empty_row[4] = percent_expr_data_m1.removed_expr_data[e][0][4]
                elif e in percent_expr_data.remove_expr:
                    empty_row[3] = percent_expr_data.removed_expr_data[e][0][3]
                    empty_row[4] = percent_expr_data.removed_expr_data[e][0][4]
                print("no init: ", e)
                no_init += 1
                writer.writerow(empty_row)
            else:
                writer.writerow(init_compilations[e])
                no_full_sb_count+=1
                print("no_full_sb_count: ",no_full_sb_count, e)
            writer.writerow([-1 for i in range(len(columns))])
        elif e in selective_backbone_line:
            writer.writerow([e])
            writer.writerow(columns)
            writer.writerow(init_compilations[e])
            if e not in sb_compilations:
                no_full_sb_count += 1
                print("no full SB found ", e, (selective_backbone_line[e][0]*100)/init_compilations[e][nb_vars_index], selective_backbone_line[e][0] , init_compilations[e][nb_vars_index])
                writer.writerow(selective_backbone_line[e])
            else:
                expr_count+=1
                comp_row = sb_compilations[e]
                comp_row[0] = selective_backbone_line[e][0]
                comp_row[1] = selective_backbone_line[e][1]
                comp_row[2] = selective_backbone_line[e][2]
                comp_row[3] = selective_backbone_line[e][3]
                comp_row[4] = selective_backbone_line[e][4]
                writer.writerow(comp_row)
        elif e in selective_backbone_line_m1:
            expr_count+=1
            writer.writerow([e])
            writer.writerow(columns)
            writer.writerow(init_compilations[e])
            comp_row = sb_compilations[e]
            comp_row[0] = selective_backbone_line_m1[e][0]
            comp_row[1] = selective_backbone_line_m1[e][1]
            comp_row[2] = selective_backbone_line_m1[e][2]
            comp_row[3] = selective_backbone_line_m1[e][3]
            comp_row[4] = selective_backbone_line_m1[e][4]
            writer.writerow(comp_row)
        else:
            print("something wrong: ", e)

    f.flush()
    f.close()

    print("no_full_sb_count: ", no_full_sb_count)
    print("no_init: ", no_init)
    print("expr_count: ", expr_count)


def evaluate_prediction():
    fname = "./results_aaai2/Dataset_preproc_hybrid_wmc/ratio_at_p22_allmedium.csv"
    f = open(fname, "r")
    percent_compilations = {}
    all_sb_compilation = {}
    all_init_compilation = {}
    expr_full_sb = []
    expr_no_init = []
    expr_partial_sb = []
    nb_vars_index = columns.index("nb_vars")
    wmc_index = columns.index("WMC")
    size_index = columns.index("edge_count")
    while True:
        expr_name = f.readline().strip()
        cols = f.readline()
        temp_init_compilation = f.readline()
        temp_sb_compilation = f.readline()
        if not temp_sb_compilation: break  # EOF
        sb_compilation = []
        init_compilation = []
        for x in temp_sb_compilation.split(","):
            x_val = float(x.strip('"'))
            if math.isinf(x_val):
                x_val = np.float128(x.strip('"'))
            sb_compilation.append(x_val)
        all_sb_compilation[expr_name] = sb_compilation.copy()
        for x in temp_init_compilation.split(","):
            x_val = float(x.strip('"'))
            if math.isinf(x_val):
                x_val = np.float128(x.strip('"'))
            init_compilation.append(x_val)
        all_init_compilation[expr_name] = init_compilation.copy()

        if init_compilation[wmc_index] == -1:
            expr_no_init.append(expr_name)
        if sb_compilation[wmc_index] > -1 and round( (sb_compilation[0]*100)/sb_compilation[nb_vars_index]) == 22:
            expr_full_sb.append(expr_name)
        else:
            expr_partial_sb.append(expr_name)
            if sb_compilation[0] == -1:
                print("no compilation at all: ", expr_name, (sb_compilation[0] * 100) / sb_compilation[nb_vars_index])
                # if expr_name in medium_part2:
                #     print("second part")
            else:
                print(expr_name, (sb_compilation[0] * 100) / sb_compilation[nb_vars_index])


    ratios={}
    conflict_expr_fullSB = 0
    medium3 = []
    for e in expr_partial_sb+expr_partial_sb+expr_no_init:
        if e in medium_instances:
            print(e)
            if e not in medium3:
                medium3.append(e)
        else:
            print("---------------------",e)
    for e in expr_full_sb:
        init_ratio = all_init_compilation[e][wmc_index] / all_init_compilation[e][size_index]
        if all_sb_compilation[e][size_index] == 0:
            print(e, "is 0 size , expr finished but reached conflic SB", all_init_compilation[e][nb_vars_index])
            # if e in medium_part2:
            #     print("part 2")
            current_ratio = 0
            conflict_expr_fullSB += 1
            if e in medium_instances:
                if e not in medium3:
                    medium3.append(e)
        else:
            current_ratio = all_sb_compilation[e][wmc_index] / all_sb_compilation[e][size_index]

        ar = current_ratio / init_ratio
        ratios[e]=ar
    print("m3", len(medium3), sorted(medium3))

    # print(len(expr_no_init), len(expr_partial_sb), len(expr_full_sb), conflict_expr_fullSB)
    exit(6)
    sorted_exprs = dict(sorted(ratios.items(), key=lambda kv: kv[1]))
    nb_expr= len(sorted_exprs)
    y = [sorted_exprs[k] for k in sorted_exprs]
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [i for i in range(nb_expr)]
    # ax1.bar(x, y)
    # ax1.plot(x, y)

    ax1.plot(x, list(ratios.value()))
    # ax1.scatter(instance_sizes, y)
    # ax1.plot(instance_sizes, y)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    # fig.tight_layout()
    # plt.yticks([i for i in range(300)])
    # plt.xticks(instance_sizes)
    plt.ylim(0.01, max(y) + 10)
    plt.yscale("log")
    plt.grid()
    plt.show()
    # plt.savefig("./results_aaai2/Dataset_preproc_hybrid_wmc/"  + "ratio_at_p"+str(percent)+"_ordered_log.png")



def count_hybrid_call():
    filename = "./results_aaai2/Dataset_preproc_hybrid_wmc/dataset_stats_p_dynamic.txt"
    total_d4_callcount = {}
    iteration_d4_callcount = {}

    d4_calls = 0
    prev_expr = ""
    with open(filename, "r") as f:
        content = f.readlines()
        for line in content:
            if "input/" in line:
                if line.count(".") > 1:
                    save_expr_name = line.strip("\n").replace(".", "_", line.count(".") - 1)  # actually first . will always be ./input so should skipp tha
                    save_expr_name = save_expr_name.split("/")[-1]
                if prev_expr != "":
                    total_d4_callcount[prev_expr] = d4_calls
                iteration_d4_callcount[save_expr_name] = []
                total_d4_callcount[save_expr_name] = 0
                prev_expr = save_expr_name
                d4_calls = 0
            elif "A:" in line:
                nb_d4 = int(line.split(",")[-1].strip("]\n"))
                iteration_d4_callcount[prev_expr].append(nb_d4)
                d4_calls +=nb_d4

    columns = [ "p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4

    expr_data = ExprData(columns)
    stats_file = "./results_aaai2/Dataset_preproc_hybrid_wmc/" + "dataset_stats_p8_dynamic.csv"
    expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=1, padding=False, filter_timeout=False,
                              filter_conflict=False)
    sb_lines = expr_data.get_line(1)
    print(len(sb_lines))

    for e in total_d4_callcount:
        if e in sb_lines:
            nb_vars = sb_lines[e][3]
            iteration_len = len(iteration_d4_callcount[e])
            percentages = []
            for i in range(iteration_len):
                vars = nb_vars-i
                assignments = 2*vars
                p =  ( iteration_d4_callcount[e][i] * 100 ) / assignments
                formatted = "{:.3f}".format(p)
                percentages.append(formatted)
            print(e, "-" ,sb_lines[e][3], "-" , sb_lines[e][4], "-", total_d4_callcount[e], "-", iteration_len , "-",  max(percentages), "-", percentages,  "-", iteration_d4_callcount[e])
    print(len(total_d4_callcount))


medium_instances = ['03_iscas85_c1355_isc.cnf', '03_iscas85_c1908_isc.cnf', '03_iscas85_c880.isc.cnf', '04_iscas89_s1196.bench.cnf', '04_iscas89_s1238.bench.cnf',
                    '04_iscas89_s1423.bench.cnf', '04_iscas89_s1488.bench.cnf', '04_iscas89_s1494.bench.cnf', '04_iscas89_s641.bench.cnf', '04_iscas89_s713.bench.cnf',
                    '04_iscas89_s820.bench.cnf', '04_iscas89_s832.bench.cnf', '04_iscas89_s838.1.bench.cnf', '04_iscas89_s953.bench.cnf', '05_iscas93_s1196.bench.cnf',
                    '05_iscas93_s1269_bench.cnf', '05_iscas93_s1512.bench.cnf', '05_iscas93_s635.bench.cnf', '05_iscas93_s938.bench.cnf', '05_iscas93_s967.bench.cnf',
                    '05_iscas93_s991.bench.cnf', '06_iscas99_b04.cnf', '06_iscas99_b07.cnf', '06_iscas99_b11.cnf', '06_iscas99_b13.cnf', '07_blocks_right_2_p_t10.cnf',
                    '07_blocks_right_2_p_t4.cnf', '07_blocks_right_2_p_t5.cnf', '07_blocks_right_2_p_t6.cnf', '07_blocks_right_2_p_t7.cnf', '07_blocks_right_2_p_t8.cnf',
                    '07_blocks_right_2_p_t9.cnf', '07_blocks_right_3_p_t2.cnf', '07_blocks_right_3_p_t3.cnf', '07_blocks_right_3_p_t4.cnf', '07_blocks_right_3_p_t5.cnf',
                    '07_blocks_right_4_p_t2.cnf', '07_blocks_right_4_p_t3.cnf', '07_blocks_right_5_p_t1.cnf', '07_blocks_right_5_p_t2.cnf', '07_blocks_right_6_p_t1.cnf',
                    '08_bomb_b10_t5_p_t1.cnf', '08_bomb_b5_t1_p_t3.cnf', '08_bomb_b5_t1_p_t4.cnf', '08_bomb_b5_t1_p_t5.cnf', '08_bomb_b5_t1_p_t6.cnf', '08_bomb_b5_t1_p_t7.cnf',
                    '08_bomb_b5_t1_p_t8.cnf', '08_bomb_b5_t5_p_t2.cnf', '08_bomb_b5_t5_p_t3.cnf', '09_coins_p01_p_t2.cnf', '09_coins_p01_p_t3.cnf', '09_coins_p01_p_t4.cnf',
                    '09_coins_p01_p_t5.cnf', '09_coins_p02_p_t2.cnf', '09_coins_p02_p_t3.cnf', '09_coins_p02_p_t4.cnf', '09_coins_p02_p_t5.cnf', '09_coins_p03_p_t2.cnf',
                    '09_coins_p03_p_t3.cnf', '09_coins_p03_p_t4.cnf', '09_coins_p03_p_t5.cnf', '09_coins_p04_p_t2.cnf', '09_coins_p04_p_t3.cnf', '09_coins_p04_p_t4.cnf',
                    '09_coins_p04_p_t5.cnf', '09_coins_p05_p_t2.cnf', '09_coins_p05_p_t3.cnf', '09_coins_p05_p_t4.cnf', '09_coins_p05_p_t5.cnf', '09_coins_p10_p_t1.cnf',
                    '09_coins_p10_p_t2.cnf', '10_comm_p01_p_t3.cnf', '10_comm_p01_p_t4.cnf', '10_comm_p01_p_t5.cnf', '10_comm_p01_p_t6.cnf', '10_comm_p02_p_t2.cnf',
                    '10_comm_p02_p_t3.cnf', '10_comm_p03_p_t1.cnf', '10_comm_p03_p_t2.cnf', '10_comm_p04_p_t1.cnf', '10_comm_p05_p_t1.cnf', '11_emptyroom_d12_g6_p_t3.cnf',
                    '11_emptyroom_d12_g6_p_t4.cnf', '11_emptyroom_d12_g6_p_t5.cnf', '11_emptyroom_d12_g6_p_t6.cnf', '11_emptyroom_d12_g6_p_t7.cnf', '11_emptyroom_d16_g8_p_t2.cnf',
                    '11_emptyroom_d16_g8_p_t3.cnf', '11_emptyroom_d16_g8_p_t4.cnf', '11_emptyroom_d16_g8_p_t5.cnf', '11_emptyroom_d20_g10_corners_p_t2.cnf', '11_emptyroom_d20_g10_corners_p_t3.cnf',
                    '11_emptyroom_d20_g10_corners_p_t4.cnf', '11_emptyroom_d24_g12_p_t2.cnf', '11_emptyroom_d24_g12_p_t3.cnf', '11_emptyroom_d28_g14_corners_p_t1.cnf',
                    '11_emptyroom_d28_g14_corners_p_t2.cnf', '11_emptyroom_d28_g14_corners_p_t3.cnf', '11_emptyroom_d4_g2_p_t10.cnf', '11_emptyroom_d4_g2_p_t9.cnf', '11_emptyroom_d8_g4_p_t10.cnf',
                    '11_emptyroom_d8_g4_p_t4.cnf', '11_emptyroom_d8_g4_p_t5.cnf', '11_emptyroom_d8_g4_p_t6.cnf', '11_emptyroom_d8_g4_p_t7.cnf', '11_emptyroom_d8_g4_p_t8.cnf',
                    '11_emptyroom_d8_g4_p_t9.cnf', '13_ring2_r6_p_t10.cnf', '13_ring2_r6_p_t5.cnf', '13_ring2_r6_p_t6.cnf', '13_ring2_r6_p_t7.cnf', '13_ring2_r6_p_t8.cnf', '13_ring2_r6_p_t9.cnf',
                    '13_ring2_r8_p_t10.cnf', '13_ring2_r8_p_t4.cnf', '13_ring2_r8_p_t5.cnf', '13_ring2_r8_p_t6.cnf', '13_ring2_r8_p_t7.cnf', '13_ring2_r8_p_t8.cnf', '13_ring2_r8_p_t9.cnf',
                    '13_ring_3_p_t10.cnf', '13_ring_3_p_t7.cnf', '13_ring_3_p_t8.cnf', '13_ring_3_p_t9.cnf', '13_ring_4_p_t10.cnf', '13_ring_4_p_t5.cnf', '13_ring_4_p_t6.cnf', '13_ring_4_p_t7.cnf',
                    '13_ring_4_p_t8.cnf', '13_ring_4_p_t9.cnf', '13_ring_5_p_t10.cnf', '13_ring_5_p_t4.cnf', '13_ring_5_p_t5.cnf', '13_ring_5_p_t6.cnf', '13_ring_5_p_t7.cnf', '13_ring_5_p_t8.cnf',
                    '13_ring_5_p_t9.cnf', '14_safe_safe_10_p_t10.cnf', '14_safe_safe_30_p_t3.cnf', '14_safe_safe_30_p_t4.cnf', '14_safe_safe_30_p_t5.cnf', '14_safe_safe_30_p_t6.cnf',
                    '14_safe_safe_30_p_t7.cnf', '14_safe_safe_30_p_t8.cnf', '14_safe_safe_30_p_t9.cnf', '15_sort_num_s_3_p_t10.cnf', '15_sort_num_s_4_p_t4.cnf', '15_sort_num_s_4_p_t5.cnf',
                    '15_sort_num_s_4_p_t6.cnf', '15_sort_num_s_4_p_t7.cnf', '15_sort_num_s_4_p_t8.cnf', '15_sort_num_s_4_p_t9.cnf', '15_sort_num_s_5_p_t2.cnf', '15_sort_num_s_6_p_t1.cnf',
                    '15_sort_num_s_7_p_t1.cnf', '16_uts_k2_p_t4.cnf', '16_uts_k2_p_t5.cnf', '16_uts_k2_p_t6.cnf', '16_uts_k2_p_t7.cnf', '16_uts_k2_p_t8.cnf', '16_uts_k3_p_t2.cnf',
                    '16_uts_k4_p_t1.cnf', '16_uts_k5_p_t1.cnf']


large_instances = ['05_iscas93_s3271_bench.cnf', '05_iscas93_s3330_bench.cnf', '05_iscas93_s3384_bench.cnf',
                       '05_iscas93_s4863_bench.cnf', '06_iscas99_b05.cnf',
                       '06_iscas99_b12.cnf', '07_blocks_right_3_p_t10.cnf', '07_blocks_right_3_p_t6.cnf',
                       '07_blocks_right_3_p_t7.cnf', '07_blocks_right_3_p_t8.cnf',
                       '07_blocks_right_3_p_t9.cnf', '07_blocks_right_4_p_t4.cnf', '07_blocks_right_4_p_t5.cnf',
                       '07_blocks_right_5_p_t3.cnf', '07_blocks_right_6_p_t2.cnf',
                       '08_bomb_b10_t10_p_t10.cnf', '08_bomb_b10_t10_p_t11.cnf', '08_bomb_b10_t10_p_t12.cnf',
                       '08_bomb_b10_t10_p_t13.cnf', '08_bomb_b10_t10_p_t14.cnf',
                       '08_bomb_b10_t10_p_t15.cnf', '08_bomb_b10_t10_p_t16.cnf', '08_bomb_b10_t10_p_t17.cnf',
                       '08_bomb_b10_t10_p_t18.cnf', '08_bomb_b10_t10_p_t1.cnf',
                       '08_bomb_b10_t10_p_t2.cnf', '08_bomb_b10_t10_p_t3.cnf', '08_bomb_b10_t10_p_t4.cnf',
                       '08_bomb_b10_t10_p_t5.cnf', '08_bomb_b10_t10_p_t6.cnf', '08_bomb_b10_t10_p_t7.cnf',
                       '08_bomb_b10_t10_p_t8.cnf', '08_bomb_b10_t10_p_t9.cnf', '08_bomb_b10_t5_p_t10.cnf',
                       '08_bomb_b10_t5_p_t2.cnf', '08_bomb_b10_t5_p_t3.cnf', '08_bomb_b10_t5_p_t4.cnf',
                       '08_bomb_b10_t5_p_t5.cnf', '08_bomb_b10_t5_p_t6.cnf', '08_bomb_b10_t5_p_t7.cnf',
                       '08_bomb_b10_t5_p_t8.cnf', '08_bomb_b10_t5_p_t9.cnf', '08_bomb_b20_t5_p_t10.cnf',
                       '08_bomb_b20_t5_p_t1.cnf', '08_bomb_b20_t5_p_t2.cnf', '08_bomb_b20_t5_p_t3.cnf',
                       '08_bomb_b20_t5_p_t4.cnf', '08_bomb_b20_t5_p_t5.cnf', '08_bomb_b20_t5_p_t6.cnf',
                       '08_bomb_b20_t5_p_t7.cnf', '08_bomb_b20_t5_p_t8.cnf', '08_bomb_b20_t5_p_t9.cnf',
                       '08_bomb_b5_t1_p_t10.cnf', '08_bomb_b5_t1_p_t9.cnf', '08_bomb_b5_t5_p_t10.cnf',
                       '08_bomb_b5_t5_p_t4.cnf', '08_bomb_b5_t5_p_t5.cnf', '08_bomb_b5_t5_p_t6.cnf',
                       '08_bomb_b5_t5_p_t7.cnf', '08_bomb_b5_t5_p_t8.cnf', '08_bomb_b5_t5_p_t9.cnf',
                       '09_coins_p01_p_t6.cnf', '09_coins_p01_p_t7.cnf', '09_coins_p01_p_t8.cnf',
                       '09_coins_p01_p_t9.cnf', '09_coins_p02_p_t6.cnf', '09_coins_p02_p_t7.cnf',
                       '09_coins_p02_p_t8.cnf', '09_coins_p02_p_t9.cnf', '09_coins_p03_p_t6.cnf',
                       '09_coins_p03_p_t7.cnf', '09_coins_p03_p_t8.cnf', '09_coins_p03_p_t9.cnf',
                       '09_coins_p04_p_t6.cnf', '09_coins_p04_p_t7.cnf', '09_coins_p04_p_t8.cnf',
                       '09_coins_p04_p_t9.cnf', '09_coins_p05_p_t6.cnf', '09_coins_p05_p_t7.cnf',
                       '09_coins_p05_p_t8.cnf', '09_coins_p05_p_t9.cnf', '09_coins_p10_p_t3.cnf',
                       '09_coins_p10_p_t4.cnf', '10_comm_p01_p_t10.cnf', '10_comm_p01_p_t7.cnf',
                       '10_comm_p01_p_t8.cnf', '10_comm_p01_p_t9.cnf', '10_comm_p02_p_t10.cnf', '10_comm_p02_p_t4.cnf',
                       '10_comm_p02_p_t5.cnf', '10_comm_p02_p_t6.cnf',
                       '10_comm_p02_p_t7.cnf', '10_comm_p02_p_t8.cnf', '10_comm_p02_p_t9.cnf', '10_comm_p03_p_t10.cnf',
                       '10_comm_p03_p_t3.cnf', '10_comm_p03_p_t4.cnf',
                       '10_comm_p03_p_t5.cnf', '10_comm_p03_p_t6.cnf', '10_comm_p03_p_t7.cnf', '10_comm_p03_p_t8.cnf',
                       '10_comm_p03_p_t9.cnf', '10_comm_p04_p_t10.cnf',
                       '10_comm_p04_p_t2.cnf', '10_comm_p04_p_t3.cnf', '10_comm_p04_p_t4.cnf', '10_comm_p04_p_t5.cnf',
                       '10_comm_p04_p_t6.cnf', '10_comm_p04_p_t7.cnf',
                       '10_comm_p04_p_t8.cnf', '10_comm_p04_p_t9.cnf', '10_comm_p05_p_t2.cnf', '10_comm_p05_p_t3.cnf',
                       '10_comm_p05_p_t4.cnf', '10_comm_p05_p_t5.cnf',
                       '10_comm_p05_p_t6.cnf', '10_comm_p05_p_t7.cnf', '10_comm_p05_p_t8.cnf', '10_comm_p05_p_t9.cnf',
                       '10_comm_p10_p_t1.cnf', '10_comm_p10_p_t2.cnf',
                       '11_emptyroom_d12_g6_p_t10.cnf', '11_emptyroom_d12_g6_p_t8.cnf', '11_emptyroom_d12_g6_p_t9.cnf',
                       '11_emptyroom_d16_g8_p_t10.cnf', '11_emptyroom_d16_g8_p_t6.cnf',
                       '11_emptyroom_d16_g8_p_t7.cnf', '11_emptyroom_d16_g8_p_t8.cnf', '11_emptyroom_d16_g8_p_t9.cnf',
                       '11_emptyroom_d20_g10_corners_p_t5.cnf',
                       '11_emptyroom_d20_g10_corners_p_t6.cnf', '11_emptyroom_d20_g10_corners_p_t7.cnf',
                       '11_emptyroom_d20_g10_corners_p_t8.cnf', '11_emptyroom_d20_g10_corners_p_t9.cnf',
                       '11_emptyroom_d24_g12_p_t10.cnf', '11_emptyroom_d24_g12_p_t4.cnf',
                       '11_emptyroom_d24_g12_p_t5.cnf', '11_emptyroom_d24_g12_p_t6.cnf',
                       '11_emptyroom_d24_g12_p_t7.cnf', '11_emptyroom_d24_g12_p_t8.cnf',
                       '11_emptyroom_d24_g12_p_t9.cnf', '11_emptyroom_d28_g14_corners_p_t4.cnf',
                       '11_emptyroom_d28_g14_corners_p_t5.cnf', '11_emptyroom_d28_g14_corners_p_t6.cnf',
                       '11_emptyroom_d28_g14_corners_p_t7.cnf', '11_emptyroom_d28_g14_corners_p_t8.cnf',
                       '11_emptyroom_d28_g14_corners_p_t9.cnf', '14_safe_safe_30_p_t10.cnf',
                       '15_sort_num_s_4_p_t10.cnf', '16_uts_k2_p_t10.cnf']
columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
           "obj"]  # for d4

def get_medium_instances():
    medium = []
    directory = "./input/Dataset_preproc/"
    init_exprs =write_inits()
    for filename in os.listdir(directory):
        if filename.endswith(".cnf"):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                # print(f)
                with open(f, "r") as f:
                    content = f.readlines()
                    nb_vars = int(content[0].strip().split(" ")[2])
                    if nb_vars > 300 and nb_vars < 900:
                        temp = filename.split("/")[-1]
                        if temp.count(".") > 1:
                            temp = temp.replace(".", "_", temp.count(".") - 1)
                        if temp in init_exprs:
                            medium.append(filename)
                        else:
                            print("no init: ", filename, nb_vars)
    print(len(medium))
    medium = sorted(medium)
    k=4
    n=len(medium)
    splits = [medium[i * (n // k) + min(i, n % k):(i+1) * (n // k) + min(i+1, n % k)] for i in range(k)]
    for k in splits:
        print(k)
    #1- 03 -07
    #2 - 08-10
    #3 11-13
    #4 14-16

if __name__ == "__main__":
    # get_medium_instances()
    evaluate_prediction()
    # read_medium2()
    # filer_instances()
    # get_best_variable_percentage(50)
    # write_inits()
    # plot_percentage_experiments(22)
    # log_plot_percentage_experiment(22)
    # plot_percentage_experiments(8)
    # count_hybrid_call()
    exit(8)


    # alg_types = [ "static", "dynamic",  "random_selection_1234" ]
    # alg_types = [ "rand_dynamic" ]# ,  "random_selection_1234" ]
    alg_types = [ "static", "dynamic" ]
    # alg_types = [  "dynamic" ]
    # alg_types = [  "dynamic" , "static"]
    # FOLDER = "Dataset_preproc"
    # result_folder = "./results_aaai2/"
    result_folder = "./results/"
    FOLDER = "Dataset_preproc_final"
    # FOLDER = "Dataset_preproc_NO_COMPILE_2"
    HEUR_NAMES = {"MC/": "actual_MC", "WMC/": "actual_WMC", "half/": "relative_weight", "estimate/": "estimated_WMC", "random":"random", "hybrid_wmc/": "hybrid"}
    # FOLDER = "Dataset_preproc_part2"
    # expr_folders =  [  "./results/"+FOLDER+"_rand_dynamic/"]
    # expr_folders =  [ "./results/"+FOLDER+"_WMC/",  "./results/"+FOLDER+"_wscore_half/", "./results/"+FOLDER+"_wscore_estimate/",  "./results/"+FOLDER+"_rand_dynamic/"]
    # expr_folders =  [ "./results/"+FOLDER+"_MC/" ]#,  "./results/"+FOLDER+"_wscore_half/", "./results/"+FOLDER+"_wscore_estimate/",  "./results/"+FOLDER+"_rand_dynamic/"]
    # expr_folders =  [ result_folder+FOLDER+"_WMC/", result_folder+FOLDER+"_wscore_estimate/" ]#,  "./results/"+FOLDER+"_wscore_half/", "./results/"+FOLDER+"_wscore_estimate/",  "./results/"+FOLDER+"_rand_dynamic/"]
    # expr_folders =  [  result_folder+FOLDER+"_hybrid_wmc/" ]#,  "./results/"+FOLDER+"_wscore_half/", "./results/"+FOLDER+"_wscore_estimate/",  "./results/"+FOLDER+"_rand_dynamic/"]

    expr_folders =  [ result_folder+FOLDER+"_WMC/", result_folder+FOLDER+"_wscore_estimate/" , result_folder+FOLDER+"_wscore_half/" , result_folder+FOLDER+"_hybrid_wmc/" ,  result_folder+FOLDER+"_rand_dynamic/" ]
    # expr_folders =  [ result_folder+FOLDER+"_hybrid_wmc/" , result_folder+FOLDER+"_rand_dynamic/" ]

    # expr_folders =  [ "./results/"+FOLDER+"_wscore_half/", "./results/"+FOLDER+"_wscore_estimate/",  "./results/"+FOLDER+"_rand_dynamic/"]
    # expr_folders = [  "./results/Benchmark_preproc2_WMC/" ,  "./results/Benchmark_preproc2_wscore_half/", "./results/Benchmark_preproc2_wscore_estimate/", "./results/Benchmark_preproc2_rand_dynamic/"]
    # expr_folders = [ "./results/Benchmark_preproc2_wscore_half/" ,"./results/Benchmark_preproc2_wscore_estimate/" ,"./results/Benchmark_preproc_wscore_adjoccratio/"   ]#, "./results/Benchmark_preproc_wscore_estimate/"]# "./results/sdd/wmc2022_track2_private_WMC/"
    # expr_folders = ["./results/Benchmark_preproc_WMC/"  , "./results/Benchmark_preproc_wscore_estimate/", "./results/Benchmark_preproc_wscore_half/" ,"./results/Benchmark_preproc_wscore_occratio/" ,"./results/Benchmark_preproc_wscore_adjoccratio/"   ]#, "./results/Benchmark_preproc_wscore_estimate/"]# "./results/sdd/wmc2022_track2_private_WMC/"
    columns = [ "p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4
    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time', 'WMC',
    #            "logWMC","obj"]  # for weighted sdd


    subfolder = "planning"
    # obj = "MC"
    obj = "WMC"
    out_file = result_folder+FOLDER+"_avg_weighted_"#+subfolder+"_" #this is actually ecai23 data
    # out_file = result_folder+FOLDER+"_median_weighted_"#+subfolder+"_" #this is actually ecai23 data
    # if obj == "MC":
    #     out_file = result_folder+"Dataset_preproc_avg_MC_"

    same_expr = True
    filter_timeout = False
    filter_conflict = False
    median = True
    if median:
         out_file = result_folder+"Dataset_preproc_median_MC_"

    # out_file = "./results2/Dataset_preproc_avg_MC"
    # out_file = "./results/Dataset_preproc_avg_MC_and_WMC"
    # average_efficiency_WMC_MC(expr_folders, out_file +"_efficiency", "", alg_types, 50, columns, obj, padding=True, same_expr=same_expr,
    # average_efficiency_WMC_MC(expr_folders, out_file +"_VIRTUAL_BEST_efficiency", "", alg_types, 50, columns, obj, padding=True, same_expr=same_expr,
    #                    filter_timeout=filter_timeout, filter_conflict=filter_conflict, subfolder=subfolder)
    # average_ratio_MC_WMC(expr_folders, out_file +"ratio", "", alg_types, 50, columns, obj, padding=True, same_expr=same_expr,
    #               filter_timeout=filter_timeout, filter_conflict=filter_conflict, subfolder=subfolder)
    # exit(8)

    # out_file = "./results/Benchmark_preproc2_avg_weighted_"
    if not same_expr:
        out_file = out_file+"diff_exprs_"
    if filter_timeout:
        out_file = out_file+"filterT_"
    if filter_conflict:
        out_file = out_file+"filterC_"
    title = "Average weighted efficiency over dataset "
    if median:
        title = "Median weighted efficiency over dataset "
    # if obj == "MC":
    #     title = "Average MC efficiency over instances "
    average_efficiency(expr_folders, out_file +"efficiency", title, alg_types, 50, columns, obj, padding=True, same_expr=same_expr,
                       filter_timeout=filter_timeout, filter_conflict=filter_conflict, subfolder=subfolder, median=median)
    title = "Average weighted ratio over instances"
    if median:
        title = "Median weighted ratio over instances"
    # if obj == "MC":
    #     title = "Average MC efficiency over instances"
    average_ratio(expr_folders, out_file +"ratio", title, alg_types, 50, columns, obj, padding=True, same_expr=same_expr,
                  filter_timeout=filter_timeout, filter_conflict=filter_conflict, subfolder=subfolder, median=median)
    # col = "WMC"
    # title = "Average weighted " + col
    # average_column(expr_folders, out_file + col, title, alg_types, 100, columns, "WMC", padding=True, plot_tye=col,
    #                same_expr=same_expr, filter_timeout=filter_timeout, filter_conflict=filter_conflict, subfolder=subfolder)
    # col = "edge_count"
    # title = "Average weighted " + col
    # average_column(expr_folders, out_file + col, title, alg_types, 100, columns, "WMC", padding=True, plot_tye=col,
    #                same_expr=same_expr, filter_timeout=filter_timeout, filter_conflict=filter_conflict , subfolder=subfolder)


