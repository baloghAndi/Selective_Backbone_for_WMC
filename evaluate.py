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

import CNFmodelBDD as _cnfBDD


class Logger:

    def __init__(self, filename, column_names, expr_data, out_folder, compile=False):
        print(os.getcwd())
        self.f = open(filename,"a+")
        self.writer = csv.writer(self.f, delimiter=',')
        self.progress_log =  open(filename.replace(".csv", ".txt"),"a+")
        self.error_log =  open(filename.replace(".csv", "_error.txt"),"a+")
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

    def log_error(self, cnf, message):
        self.error_log.write(cnf+"\n")
        for m in message:
            self.error_log.write(m)
        self.error_log.flush()
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

    def convert_to_columns(self, new_columns, new_exp_names): #TODO: PRINT THIS TO A FILE SO i DONT HAVE TO CHANGE IT EVERYWHERE
        # exp_names = []
        # for e in self.exprs:
        #     expr_name = e.split("/")[-1]
        #     found = False
        #     for x in new_exp_names:
        #         if expr_name in x:
        #             found = True
        #             break
        #     if found:
        #         exp_names.append(e)
        new_column_index = []
        for c in new_columns:
            if c in self.column_names:
                i =  self.column_names.index(c)
                new_column_index.append(i)
            else:
                new_column_index.append(-1)
        self.column_names = new_columns
        new_all_expr_data = {}
        for e_index, expr in enumerate(self.exprs):
            print(self.exprs)
            # print(new_exp_names)
            print(e_index)
            # if new_exp_names[e_index].split("/")[-1] != expr.split("/")[-1]:
            #     print(new_exp_names[e_index].split("/")[-1], expr.split("/")[-1])
            #     print("error with expr names")
            #     exit(9)
            print(expr)
            if "Planning" in expr:
                new_expr_name = expr.replafce("./Planning/pddlXpSym/", "./aaai_data/input/Planning/")
            else:
                new_expr_name = expr.replace("./", "./aaai_data/input/")
            new_all_expr_data[new_expr_name] = []
            print(expr, new_expr_name)
            new_expr_data = []
            for row in self.all_expr_data[expr]:
                new_row = []
                for i in new_column_index:
                    if i == -1:
                        new_row.append(-1)
                    else:
                        new_row.append(row[i])
                new_expr_data.append(new_row.copy())
            new_all_expr_data[new_expr_name].extend(new_expr_data.copy())
        self.all_expr_data = new_all_expr_data
        self.exprs = list(new_all_expr_data.keys())
        if "random" in self.filename:
            filename= self.filename.replace("_1234_reorder.csv", ".csv")
        else:
            filename = self.filename.replace("_reorder.csv",".csv")
        f = open(filename, "w")
        print("==================================== create ", filename)
        writer = csv.writer(f, delimiter=',')
        for e in self.exprs:
            writer.writerow([e])
            writer.writerow(new_columns)
            for exp_data in self.all_expr_data[e]:
                writer.writerow(exp_data)
        f.flush()
        f.close()




    def plot_all_efficiencies(self, column_name, name_extension):
        print("CALCULATE")
        for expr in self.exprs:
            title = expr.split("/")[-1]
            out_file = expr.replace(".cnf", name_extension+".png")
            self.plot_efficiency(self.all_expr_data[expr], title, out_file, column_name)
            # self.plot_efficiency_MC_BDD(self.all_expr_data[expr], title, out_file)

    def plot_all_efficiencies_percentage(self):
        print("CALCULATE")
        for expr in self.exprs:
            title = expr.split("/")[-1]
            out_file = expr.replace(".cnf", "_percentage2.png")
            self.plot_efficiency_percentage(self.all_expr_data[expr], title, out_file)

    def plot_efficiency(self, data, title, file, column_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        x = [i for i in range(1, len(data))]
        column_index = self.column_names.index(column_name)
        #use this y value to compare incrementally
        y = []
        for i in range(len(data) - 1):
            if data[i][column_index] == 0:
                if data[i + 1][column_index] == 0:
                    y.append(0)
                else:
                    y.append(100)
            else:
                y.append(100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index])
        # y = [100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index] if  data[i][column_index]!=0 else 100 for i in range(len(data) - 1)]
        file = file.replace(".png", "_" + column_name + ".png")

        #use the below y to compate to the original problem - initcompare
        # init_value = data[0][column_index]
        # y = [100 * (init_value - data[i][column_index]) / init_value for i in range(1,len(data))]
        # file = file.replace(".png", "_"+column_name+"_initcompare.png")

        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label=column_name+" ratio")
        ax1.plot(x, y, c="green")
        plt.xticks(x)
        title = title.replace(".cnf", "")
        plt.xlabel("Size of selective backbone")
        plt.ylabel("Percentage of "+column_name+" reduction")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()


        print(file)
        plt.savefig(file)

    def plot_efficiency_MC_BDD(self, data, title, file):
        """
        This plots the percentage reduction
        :param data:
        :param title:
        :param file:
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        # x = [i for i in range(1, len(data))]
        # print([data[i][3] for i in range(len(data))])
        # print([data[i][3] - data[i + 1][3] for i in range(len(data) - 1)])
        column_index_MC= self.column_names.index("MC")
        column_index_BDD= self.column_names.index("dag_size")

        #incremental
        # x = [100 * (data[i][column_index_MC] - data[i + 1][column_index_MC]) / data[i][column_index_MC] for i in
        #      range(len(data) - 1)]
        # y = [100 * (data[i][column_index_BDD] - data[i + 1][column_index_BDD]) / data[i][column_index_BDD] for i in
        #      range(len(data) - 1)]
        # file = file.replace(".png", "_Mc_BDD_incremental" + ".png")

        #use the below x and y to compare to initial problem
        x_init = data[0][column_index_MC]
        x = [100 * (x_init - data[i][column_index_MC]) / x_init for i in range(len(data) - 1)]
        y_init = data[0][column_index_BDD]
        y = [100 * (y_init - data[i][column_index_BDD]) / y_init for i in range(len(data) - 1)]
        file = file.replace(".png", "_Mc_BDD_initcompare" + ".png")

        # y = [ data[i][3]-data[i+1][3] for i in range(len(data)-1) ]
        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label= "")
        ax1.plot(x, y, c="green")
        # plt.xticks(x)
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        ax1.axline([0, 0], [100, 100], color="grey")
        title = title.replace(".cnf", "")
        plt.xlabel("BDD node count reduction percentage")
        plt.ylabel("Model count reduction percentage")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()

        print(file)
        plt.savefig(file)



    def plot_efficiency_percentage(self, data, title, file):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        n = len(data)
        x = [i for i in range(1, len(data))]
        # x = [ 100*(i/n) for i in range(1,len(data)) ]
        max_mc = data[0][3]
        y = [100 * (max_mc - data[i][3]) / max_mc for i in range(1, len(data))]
        # y = [ data[i][3]-data[i+1][3] for i in range(len(data)-1) ]
        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label="model count ratio")
        ax1.plot(x, y, c="green")
        plt.xticks(x)
        # plt.yticks(y)
        title = title.replace(".cnf", "")
        plt.xlabel("Percentage of selective backbone")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        plt.ylabel("Percentage of model count reduction from initial")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()
        # file = file.replace(".png", "_MC.png")
        print(file)
        plt.savefig(file)

    def plot_all_exprs(self, file, column_name):
        all_expr_data = []
        title = "All experiments"
        all_y = []
        all_x = []
        column_index = self.column_names.index(column_name)
        for expr in self.exprs:
            data=  self.all_expr_data[expr]
            y = [100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index] for i in
                 range(len(data) - 1)]
            all_y.append(y.copy())

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        x = [i for i in range(1, len(data))]
        # use this y value to compare incrementally


        file = file.replace(".png", "_" + column_name + "_incremental.png")

        n_lines = 10
        colors = pl.cm.jet(np.linspace(0, 1, n_lines))
        for i,y in  enumerate(all_y):
            ax1.plot(x, y,color="green")
        print(x)
        print(y)
        # ax1.scatter(x, y, c="green", label=column_name + " ratio")
        # ax1.plot(x, y, c="green")
        # plt.xticks(x)
        # title = title.replace(".cnf", "")
        plt.xlabel("Size of selective backbone")
        plt.ylabel("Percentage of " + column_name + " reduction")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()

        print(file)
        plt.savefig(file)

    def best_ratio_per_instance(self ):
        """
        used to be best_ratio_table_per_alg
        create  table with header(expr name, N(nb variables), Best p ( nb vars where best ratio was achieved), best ratio MC/BDD size, Initial BDD size, Initial MC)
        actually in here we only have access to one alg type
        so we need to iterate through all exprs and return best for all exprs
        !!! changed to WMC/size
        :return:
        """
        #TODO: should we look at best ratio per alg or overall?
        result = { e:{} for e in self.exprs}
        ratios =[]
        wmc_index = self.column_names.index("WMC")
        size_index = self.column_names.index("edge_count")
        N_index = self.column_names.index("nb_vars")
        # N_index = self.column_names.index("n_vars")
        for expr in self.all_expr_data.keys():
            init_WMC = self.all_expr_data[expr][0][wmc_index]
            init_size = self.all_expr_data[expr][0][size_index]
            best_index = 0
            best_wmc = 0
            best_size = 0
            init_ratio = init_WMC / init_size #TODO or do we want it with respect to initial ratio or take init ratio as iit best?
            best_ratio = init_ratio

            for i,data in enumerate(self.all_expr_data[expr]):
                if data[size_index] != 0:
                    r = data[wmc_index] / data[size_index]
                else:
                    r =0
                if r > best_ratio :
                    best_ratio = r
                    best_index = i
                    best_wmc = data[wmc_index]
                    best_size = data[size_index]
            # result[expr] = {"ratio":best_ratio, "mc":best_col1, "bdd":best_size, "index": best_index , "N": data[N_index],
            #                 "init_bdd":self.all_expr_data[expr][0][size_index], "init_MC": self.all_expr_data[expr][0][mc_index]}
            result[expr] = {"ratio":best_ratio, "wmc":best_wmc, "size":best_size, "index": best_index , "nb_vars": data[N_index],
                            "init_size":self.all_expr_data[expr][0][size_index], "init_WMC": self.all_expr_data[expr][0][wmc_index] }
            ratios.append(best_ratio)
        return result, ratios

    def count_proper_backbones(self):
        """
        Can only count at init stat file
        :return:
        """
        backbones = []
        if "init" not in self.filename:
            print("can't count backbones")
            return
        mc_index = self.column_names.index("MC")
        for expr in self.exprs:
            nb_backbone = 0
            for data in self.all_expr_data[expr]:
                if data[mc_index] == 0:
                    nb_backbone += 1
            if nb_backbone > 0:
                n_index = self.column_names.index("n_vars")
                N = self.all_expr_data[expr][0][n_index]
                cnf, mc = self.reload_expr(expr, N)
                print(expr, nb_backbone, N)
            # print(expr, ",", nb_backbone)
            backbones.append(nb_backbone)
        return backbones

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


    def reload_expr(self, expr_name,N):
        cnf = CNFmodel.CNF()
        literals = ["x" + str(i) for i in range(1, N+1)]
        cnf.bdd.declare(*literals)
        bdd_file = expr_name.replace(".cnf", ".dddmp")
        loaded_bdd = cnf.bdd.load(bdd_file)
        root = loaded_bdd[0]
        mc = root.count(len(literals))
        cnf.root_node = root
        return cnf, mc

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



def get_metric_per_alg(folder, labels, metric,columns, obj,padding=False):
    "return metrics for a specified objective, a full stats file"
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    results = []
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + ".csv"
        # stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        # print(stats_file)
        expr_data.read_stats_file(stats_file,full_expr_only=False, min_nb_expr=2, padding=padding)
        # print(expr_data.all_expr_data)
        # print("@@@@@@@@@@@@@@@@ get metric",expr_data.full_expr_name)
        percentage_results, smallest_n = expr_data.get_metric_wrt_initial_per_expr(metric, obj)
        # print(percentage_results.keys())
        # print(len(percentage_results))
        # exit(1234)

        results.append(percentage_results)

    return results, smallest_n

def all_best_ratios_grouped(expr_data_per_alg, algs, table_header):
    """
    return data such that it contains min and max for the set of experiments
    :param expr_data_per_alg:
    :param algs:
    :param table_header:
    :return:
    """
    expr_names = expr_data_per_alg[0].exprs
    table_content = []
    all_ratios = {}
    best_ratio_table = {i:{} for i in expr_names}
    name = expr_names[0].replace(".cnf", "").replace("./", "")
    folder = "/".join(name.split("/")[:-1])
    if "Planning" in folder:
        folder = folder.replace("pddlXpSym/", "")
    stats_file = "./paper_data/" + folder + "/dataset_stats_init_reorder.csv"
    print(type, stats_file)
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    init_expr = ExprData(columns)
    init_expr.read_stats_file(stats_file)
    proper_backbone = init_expr.count_proper_backbones()
    min_table_content = {k:None for k in table_header}
    max_table_content = {k:None for k in table_header}

    for exp_data, alg in zip(expr_data_per_alg, algs):
        result, ratios = exp_data.best_ratio_table_per_alg()
        all_ratios[alg] = [result, ratios]
    for index in range(len(expr_names)):
        best_ratio = 0
        best_result = []
        best_alg = ""
        for alg in algs:
            ratio = all_ratios[alg][1][index]
            if ratio >= best_ratio:
                best_ratio = ratio
                best_result = all_ratios[alg][0][expr_names[index]]
                best_alg = alg
        best_ratio_table[expr_names[index]] = {"best_ratio": best_ratio, "best_alg":best_alg, "details": best_result}
        name = expr_names[index].replace(".cnf","").replace("./","")
        name = name.split("/")[-1]
        row = [name, round(best_result["index"]/best_result["N"], 3), round(proper_backbone[index]/best_result["N"], 3) ,round(best_ratio,3), best_alg, best_result["init_bdd"], best_result["init_MC"],  best_result["index"], best_result["N"]]
        table_content.append(row)
    for e in best_ratio_table.keys():
        print(e, best_ratio_table[e])
    print(table_header)
    for l in table_content:
        print(l)
    return table_content

def all_best_ratios(expr_data_per_alg, algs, table_header):
    expr_names = expr_data_per_alg[0].exprs
    table_content = []
    all_ratios = {}
    best_ratio_table = {i:{} for i in expr_names}
    name = expr_names[0].replace(".cnf", "").replace("./", "")
    folder = "/".join(name.split("/")[:-1])
    if "Planning" in folder:
        folder = folder.replace("pddlXpSym/", "")
    stats_file = "./paper_data/" + folder + "/dataset_stats_init_reorder.csv"
    # print(type, stats_file)
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    init_expr = ExprData(columns)
    init_expr.read_stats_file(stats_file)
    proper_backbone = init_expr.count_proper_backbones()
    nb_backbone_count = sum(i > 0 for i in proper_backbone)
    nb_clauses = [get_nb_clauses(e) for e in expr_names ]

    for exp_data, alg in zip(expr_data_per_alg, algs):
        result, ratios = exp_data.best_ratio_table_per_alg()
        all_ratios[alg] = [result, ratios]
    for index in range(len(expr_names)):
        best_ratio = 0
        best_result = []
        best_alg = ""
        for alg in algs:
            ratio = all_ratios[alg][1][index]
            if ratio >= best_ratio:
                best_ratio = ratio
                best_result = all_ratios[alg][0][expr_names[index]]
                best_alg = alg
        best_ratio_table[expr_names[index]] = {"best_ratio": best_ratio, "best_alg":best_alg, "details": best_result}
        name = expr_names[index].replace(".cnf","").replace("./","")
        name = name.split("/")[-1]

        # table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size",
        # "Initial MC", "P", "N", "M", "mc","bdd",
        # "m/n", "mc/2^n","instance count", "nb inst with B"]


        row = [name, round(best_result["index"]/best_result["N"], 3), round(proper_backbone[index]/best_result["N"], 3) ,round(best_ratio,3), best_alg,
               best_result["init_bdd"], best_result["init_MC"],  best_result["index"], best_result["N"] , nb_clauses[index],  best_result["mc"], best_result["bdd"],
               round(nb_clauses[index]/ best_result["N"], 3), round(best_result["mc"]/ math.pow(2, best_result["N"]), 3),  len(expr_names), nb_backbone_count ]
        table_content.append(row)
    # for e in best_ratio_table.keys():
    #     print(e, best_ratio_table[e])
    # print(table_header)
    # for l in table_content:
    #     print(l)
    return table_content

def get_nb_clauses(filename):
    with open(filename, "r") as f:
        content = f.readline()
        nb_clauses = int(content.strip().split(" ")[3])
    return nb_clauses
def get_best_ratio_data(folder, labels, table_header, aggregate):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        print(type, stats_file)
        expr_data = ExprData(columns)
        expr_data.read_stats_file(stats_file)
        expr_datas.append(expr_data)
    table_content = all_best_ratios( expr_datas, labels, table_header)
    # table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size", "Initial MC", "P", "N", "M", "mc","bdd", "m/n", "mc/2^n","instance count", "nb inst with B"]

    if aggregate: #get min and max per experiments in a folder per column
        aggregated_table_content = [[], []]
        aggregated_table_content[0] = [old_value for old_value in table_content[0]]
        # aggregated_table_content[1] = [old_value for old_value in table_content[-1]]
        aggregated_table_content[1] = [ 0, 0 ]
        print("===================aggregate")
        for i in [1,2,3,5,6,7,8,9 ]: #column indexes to aggregate
            temp = [t[i] for t in table_content]
            aggregated_table_content[0][i].extend([ min(temp), max(temp)])
            # aggregated_table_content[1][i] = max(temp)
            if i == 3:
                aggregated_table_content[1][0] = np.average(temp)
                aggregated_table_content[1][1] = np.median(temp)
            print(folder,temp, table_header[i])
        return aggregated_table_content
    return table_content

def create_average_ratio_plot(folders, outputfile, title, labels, min_n, columns, obj, padding=False):
    ratios_to_average = [{} for i in labels]
    exps_to_average = [[] for i in labels]
    smallest_n = 600
    all_exps_to_average = []
    for folder in folders:
        print(folder)
        if obj == "WMC":
            folder_ratio_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "weighted_ratio", columns, obj, padding=padding)
        else:
            folder_ratio_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "ratio", columns, obj, padding=padding)
        if folder_smallest_n < smallest_n:
            smallest_n = folder_smallest_n
        for i, data in enumerate(folder_ratio_percentages):

            for d in data:
                if d not in all_exps_to_average:
                    all_exps_to_average.append(d)
                exps_to_average[i].append(d)
                ratios_to_average[i][d] = data[d]
                # print(d, data[d])
            # print("avg", ratios_to_average[i])
    #print per label all folders - plot should be here - need to first normalize to same length
    if smallest_n < min_n:
        smallest_n = min_n
    plot_data = []
    plot_data_before_sample = []
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [(100*i) /smallest_n for i in range(smallest_n+1)]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    labels = [s.replace("_1234", "") for s in labels]
    # labels[2] = "random_selection_ratio"
    plt.xlabel("Percentage of selective backbone size")
    if obj == "WCM":
        plt.ylabel("Average of WMC/size ratio percentage wrt initial ratio")
    else:
        plt.ylabel("Average of MC/size ratio percentage wrt initial ratio")
    lable_exp = {}
    for i, l in enumerate(labels):
        lable_exp[l] = []
        for exp_name in  exps_to_average[i]:
            lable_exp[l].append(exp_name)
    # print(lable_exp)
        # create the list of experiments to average - names that are present with all labels
    avg_exprs = []
    for name in all_exps_to_average:
        add = True
        for i, l in enumerate(labels):
            print(lable_exp,l)
            if name not in lable_exp[l]:
                add = False
                break
        if add:
            avg_exprs.append(name)
    # avg_exprs = list(set.intersection(*map(set, list(lable_exp.values())))) #these are the exprs common to all lables/objs/stat files
    filtered_ratios_to_average = [[0 for x in avg_exprs] for i in labels]  # data for each label/obj to aveage
    print("EXPRS TO AVERAGE:", len(avg_exprs))
    # print(avg_exprs)
    for i, l in enumerate(labels):
        j = 0
        for k in ratios_to_average[i].keys():
            if k in avg_exprs:
                filtered_ratios_to_average[i][j] = ratios_to_average[i][k]
                j += 1

    for i, l in enumerate(labels):
        avg_exprs_index = 0
        # print(l)
        label_average = [0 for i in range(smallest_n+1)]
        exprs_to_avg = len(filtered_ratios_to_average[i])
        count = 0
        for d in filtered_ratios_to_average[i]:
            # print(d)
            sampled_data, finished = sample_data(d,smallest_n+1)
            # print(len(sampled_data))
            label_average = [label_average[i]+sampled_data[i] for i in range(smallest_n+1)]
            # print(label_average)
            plot_data.append([l, avg_exprs[avg_exprs_index], sampled_data])
            plot_data_before_sample.append([l, avg_exprs[avg_exprs_index], d])
            avg_exprs_index += 1

        label_average = [label_average[i]/exprs_to_avg for i in range(smallest_n+1)]
        # print(len(x), len(label_average), smallest_n)
        ax1.scatter(x, label_average, c=colors[i], label=l, marker=marks[i])
        ax1.plot(x, label_average, c=colors[i])



        # print(label_average)
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    # plt.xticks(x)
    # plt.ylim(top=10)
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    # plt.show()
    # file = "./paper_data/all_avg_ratios_above100.png"
    # file = "./paper_data/all_avg_ratios_below100.png"
    # file = "./paper_data/all_avg_ratios.png"
    # expr_name = folders[0].split("/")[-2].replace(".cnf", "")
    # if len(folders) == 1:
    #     file = "./paper_data/" + expr_name+ "_avg_ratios.png"
    print(outputfile)
    outputfile = outputfile + "_pad"+str(padding)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()

    plot_data_file = outputfile + ".csv"
    f = open(plot_data_file, "w")
    writer = csv.writer(f, delimiter=',')
    for r in plot_data:
        row = [r[0], r[1]]
        row.extend(r[2])
        writer.writerow(row)
    f.close()

    plot_data_file = outputfile + "_before_sample.csv"
    f = open(plot_data_file, "w")
    writer = csv.writer(f, delimiter=',')
    for r in plot_data_before_sample:
        row = [r[0], r[1]]
        row.extend(r[2])
        writer.writerow(row)
        # writer.writerow(r[2])
    f.close()
    return outputfile

def create_average_efficiency_plot(folders,outputfile, title,  labels, min_n, columns, obj, padding=False, filter=True):
    # mc_column ='MC'
    # print(obj)
    # if obj == "WMC":
    #     mc_column = "WMC"
    # if "weighted" in outputfile:
    #     print("WEIGHTED MC")
    #     mc_column = 'WMC'
    MC_to_average = [{} for i in labels] #data for each label/obj to aveage
    size_to_average = [{} for i in labels]
    exps_to_average = [ [] for i in labels ]
    smallest_n = 600
    all_exps_to_average = []
    for folder in folders:
        folder_MC_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, obj ,columns, obj, padding=padding)
        folder_size_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "edge_count",columns, obj, padding=padding)
        # folder_size_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "BDD",columns)
        # print(folder, folder_smallest_n)
        #a folder is equivalent to a label/ objective
        if folder_smallest_n < smallest_n:
            smallest_n = folder_smallest_n
        for i, data in enumerate(folder_MC_percentages):
            for d in data:
                if d not in all_exps_to_average:
                    all_exps_to_average.append(d)
                exps_to_average[i].append(d)
                MC_to_average[i][d] = data[d]
            # print("avg", MC_to_average[i])
        for i, data in enumerate(folder_size_percentages):
            for d in data:
                size_to_average[i][d] = data[d]

    #print per label all folders - plot should be here - need to first normalize to same length
    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive"]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    plt.xlabel("Average BDD size percentage")
    if obj == "WMC":
        plt.ylabel("Average Weighted Model Count percentage")
    else:
        plt.ylabel("Average Model count percentage")
    labels = [s.replace("_1234", "") for s in labels]
    # labels[2] = "random_selection_ratio"

    lable_exp = {} # for each label list of exprs that are present
    for i, l in enumerate(labels):
        lable_exp[l] = []
        for exp_name in  exps_to_average[i]:
            lable_exp[l].append(exp_name)
    # print(lable_exp)

    if filter: #use this to eliminate experiments for which not all lalbels have data
        avg_exprs, filtered_MC_to_average, filtered_size_to_average  = filter_instances(MC_to_average,
                                                                                              all_exps_to_average,
                                                                                              labels, lable_exp,
                                                                                              size_to_average)

    plot_data = []
    for i, l in enumerate(labels):
        # print(l)
        mc_average = [0 for ii in range(smallest_n+1)]
        size_average = [0 for ii in range(smallest_n+1)]
        exprs_to_avg = len(filtered_MC_to_average[i])
        exp_name_index = 0
        for mc_d, bdd_d in zip(filtered_MC_to_average[i], filtered_size_to_average[i]):
            # print(d)

            sampled_mc_data, finished_expr = sample_data(mc_d,smallest_n+1)
            sampled_bdd_data, finished_expr = sample_data(bdd_d,smallest_n+1)
            print("------------------------------------------------------------")
            print(avg_exprs, exp_name_index)
            # print("sampled_mc_data: ", avg_exprs[exp_name_index], [round(100*x, 4) for x in  sampled_mc_data])
            # print("sampled_bdd_data: ", avg_exprs[exp_name_index],  [round(100*x, 4) for x in sampled_bdd_data])
            plot_data.append([l,avg_exprs[exp_name_index], "MC", [round(100*x, 4) for x in  sampled_mc_data]])
            plot_data.append([l,avg_exprs[exp_name_index], "SIZE", [round(100*x, 4) for x in  sampled_bdd_data]])
            exp_name_index += 1



            # print(label_average)
            # if not finished_expr:
            #     exprs_to_avg -= 1
            # else:
            mc_average = [mc_average[k] + sampled_mc_data[k] for k in range(smallest_n + 1)]
            size_average = [size_average[k] + sampled_bdd_data[k] for k in range(smallest_n + 1)]

            #TODO: plot here the data to be averaged

        mc_average = [mc_average[i]/exprs_to_avg for i in range(smallest_n+1)]
        size_average = [size_average[i]/exprs_to_avg for i in range(smallest_n+1)]


        # print(mc_average)
        # print(bdd_average)
        ax1.scatter(size_average, mc_average, c=colors[i], label=l, marker=marks[i])
        ax1.plot(size_average, mc_average, c=colors[i],  alpha=0.7, linewidth=1)

    plt.ylim(0, 1)
    plt.xlim(1, 0)

    ax1.axline([1, 1], [0, 0], color="grey")
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()
    # plt.show()
    # file = "./paper_data/all_avg_efficiency_AB.png"
    # file = "./paper_data/all_avg_efficiency_exceptAB.png"
    # file = "./paper_data/all_avg_efficiency_below100.png"
    # file = "./paper_data/all_avg_efficiency_above100.png"
    # file = "./paper_data/all_avg_efficiency.png"
    # expr_name = folders[0].split("/")[-2].replace(".cnf", "")
    # print(expr_name)
    # if len(folders) == 1:
    #     file = "./paper_data/" + expr_name+"_avg_efficiency.png"
    outputfile = outputfile + "_pad"+str(padding)
    print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()
    plot_data_file = outputfile + ".csv"
    f = open(plot_data_file, "w")
    writer = csv.writer(f, delimiter=',')
    for r in plot_data:
        row = [r[0],r[1], r[2]]
        row.extend(r[3])
        writer.writerow( row )
        # writer.writerow(r[3])

    return outputfile


def what_more_to_run(folders, labels, columns ):
    ecai23 = ['01_istance_K3_N15_M45_01.cnf', '01_istance_K3_N15_M45_02.cnf', '01_istance_K3_N15_M45_03.cnf',
    '01_istance_K3_N15_M45_04.cnf', '01_istance_K3_N15_M45_05.cnf', '01_istance_K3_N15_M45_06.cnf',
    '01_istance_K3_N15_M45_07.cnf', '01_istance_K3_N15_M45_08.cnf', '01_istance_K3_N15_M45_09.cnf',
    '01_istance_K3_N15_M45_10.cnf', '02_instance_K3_N30_M90_01.cnf',
    '02_instance_K3_N30_M90_02.cnf', '02_instance_K3_N30_M90_03.cnf',
    '02_instance_K3_N30_M90_04.cnf', '02_instance_K3_N30_M90_05.cnf',
    '02_instance_K3_N30_M90_06.cnf', '02_instance_K3_N30_M90_07.cnf',
    '02_instance_K3_N30_M90_08.cnf', '02_instance_K3_N30_M90_09.cnf',
    '02_instance_K3_N30_M90_10.cnf', '04_iscas89_s400_bench.cnf', '04_iscas89_s420_1_bench.cnf', '04_iscas89_s444_bench.cnf',
    '04_iscas89_s526_bench.cnf', '04_iscas89_s526n_bench.cnf', '05_iscas93_s344_bench.cnf','05_iscas93_s499_bench.cnf', '06_iscas99_b01.cnf', '06_iscas99_b02.cnf', '06_iscas99_b03.cnf', '06_iscas99_b06.cnf',
     '06_iscas99_b08.cnf', '06_iscas99_b09.cnf', '06_iscas99_b10.cnf',"07_blocks_right_2_p_t1.cnf",
     "07_blocks_right_2_p_t1.cnf", "07_blocks_right_2_p_t2.cnf", "07_blocks_right_2_p_t3.cnf", "07_blocks_right_2_p_t4.cnf", "07_blocks_right_2_p_t5.cnf", "07_blocks_right_3_p_t1.cnf", "07_blocks_right_3_p_t2.cnf", "07_blocks_right_4_p_t1.cnf", "08_bomb_b10_t5_p_t1.cnf", "08_bomb_b5_t1_p_t1.cnf", "08_bomb_b5_t1_p_t2.cnf", "08_bomb_b5_t1_p_t3.cnf", "08_bomb_b5_t1_p_t4.cnf", "08_bomb_b5_t1_p_t5.cnf", "08_bomb_b5_t5_p_t1.cnf", "08_bomb_b5_t5_p_t2.cnf", "09_coins_p01_p_t1.cnf", "09_coins_p02_p_t1.cnf", "09_coins_p03_p_t1.cnf", "09_coins_p04_p_t1.cnf", "09_coins_p05_p_t1.cnf", "09_coins_p05_p_t2.cnf", "09_coins_p10_p_t1.cnf", "10_comm_p01_p_t1.cnf", "10_comm_p01_p_t2.cnf", "10_comm_p02_p_t1.cnf", "10_comm_p03_p_t1.cnf", "11_emptyroom_d12_g6_p_t1.cnf", "11_emptyroom_d12_g6_p_t2.cnf", "11_emptyroom_d16_g8_p_t1.cnf", "11_emptyroom_d16_g8_p_t2.cnf", "11_emptyroom_d20_g10_corners_p_t1.cnf", "11_emptyroom_d24_g12_p_t1.cnf", "11_emptyroom_d28_g14_corners_p_t1.cnf", "11_emptyroom_d4_g2_p_t10.cnf", "11_emptyroom_d4_g2_p_t1.cnf", "11_emptyroom_d4_g2_p_t2.cnf", "11_emptyroom_d4_g2_p_t3.cnf", "11_emptyroom_d4_g2_p_t4.cnf", "11_emptyroom_d4_g2_p_t5.cnf", "11_emptyroom_d4_g2_p_t6.cnf", "11_emptyroom_d4_g2_p_t7.cnf", "11_emptyroom_d4_g2_p_t8.cnf", "11_emptyroom_d4_g2_p_t9.cnf", "11_emptyroom_d8_g4_p_t1.cnf", "11_emptyroom_d8_g4_p_t2.cnf", "11_emptyroom_d8_g4_p_t3.cnf", "11_emptyroom_d8_g4_p_t4.cnf", "12_flip_1_p_t10.cnf", "12_flip_1_p_t1.cnf", "12_flip_1_p_t2.cnf", "12_flip_1_p_t3.cnf", "12_flip_1_p_t4.cnf", "12_flip_1_p_t5.cnf", "12_flip_1_p_t6.cnf", "12_flip_1_p_t7.cnf", "12_flip_1_p_t8.cnf", "12_flip_1_p_t9.cnf", "12_flip_no_action_1_p_t10.cnf", "12_flip_no_action_1_p_t1.cnf", "12_flip_no_action_1_p_t2.cnf", "12_flip_no_action_1_p_t3.cnf", "12_flip_no_action_1_p_t4.cnf", "12_flip_no_action_1_p_t5.cnf", "12_flip_no_action_1_p_t6.cnf", "12_flip_no_action_1_p_t7.cnf", "12_flip_no_action_1_p_t8.cnf", "12_flip_no_action_1_p_t9.cnf", "13_ring2_r6_p_t1.cnf", "13_ring2_r6_p_t2.cnf", "13_ring2_r6_p_t3.cnf", "13_ring2_r8_p_t1.cnf", "13_ring2_r8_p_t2.cnf", "13_ring2_r8_p_t3.cnf", "13_ring_3_p_t1.cnf", "13_ring_3_p_t2.cnf", "13_ring_3_p_t3.cnf", "13_ring_3_p_t4.cnf", "13_ring_4_p_t1.cnf", "13_ring_4_p_t2.cnf", "13_ring_4_p_t3.cnf", "13_ring_5_p_t1.cnf", "13_ring_5_p_t2.cnf", "13_ring_5_p_t3.cnf", "14_safe_safe_10_p_t10.cnf", "14_safe_safe_10_p_t1.cnf", "14_safe_safe_10_p_t2.cnf", "14_safe_safe_10_p_t3.cnf", "14_safe_safe_10_p_t4.cnf", "14_safe_safe_10_p_t5.cnf", "14_safe_safe_10_p_t6.cnf", "14_safe_safe_10_p_t7.cnf", "14_safe_safe_10_p_t8.cnf", "14_safe_safe_10_p_t9.cnf", "14_safe_safe_30_p_t1.cnf", "14_safe_safe_30_p_t2.cnf", "14_safe_safe_30_p_t3.cnf", "14_safe_safe_30_p_t4.cnf", "14_safe_safe_30_p_t5.cnf", "14_safe_safe_30_p_t6.cnf", "14_safe_safe_5_p_t10.cnf", "14_safe_safe_5_p_t1.cnf", "14_safe_safe_5_p_t2.cnf", "14_safe_safe_5_p_t3.cnf", "14_safe_safe_5_p_t4.cnf", "14_safe_safe_5_p_t5.cnf", "14_safe_safe_5_p_t6.cnf", "14_safe_safe_5_p_t7.cnf", "14_safe_safe_5_p_t8.cnf", "14_safe_safe_5_p_t9.cnf", "15_sort_num_s_3_p_t10.cnf", "15_sort_num_s_3_p_t1.cnf", "15_sort_num_s_3_p_t2.cnf", "15_sort_num_s_3_p_t3.cnf", "15_sort_num_s_3_p_t4.cnf", "15_sort_num_s_3_p_t5.cnf", "15_sort_num_s_3_p_t6.cnf", "15_sort_num_s_3_p_t7.cnf", "15_sort_num_s_3_p_t8.cnf", "15_sort_num_s_3_p_t9.cnf", "15_sort_num_s_4_p_t1.cnf", "16_uts_k1_p_t10.cnf", "16_uts_k1_p_t1.cnf", "16_uts_k1_p_t2.cnf", "16_uts_k1_p_t3.cnf", "16_uts_k1_p_t4.cnf", "16_uts_k1_p_t5.cnf", "16_uts_k1_p_t6.cnf", "16_uts_k1_p_t7.cnf", "16_uts_k1_p_t8.cnf", "16_uts_k1_p_t9.cnf", "16_uts_k2_p_t1.cnf", "16_uts_k2_p_t2.cnf", "16_uts_k3_p_t1.cnf"]
    not_yet_seen = {f.split("/")[-2].split("_")[-1] : {} for f in folders}
    for folder in folders:
        for type in labels:
            if 'rand_dynamic' in folder and type == 'static' :
                continue
            not_yet_seen[folder.split("/")[-2].split("_")[-1]][type] = []
            seen_expr = []
            seen_expr_original_name = []

            stats_file = folder + "dataset_stats_" + type + ".csv"
            with (open(stats_file) as csvfile):
                reader = csv.reader(csvfile, delimiter=',')
                for line in reader:
                    if "input" in line[0]:
                        name = line[0].split("/")[-1]
                        if name.count(".") > 1:
                            name = name.replace(".", "_", name.count(".") - 1)
                        seen_expr.append(name)
                        seen_expr_original_name.append(line[0].split("/")[-1])
                        print(name)
            print("seen_expr: ", seen_expr)
            for e23 in ecai23:
                if e23.count(".") > 1:
                    e23 = e23.replace(".", "_", e23.count(".") - 1)
                if e23 not in seen_expr:
                    not_yet_seen[folder.split("/")[-2].split("_")[-1]][type].append(e23)

            #print in csv
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
            seen_expr_original_name.sort()
            stats_file = stats_file.replace("preproc2", "preproc3")
            f = open(stats_file, "w")
            writer = csv.writer(f, delimiter=',')
            for exp in seen_expr_original_name:
                writer.writerow(["./input/Dataset_preproc/"+exp])
                writer.writerow(columns)
                for d in expr_data.all_expr_data[exp]:
                    writer.writerow(d)
            f.flush()
            f.close()



    for folder in folders:
        for type in labels:
            if 'rand_dynamic' in folder and type == 'static':
                continue
            print(folder, type, len(not_yet_seen[folder.split("/")[-2].split("_")[-1]][type]), not_yet_seen[folder.split("/")[-2].split("_")[-1]][type])


def eval_progress(folders,outputfile, title,  labels, min_n, columns, obj, padding=False, same_length=False):
    nb_exps = 0
    correct = {f.split("/")[-2].split("_")[-1] : {} for f in folders}
    incorrect = {f : {} for f in folders}
    data = {f: {} for f in folders }
    all_incorrect = []

    for folder in folders:
        for type in labels:
            if 'rand_dynamic' in folder and type == 'static' :
                continue
            incorrect[folder][type] = []
            seen_expr = []
            correct[folder.split("/")[-2].split("_")[-1] ][type] = []
            nb_exps +=1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            with (open(stats_file) as csvfile):
                reader = csv.reader(csvfile, delimiter=',')
                for line in reader:
                    if "input" in line[0]:
                        name = line[0].split("/")[-1]
                        if name.count(".") > 1:
                            name = name.replace(".", "_", name.count(".") - 1)
                        seen_expr.append(name)


            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding)
            for e in expr_data.all_expr_data:
                if math.isinf(expr_data.all_expr_data[e][0][expr_data.column_names.index("WMC")] ) or math.isinf(expr_data.all_expr_data[e][0][expr_data.column_names.index("MC")] ):
                    incorrect[folder][type].append(e)
                    if e not in all_incorrect:
                        all_incorrect.append(e)
                else:
                    correct[folder.split("/")[-2].split("_")[-1] ][type].append(e)
            for e in seen_expr:
                if e in seen_expr and e not in incorrect[folder][type] and e not in correct[folder.split("/")[-2].split("_")[-1]][type]:
                    correct[folder.split("/")[-2].split("_")[-1]][type].append(e)
    # print(incorrect)
    print(correct)
    # for folder in folders:
    #     for type in labels:
    #         print(folder, type)
    #         # print("I: ", incorrect[folder][type])
    #
    #         print("C: ", correct[folder][type])
    # print(len(all_incorrect), all_incorrect)

    redo = ['08_bomb_b10_t10_p_t10.cnf', '08_bomb_b10_t10_p_t11.cnf', '08_bomb_b10_t10_p_t12.cnf', '08_bomb_b10_t10_p_t13.cnf', '08_bomb_b10_t10_p_t14.cnf', '08_bomb_b10_t10_p_t15.cnf', '08_bomb_b10_t10_p_t16.cnf', '08_bomb_b10_t10_p_t17.cnf', '08_bomb_b10_t10_p_t18.cnf', '08_bomb_b10_t10_p_t19.cnf', '08_bomb_b10_t10_p_t20.cnf', '08_bomb_b10_t10_p_t2.cnf', '08_bomb_b10_t10_p_t3.cnf', '08_bomb_b10_t10_p_t4.cnf', '08_bomb_b10_t10_p_t5.cnf', '08_bomb_b10_t10_p_t6.cnf', '08_bomb_b10_t10_p_t7.cnf', '08_bomb_b10_t10_p_t8.cnf', '08_bomb_b10_t10_p_t9.cnf', '08_bomb_b10_t5_p_t10.cnf', '08_bomb_b10_t5_p_t3.cnf', '08_bomb_b10_t5_p_t4.cnf', '08_bomb_b10_t5_p_t5.cnf', '08_bomb_b10_t5_p_t6.cnf', '08_bomb_b10_t5_p_t7.cnf', '08_bomb_b10_t5_p_t8.cnf', '08_bomb_b10_t5_p_t9.cnf', '08_bomb_b20_t5_p_t10.cnf', '08_bomb_b20_t5_p_t1.cnf', '08_bomb_b20_t5_p_t2.cnf', '08_bomb_b20_t5_p_t3.cnf', '08_bomb_b20_t5_p_t4.cnf', '08_bomb_b20_t5_p_t5.cnf', '08_bomb_b20_t5_p_t6.cnf', '08_bomb_b20_t5_p_t7.cnf', '08_bomb_b20_t5_p_t8.cnf', '08_bomb_b20_t5_p_t9.cnf', '08_bomb_b5_t5_p_t10.cnf', '08_bomb_b5_t5_p_t5.cnf', '08_bomb_b5_t5_p_t6.cnf', '08_bomb_b5_t5_p_t7.cnf', '08_bomb_b5_t5_p_t8.cnf', '08_bomb_b5_t5_p_t9.cnf', '10_comm_p03_p_t10.cnf', '10_comm_p03_p_t7.cnf', '10_comm_p03_p_t8.cnf', '10_comm_p03_p_t9.cnf', '10_comm_p04_p_t10.cnf', '10_comm_p04_p_t5.cnf', '10_comm_p04_p_t6.cnf', '10_comm_p04_p_t7.cnf', '10_comm_p04_p_t8.cnf', '10_comm_p04_p_t9.cnf', '10_comm_p05_p_t4.cnf', '10_comm_p05_p_t5.cnf', '10_comm_p05_p_t6.cnf', '10_comm_p05_p_t7.cnf', '10_comm_p05_p_t8.cnf', '10_comm_p10_p_t1.cnf']


def average_efficiency(folders, outputfile, title, labels, min_n, columns, obj, padding=False, same_expr=False,
                       filter_timeout=False, filter_conflict=False, subfolder="",median=False):
    "new func to plot avg"
    #if same_length : remove instances that have no results for all exprs
    wmc_data_to_average = {f: {} for f in folders}
    size_data_to_average = {f:{} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exps = 0
    smallest_n = 10000
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    else:
        outputfile = outputfile + "_ALL"
        title = title + " all"
    for folder in folders:
        for type in labels:
            if ('rand_dynamic' in folder and type == 'static' ) or  ('hybrid' in folder and type == 'static' ):
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder ) and type == 'static' :
                continue
            nb_exps +=1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_"+subfolder+ "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding, filter_timeout=filter_timeout, filter_conflict=filter_conflict)
            print("nb expr: ", len(expr_data.all_expr_data), expr_data.nb_completed_assignments)

            percentage_results_wmc, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr(obj, obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            percentage_results_size, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("edge_count", obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            for expr in percentage_results_wmc.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] = 1
                else:
                    all_expr_names_count[expr] += 1
            for expr in percentage_results_size.keys():
                if expr not in all_expr_names:
                    print("something missing")
                    exit(9)
            wmc_data_to_average[folder][type] = percentage_results_wmc
            size_data_to_average[folder][type] = percentage_results_size


    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N ")
    # exit(9)
    # print(nb_exps)
    # for e in all_expr_names_count:
    #     print(e, all_expr_names_count[e])

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive"]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    # plt.xlabel("Average BDD size percentage")
    plt.xlabel("Average representation size percentage")
    if median:
        plt.xlabel("Median representation size percentage")
    if obj == "WMC":
        if median:
            plt.ylabel("Median Weighted Model Count percentage")
        else:
            plt.ylabel("Average Weighted Model Count percentage")
    else:
        plt.ylabel("Average Model count percentage")

    index = 0
    data_file_name = outputfile + "_pad" + str(padding) + ".csv"
    data_file = open(data_file_name, "w")
    writer = csv.writer(data_file, delimiter=',')


    for f in folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
            if ('rand_dynamic' in f and l == 'static') or  ('hybrid' in f and l == 'static' ):
                continue
            wmc_to_average = []
            size_to_average = []
            for e in all_expr_names:
                if e in wmc_data_to_average[f][l] and e in size_data_to_average[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exps:
                        continue
                    #TODO: fixed : we have to ignore first datapoint as that is just 1 to show that initial ratio is 1
                    sampled_wmc_data, finished_expr = sample_data(wmc_data_to_average[f][l][e][1:], smallest_n )
                    sampled_wmc_data = [wmc_data_to_average[f][l][e][0]] + sampled_wmc_data
                    sampled_size_data, finished_expr = sample_data(size_data_to_average[f][l][e][1:], smallest_n )
                    sampled_size_data = [size_data_to_average[f][l][e][0]] + sampled_size_data
                    print("------------------smallest_n ", smallest_n, len(sampled_wmc_data),wmc_data_to_average[f][l][e][0],  sampled_wmc_data)

                    wmc_to_average.append(sampled_wmc_data.copy())
                    size_to_average.append(sampled_size_data.copy())

                    writer.writerow([e, l]+['wmc']+[100*k for k in  sampled_wmc_data])
                    writer.writerow([e, l]+['size']+[100*k for k in sampled_size_data])

            #create average and plot
            exprs_to_avg = len(wmc_to_average)
            print("-------------- Expr to avg", f, l, exprs_to_avg)
            avg_wmc = [ sum([ wmc_to_average[j][i] for j in range(len(wmc_to_average)) ]) / exprs_to_avg for i in range(len(wmc_to_average[0]))]
            avg_size = [ sum([ size_to_average[j][i] for j in range(len(size_to_average)) ]) / exprs_to_avg for i in range(len(size_to_average[0]))]
            #statistics.median
            if median:
                avg_wmc = [statistics.median([wmc_to_average[j][i] for j in range(len(wmc_to_average))])    for i in
                           range(len(wmc_to_average[0]))]
                avg_size = [statistics.median([size_to_average[j][i] for j in range(len(size_to_average))])  for i in
                            range(len(size_to_average[0]))]
            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            if "hybrid" in f:
                fname = "hybrid_wmc/"
            ax1.scatter(avg_size, avg_wmc, c=colors[index], label=HEUR_NAMES[fname]+" "+l, marker=marks[index])
            ax1.plot(avg_size, avg_wmc, c=colors[index], alpha=0.7, linewidth=1)
            index +=1

    data_file.close()
    plt.ylim(0, 1)
    plt.xlim(1, 0)

    ax1.axline([1, 1], [0, 0], color="grey")
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()
    outputfile = outputfile + "_pad" + str(padding)
    # print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()



def average_efficiency_WMC_MC(folders, outputfile, title, labels, min_n, columns, obj, padding=False, same_expr=False,filter_timeout=False, filter_conflict=False, subfolder=""):
    "new func to plot avg"
    #if same_length : remove instances that have no results for all exprs
    wmc_data_to_average = {f: {} for f in folders}
    size_data_to_average = {f:{} for f in folders}
    mc_data_to_average = {f:{} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exps = 0
    smallest_n = 600

    best_x = 0
    best_y = 0
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    for folder in folders:
        for type in labels:
            if 'rand_dynamic' in folder and type == 'static' :
                # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder ) and type == 'static' :
                continue
            nb_exps +=1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_"+subfolder+ "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding, filter_timeout=filter_timeout, filter_conflict=filter_conflict)

            percentage_results_wmc, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr( "WMC", "WMC")
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            percentage_results_size, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("edge_count", obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            #----MC
            percentage_results_mc, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("MC", "MC")
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n


            for expr in percentage_results_wmc.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] = 1
                else:
                    all_expr_names_count[expr] += 1
            for expr in percentage_results_size.keys():
                if expr not in all_expr_names:
                    print("something missing")
                    exit(9)
            wmc_data_to_average[folder][type] = percentage_results_wmc
            size_data_to_average[folder][type] = percentage_results_size
            #MC
            mc_data_to_average[folder][type] = percentage_results_mc


    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)
    #----------------------- VIRTUAL BEST
    best_ratio_x = [0] * (smallest_n+1)
    best_ratio_y = [0] * (smallest_n+1)
    best_dist = [100000] * (smallest_n+1)
    # ----------------------- VIRTUAL BEST
    # print(nb_exps)
    # for e in all_expr_names_count:
    #     print(e, all_expr_names_count[e])

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive"]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    # plt.xlabel("Average BDD size percentage")
    plt.xlabel("Average representation size percentage")
    plt.ylabel("Average Model count and weighted model count percentage")

    index = 0


    for f in folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
            if 'rand_dynamic' in f and l == 'static' :
                continue
            wmc_to_average = []
            size_to_average = []
            #MC
            mc_to_average = []
            for e in all_expr_names:
                if e in wmc_data_to_average[f][l] and e in size_data_to_average[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exps:
                        continue
                    sampled_wmc_data, finished_expr = sample_data(wmc_data_to_average[f][l][e], smallest_n + 1)
                    sampled_size_data, finished_expr = sample_data(size_data_to_average[f][l][e], smallest_n + 1)
                    #MC
                    sampled_mc_data, finished_expr = sample_data(mc_data_to_average[f][l][e], smallest_n + 1)

                    wmc_to_average.append(sampled_wmc_data.copy())
                    size_to_average.append(sampled_size_data.copy())
                    #MC
                    mc_to_average.append(sampled_mc_data.copy())

            # ----------------------- VIRTUAL BEST
            # print("-------------------------------------",f,"-------------------------------------------------")
            # min_wmc = [ min([ wmc_to_average[j][i] for j in range(len(wmc_to_average)) ])  for i in range(len(wmc_to_average[0])) ]
            # max_wmc = [ max([ wmc_to_average[j][i] for j in range(len(wmc_to_average)) ])  for i in range(len(wmc_to_average[0])) ]
            # variance_wmc = [ statistics.variance([ wmc_to_average[j][i] for j in range(len(wmc_to_average)) ])  for i in range(len(wmc_to_average[0])) ]
            #
            # min_size = [min([size_to_average[j][i] for j in range(len(size_to_average))]) for i in
            #             range(len(size_to_average[0]))]
            # max_size = [max([size_to_average[j][i] for j in range(len(size_to_average))]) for i in
            #             range(len(size_to_average[0]))]
            # variance_size = [statistics.variance([size_to_average[j][i] for j in range(len(size_to_average))]) for i in
            #                  range(len(size_to_average[0]))]
            #
            # # print("wmc min", "wmc max", "wmc variance", "size min", "size max", "size variance" )
            # print( f)
            # # print(min_wmc)
            # print("min, max, average variance")
            # print(min(variance_wmc), max(variance_wmc), sum(variance_wmc)/len(variance_wmc))
            # print(min(variance_size), max(variance_size), sum(variance_size)/len(variance_size))
            # print("-------------------------------------",f,"-------------------------------------------------")
            # ----------------------- VIRTUAL BEST
            #create average and plot
            exprs_to_avg = len(wmc_to_average)
            print("-------------- Expr to avg", f, l, exprs_to_avg)
            avg_wmc = [ sum([ wmc_to_average[j][i] for j in range(len(wmc_to_average)) ]) / exprs_to_avg for i in range(len(wmc_to_average[0]))]
            avg_size = [ sum([ size_to_average[j][i] for j in range(len(size_to_average)) ]) / exprs_to_avg for i in range(len(size_to_average[0]))]

            avg_mc = [ sum([ mc_to_average[j][i] for j in range(len(mc_to_average)) ]) / exprs_to_avg for i in range(len(mc_to_average[0]))]



            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            if "hybrid" in f:
                fname = "hybrid_wmc/"
            print(fname)
            if "WMC" not in fname:
                dist_index = 0
                for i in range(len(size_to_average[0])):
                    for j in range(len(mc_to_average)):
                        ex = size_to_average[j][i]
                        ey = wmc_to_average[j][i]
                        dist = get_distance([[0, 1]], [[ex, ey ]], 'euclidean')
                        print(dist)
                        if dist <= best_dist[dist_index]:
                            best_ratio_x[dist_index] = ex
                            best_ratio_y[dist_index] =ey
                            best_dist[dist_index] = dist

                    dist_index+=1

            ax1.scatter(avg_size, avg_wmc, c=colors[index], label=HEUR_NAMES[fname]+" "+l+" WMC", marker=marks[index])
            ax1.plot(avg_size, avg_wmc, c=colors[index], alpha=0.7, linewidth=1)
            index +=1

            # ax1.scatter(avg_size, avg_mc, c=colors[index], label=HEUR_NAMES[fname] + " " + l + " MC" , marker=marks[index])
            # ax1.plot(avg_size, avg_mc, c=colors[index], alpha=0.7, linewidth=1)
            # index +=1

    # ----------------------- VIRTUAL BEST
    #for each heuristic get virtual best
    # print(min(best_ratio_x[1:]), max(best_ratio_x[1:]), best_ratio_x)
    # print(min(best_ratio_y[1:]), max(best_ratio_y[1:]), best_ratio_y)
    # ax1.scatter(best_ratio_x, best_ratio_y, c="black", label="virtual best ", marker=marks[index])
    # ax1.plot(best_ratio_x, best_ratio_y,  c="black", alpha=0.7, linewidth=1)
    # ----------------------- VIRTUAL BEST

    plt.ylim(0, 1)
    plt.xlim(1, 0)

    ax1.axline([1, 1], [0, 0], color="grey")
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()
    outputfile = outputfile + "_pad" + str(padding)
    # print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()


def average_ratio(folders, outputfile, title, labels, min_n, columns, obj, padding=False, same_expr=False,filter_timeout=False,
                  filter_conflict=False, subfolder="",median=False):
    "new func to plot avg"
    ratio_data_to_average = {f: {} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 600
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    else:
        outputfile = outputfile + "_ALL"
        title = title + " all"
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder ) and type == 'static' :
            if ('rand_dynamic' in folder and type == 'static') or  ('hybrid' in folder and type == 'static' ) :
                continue
            nb_exprs+=1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_"+subfolder+ "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding,filter_timeout=filter_timeout, filter_conflict=filter_conflict)
            print("========",folder,type, len(expr_data.all_expr_data))
            percentage_results_ratio, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("weighted_ratio", obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            for expr in percentage_results_ratio.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] =1
                else:
                    all_expr_names_count[expr] +=1

            ratio_data_to_average[folder][type] = percentage_results_ratio

    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)

    data_file_name = outputfile + "_pad" + str(padding) + ".csv"
    data_file = open(data_file_name, "w")
    writer = csv.writer(data_file, delimiter=',')

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [(100 * i) / smallest_n for i in range(smallest_n + 1)]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    # labels = [s.replace("_1234", "") for s in labels]
    # labels[2] = "random_selection_ratio"
    plt.xlabel("Percentage of selective backbone size")
    if obj == "WMC":
        if median:
            plt.ylabel("Median of WMC/size ratio percentage wrt initial ratio")
        else:
            plt.ylabel("Average of WMC/size ratio percentage wrt initial ratio")
    else:
        plt.ylabel("Average of MC/size ratio percentage wrt initial ratio")
    index = 0

    for f in folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
            if ('rand_dynamic' in f and l == 'static') or  ('hybrid' in f and l == 'static' ):
                continue
            writer.writerow([f,l])
            ratio_to_average = []
            for e in all_expr_names:
                if e in ratio_data_to_average[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exprs:
                        continue
                    sampled_ratio_data, finished_expr = sample_data(ratio_data_to_average[f][l][e][1:], smallest_n )
                    sampled_ratio_data = [ratio_data_to_average[f][l][e][0]] + sampled_ratio_data

                    ratio_to_average.append(sampled_ratio_data.copy())
                    writer.writerow([e, l] + [100 * k for k in sampled_ratio_data])

            #create average and plot
            exprs_to_avg = len(ratio_to_average)
            print("-------------- Expr to avg", f, l, exprs_to_avg)
            avg_wmc = [ sum([ ratio_to_average[j][i] for j in range(len(ratio_to_average)) ]) / exprs_to_avg for i in range(len(ratio_to_average[0]))]
            if median:
                avg_wmc = [ statistics.median([ ratio_to_average[j][i] for j in range(len(ratio_to_average)) ])  for i in range(len(ratio_to_average[0]))]


            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            if "hybrid" in f:
                fname = "hybrid_wmc/"
            ax1.scatter(x, avg_wmc, c=colors[index], label=HEUR_NAMES[fname]+" "+l, marker=marks[index])
            ax1.plot(x, avg_wmc, c=colors[index])

            index +=1

    data_file.close()
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()
    # plt.yticks([i for i in range(26)])

    outputfile = outputfile + "_pad" + str(padding)
    print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()

def average_ratio_MC_WMC(folders, outputfile, title, labels, min_n, columns, obj, padding=False, same_expr=False,filter_timeout=False,
                  filter_conflict=False, subfolder=""):
    "new func to plot avg"
    ratio_data_to_average = {f: {} for f in folders}
    mc_ratio_data_to_average = {f: {} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 600
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder ) and type == 'static' :
            if 'rand_dynamic' in folder and type == 'static' :
                continue
            nb_exprs+=1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_"+subfolder+ "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding,filter_timeout=filter_timeout, filter_conflict=filter_conflict)
            print("========",folder,type, len(expr_data.all_expr_data))
            percentage_results_ratio, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("weighted_ratio", "WMC")
            mc_percentage_results_ratio, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("weighted_ratio", "MC")
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            for expr in percentage_results_ratio.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] =1
                else:
                    all_expr_names_count[expr] +=1

            ratio_data_to_average[folder][type] = percentage_results_ratio
            mc_ratio_data_to_average[folder][type] = mc_percentage_results_ratio

    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)

    # data_file_name = outputfile + "_pad" + str(padding) + ".csv"
    # data_file = open(data_file_name, "w")
    # writer = csv.writer(data_file, delimiter=',')

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [(100 * i) / smallest_n for i in range(smallest_n + 1)]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    # labels = [s.replace("_1234", "") for s in labels]
    # labels[2] = "random_selection_ratio"
    plt.xlabel("Percentage of selective backbone size")
    plt.ylabel("Average of MC/size and WMC/size ratio percentage wrt initial ratio")
    index = 0

    for f in folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
            if 'rand_dynamic' in f and l == 'static' :
                continue
            ratio_to_average = []
            mc_ratio_to_average = []
            for e in all_expr_names:
                if e in ratio_data_to_average[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exprs:
                        continue
                    sampled_ratio_data, finished_expr = sample_data(ratio_data_to_average[f][l][e], smallest_n + 1)
                    ratio_to_average.append(sampled_ratio_data.copy())

                    mc_sampled_ratio_data, finished_expr = sample_data(mc_ratio_data_to_average[f][l][e], smallest_n + 1)
                    mc_ratio_to_average.append(mc_sampled_ratio_data.copy())

            #create average and plot
            exprs_to_avg = len(ratio_to_average)
            print("-------------- Expr to avg", f, l, exprs_to_avg)
            avg_wmc = [ sum([ ratio_to_average[j][i] for j in range(len(ratio_to_average)) ]) / exprs_to_avg for i in range(len(ratio_to_average[0]))]
            avg_mc = [ sum([ mc_ratio_to_average[j][i] for j in range(len(mc_ratio_to_average)) ]) / exprs_to_avg for i in range(len(mc_ratio_to_average[0]))]

            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            if "hybrid" in f:
                fname = "hybrid_wmc/"
            ax1.scatter(x, avg_wmc, c=colors[index], label=HEUR_NAMES[fname]+" "+l+" WMC", marker=marks[index])
            ax1.plot(x, avg_wmc, c=colors[index])
            index +=1

            ax1.scatter(x, avg_mc, c=colors[index], label=HEUR_NAMES[fname] + " " + l+" MC", marker=marks[index])
            ax1.plot(x, avg_mc, c=colors[index])
            index += 1

    # data_file.close()
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    outputfile = outputfile + "_pad" + str(padding)
    print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()

def average_column(folders, outputfile, title, labels, min_n, columns, obj, padding=False, plot_tye="WMC", same_expr=False,
                   filter_timeout=False, filter_conflict=False, subfolder=""):
    "new func to plot avg"
    #if same_length : remove instances that have no results for all exprs
    col_data_to_average = {f: {l : [] for l in labels } for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exps = 0
    smallest_n = 600
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    for folder in folders:
        for type in labels:
            if 'rand_dynamic' in folder and type == 'static':
                continue
            nb_exps += 1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_" + subfolder + "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding, filter_timeout=filter_timeout, filter_conflict=filter_conflict)

            percentage_results_col, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr(plot_tye, obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            for expr in percentage_results_col.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] = 1
                else:
                    all_expr_names_count[expr] += 1
            col_data_to_average[folder][type] = percentage_results_col

    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)
    # print(nb_exps)
    # for e in all_expr_names_count:
    #     print(e, all_expr_names_count[e])

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive"]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    plt.xlabel("Average BDD size percentage")
    if obj == "WMC":
        plt.ylabel("Average Weighted Model Count percentage")
    else:
        plt.ylabel("Average Model count percentage")

    index = 0
    # data_file_name = outputfile + "_pad" + str(padding) + ".csv"
    # data_file = open(data_file_name, "w")
    # writer = csv.writer(data_file, delimiter=',')
    x = [(100 * i) / smallest_n for i in range(smallest_n + 1)]
    for f in folders:
        for l in labels:
            if 'rand_dynamic' in f and l == 'static' :
                continue
            col_to_average = []
            col_to_average_expr_name = []
            for e in all_expr_names:
                if e in col_data_to_average[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exps:
                        continue

                    sampled_col_data, finished_expr = sample_data(col_data_to_average[f][l][e], smallest_n + 1)

                    col_to_average.append(sampled_col_data.copy())
                    col_to_average_expr_name.append(e)


            #create average and plot
            exprs_to_avg = len(col_to_average)
            print("-------------- Expr to avg", f, l, exprs_to_avg, col_to_average_expr_name)
            avg_col = [ sum([ col_to_average[j][i] for j in range(len(col_to_average)) ]) / exprs_to_avg for i in range(len(col_to_average[0]))]

            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            if "hybrid" in f:
                fname = "hybrid_wmc/"
            ax1.scatter(x, avg_col, c=colors[index], label=HEUR_NAMES[fname]+" "+l, marker=marks[index])
            ax1.plot(x, avg_col, c=colors[index], alpha=0.7, linewidth=1)
            index +=1

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    outputfile = outputfile + "_pad" + str(padding)
    print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()





def filter_instances(MC_to_average, all_exps_to_average, labels, lable_exp, size_to_average):
    # create the list of experiments to average - names that are present with all labels
    avg_exprs = []
    for name in all_exps_to_average:
        add = True
        for i, l in enumerate(labels):
            if name not in lable_exp[l]:
                add = False
                break
        if add:
            avg_exprs.append(name)
    # print(all_exps_to_average)
    print(len(avg_exprs))
    # avg_exprs = list(set.intersection(*map(set, list(lable_exp.values()))))# with this I only take into account exprs for which dyn,static and rand have some results
    filtered_MC_to_average = [[0 for x in avg_exprs] for i in labels]  # data for each label/obj to aveage
    filtered_size_to_average = [[0 for x in avg_exprs] for i in labels]
    print("EXPRS TO AVERAGE:", len(avg_exprs))
    # print(avg_exprs)

    for i, l in enumerate(labels):
        j = 0
        for k in MC_to_average[i].keys():
            if k in avg_exprs:
                filtered_MC_to_average[i][j] = MC_to_average[i][k]
                filtered_size_to_average[i][j] = size_to_average[i][k]
                j += 1
    return avg_exprs, filtered_MC_to_average, filtered_size_to_average


def sample_data(data, smallest_n):
    n = len(data)
    # if n < smallest_n:
        # print("Less assignments then should be according to nb vars")
    # print(len(data),smallest_n)
    #return false in case there is less data then smallest n . smallest n is smallest nb variables, there can be less data in case exprs didn't finish
    return [data[int((i*n)/smallest_n)] for i in range(smallest_n)], n >= smallest_n
def create_best_ratio_table(out_folder, folders, labels,aggregate ):
    f = open(out_folder, "a+")
    writer = csv.writer(f, delimiter=',')
    table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size", "Initial MC", "P", "N", "M", "mc","bdd", "m/n", "mc/2^n","instance count", "nb inst with B"]
    # row = [name, best_result["index"] / best_result["N"], proper_backbone / best_result["N"], round(best_ratio, 3),  best_alg, best_result["init_bdd"], best_result["init_MC"], best_result["index"], best_result["N"]]
    writer.writerow(table_header)
    for folder in folders:
        print("analyze ", folder)
        table_content = get_best_ratio_data(folder, labels, table_header, aggregate)
        for line in table_content:
            writer.writerow(line)
    f.flush()
    f.close()

def read_ratio_table(filename, metric):
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    df = pd.read_csv(filename)
    print(df)
    metric = "Best adjusted ratio"
    df = df.sort_values(metric)
    print(df)
    # print(df.to_string())
    print("no improvement for : ", len(df[df['P/N']==0.00]))

    min_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).min()
    print(min_df)
    # min_df.to_csv('./paper_data/temp_min.csv')
    max_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).max()
    print(max_df)
    # max_df.to_csv('./paper_data/temp_max.csv')


    count_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).count()
    print("count")
    print(count_df)

    m = "Best adjusted ratio"
    m = "mc/2^n"
    temp = df.groupby(pd.qcut(df[m], 10)).count()
    print(temp)
    temp = df.groupby(pd.qcut(df[m], 10)).min()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).max()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).mean()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).median()
    print(temp)


#section to plot multiple experiments in one - same expr with a different selection criteria
def  plot_multiple(folder, expr_data_list, column_name, labels, plot_type, allow_different_len=False,overwrite=True):
    """
    column_name: what type of plot
    expr_data_list - list of all dtata for all labels, same order as in labels
    allow_different_len: allows exprs with different lengths and also if an expr is completely missing from a label
    """


    expr_data1 = expr_data_list[0]
    nb_exprs = [len(exprs.all_expr_data) for exprs in expr_data_list]
    expr_names = [e.exprs for e in expr_data_list]
    print(expr_names)

    if not allow_different_len:

        common_exprs = set.intersection(*map(set, expr_names))  # these are the exprs common to all lables/objs/stat files
        print("COMMON EXPERIMENTS FOR ALL HEURS:", len(common_exprs))
        print(common_exprs)
        if nb_exprs.count(nb_exprs[0]) != len(nb_exprs):
            print("different nb of exprs")
        figure_paths = {e:None for e in common_exprs}
    else:
        common_exprs = set.union(*map(set, expr_names))
        #check that at least one label hasa few lines in the csv
        figure_paths = {e: None for e in common_exprs}


    for  expr in common_exprs:
        print("=======================================",expr)
        if "SDD size" in expr_data1.column_names:
            size_column_name = "SDD size"
        else:
            size_column_name = "edge_count"

        if column_name == "size":
            column_name = size_column_name

        if "_" in column_name and "edge_count" != column_name:
            col = column_name.split("_")[0]
        else:
            col = column_name

        title = expr.split("/")[-1]
        out_file = folder + title.replace(".cnf", "_multiple_" + col + ".png")
        out_file = out_file.replace(".m", "m")

        efile = out_file.replace(".png", "_efficiency" + ".png")
        rfile = out_file.replace(".png", "_ratio" + ".png")

        if not overwrite:
            if "efficiency" in column_name and os.path.isfile(efile):
                print("file exist")
                figure_paths[expr] = efile
                continue
            elif "ratio" in column_name and os.path.isfile(rfile):
                print("file exist")
                figure_paths[expr] = rfile
                continue
            elif os.path.isfile(out_file): # plot for MC, WMC, size
                print("file exist")
                figure_paths[expr] = out_file
                continue
            else:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~ CREATING: ", expr)


        expr_data_list_values = [item.all_expr_data[expr] if expr in item.all_expr_data else [] for item in expr_data_list]
            # check_lens = [len(expr_d) for expr_d in expr_data_list_values[1:]] #use this in case initial compilation stats is in the eval too
        check_lens = [len(expr_d) for expr_d in expr_data_list_values]
            # if not check_lens.count(check_lens[0]) == len(check_lens): ## use this is no plot should be made if not all finished
            #     print("EXP DID NOT FINISH FOR ALL ALGS: ", expr, check_lens)
            #     continue

        if not allow_different_len and not check_lens.count(check_lens[0]) == len(check_lens):
            # use the below to get min number of pos for which exprs finished
            min_len = min(check_lens)
            expr_data_list_values = [item.all_expr_data[expr][:min_len] for item in expr_data_list]
                # print(min_len, check_lens)
            # print(expr_data1.column_names)



        column_index = expr_data1.column_names.index(col)
        # print("++++++++++++++++++",out_file, column_name, column_index)
        # column_index_BDD = expr_data1.column_names.index("edge_count")

        column_index_size = expr_data1.column_names.index(size_column_name)
        # column_index_BDD = expr_data1.column_names.index("dag_size")
        if "efficiency" in column_name:
                # column_index is used for MC or WMC or g2
            figure_paths[expr] = plot_efficiency_MC_BDD_ratio(expr_data_list_values, title, out_file, column_index, column_index_size, labels)
        elif "ratio" in column_name:
            figure_paths[expr] = plot_mc_bdd_ratio(expr_data_list_values, title, out_file, column_index, column_index_size, labels)
        else: #plot for MC, WMC, size
            # print("----------------------------------------------",out_file)
            column_index = expr_data1.column_names.index(column_name)
            figure_paths[expr] = plot_multiple_efficiency(expr_data_list_values, title, out_file, column_name, column_index, labels, plot_type)
            # break

    return figure_paths


def plot_multiple_efficiency(expr_data_list_values, title, out_file, column_name, column_index, labels, plot_type):
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)

    # colors = ["blue", "orange", "red", "green", "olive", "cyan", 'red']
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    plt.xlabel("Size of selective backbone")
    plt.ylabel("Percentage of " + column_name + " reduction")
    if column_name == "dag_size":
        plt.ylabel("Percentage of BDD size reduction")
    else:
        plt.ylabel("Percentage of Model count reduction")


    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]
        x = [i for i in range(1, len(expr_data_list_values[index]))]
        if plot_type == "inc":
            # use this y value to compare incrementally
            y_data = []
            for i in range(len(expr_data) - 1):
                if expr_data[i][column_index] == 0:
                    if expr_data[i + 1][column_index] == 0:
                        y_data.append(0)
                    else:
                        y_data.append(100)
                else:
                    y_data.append(100 * (expr_data[i][column_index] - expr_data[i + 1][column_index]) / expr_data[i][column_index])
            # y = [100 * (expr_data[i][column_index] - expr_data[i + 1][column_index]) / expr_data[i][column_index] for i in range(len(expr_data) - 1)]
            file = out_file.replace(".png", "_" + column_name + "_incremental.png")
            print(len(x),len(y_data))
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label + " ratio")
            ax1.plot(x, y_data, c=colors[index])

        elif plot_type == "init":
            # use the below y to compate to the original problem - initcompare
            file = out_file.replace(".png", "_" + column_name + "_initcompare.png")
            init_value = expr_data[0][column_index]
            y_data = [100 * (init_value - expr_data[i][column_index]) / init_value for i in range(1, len(expr_data))]
            # todo : expr_data2[i][column_index]) / init_value2 calculate this
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label + " ratio")
            ax1.plot(x, y_data, c=colors[index])

        else: #raw

            # file = out_file.replace(".png", "_raw.png")
            file = out_file
            x = [i for i in range(0, len(expr_data_list_values[index]))]
            y_data = [ d[column_index] for d in expr_data ]
            if column_name == "dag_size":
                plt.ylabel("BDD size")
            else:
                plt.ylabel( column_name)
                # plt.ylabel("Model count")
            print(len(x), len(y_data), title, label)
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label )
            ax1.plot(x, y_data, c=colors[index], marker=marks[index])
            # ax1.set_yscale('log') -- fails for bomb instances
            # ax1.set_yscale('symlog')
            # plt.xticks(rotation=30, ha='left')
            # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-30, ha="left", rotation_mode="anchor")
            # plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.xticks(x)
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    # plt.grid()

    print(file)
    plt.savefig(file)
    plt.clf()
    plt.close()
    return file
def plot_efficiency_MC_BDD( expr_data_list_values, title, file, column_index_MC, column_index_BDD, labels, plot_type):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ["blue", "orange", "green"]
    plt.xlabel("BDD node count reduction percentage")
    plt.ylabel("Model count reduction percentage")
    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]
        if plot_type == "inc":
        #incremental
            y = [100 * (expr_data[i][column_index_MC] - expr_data[i + 1][column_index_MC]) / expr_data[i][column_index_MC] for i in
                 range(len(expr_data) - 1)]
            x = [100 * (expr_data[i][column_index_BDD] - expr_data[i + 1][column_index_BDD]) / expr_data[i][column_index_BDD] for i in
                 range(len(expr_data) - 1)]
            ax1.scatter(x, y, c=colors[index], label=label)
            ax1.plot(x, y, c=colors[index])
            file = file.replace(".png", "_Mc_BDD_incremental" + ".png")
        elif plot_type == "init":
            #use the below x and y to compare to initial problem
            y_init = expr_data[0][column_index_MC]
            y = [100 * (y_init - expr_data[i][column_index_MC]) / y_init for i in range(len(expr_data) - 1)]
            x_init = expr_data[0][column_index_BDD]
            x = [100 * (x_init - expr_data[i][column_index_BDD]) / x_init for i in range(len(expr_data) - 1)]
            file = file.replace(".png", "_Mc_BDD_initcompare" + ".png")
            ax1.scatter(x, y, c=colors[index], label=label)
            # ax1.plot(x, y, c=colors[index])
        else:
            y = [ d[column_index_MC] for d in expr_data[1:] ]
            x = [ d[column_index_BDD] for d in expr_data[1:] ]

            ax1.scatter(x, y, c=colors[index], label=label)
            # ax1.plot(x, y, c=colors[index])
            file = file.replace(".png", "_Mc_BDD_raw" + ".png")
            plt.xlabel("BDD size ")
            plt.ylabel("Model count ")




    # plt.xticks(x)
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    ax1.axline([0, 0], [100, 100], color="grey")
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print(file)
    plt.savefig(file)
def plot_mc_bdd_ratio(expr_data_list_values, title, file, column_index, column_index_BDD, labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "orange",'red', "green","olive", "cyan" ]
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]

    col = file.split("_")[-1].strip(".png")
    plt.ylabel(col +"/ size")
    # plt.ylabel(col +"/ BDD node count ")
    plt.xlabel("selective backbone")
    file = file.replace(".png", "_ratio" + ".png")
    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]
        x = [i for i in range(0, len(expr_data_list_values[index]))]

        #use the below x and y to compare to initial problem
        # print(expr_data)
        y = [expr_data[i][column_index] / expr_data[i][column_index_BDD] if expr_data[i][column_index_BDD] != 0 else 0 for i in range(0, len(expr_data))]

        print(label)
        ax1.scatter(x, y, c=colors[index], label=label)
        ax1.plot(x, y, c=colors[index])



    # plt.xticks(x)
    # plt.xlim(0,1)
    # plt.ylim(0, 100)
    # ax1.axline([0, 0], [1, 1], color="grey")

    plt.title(title)
    title = title.replace(".cnf", "")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print(file)
    plt.savefig(file)
    plt.clf()
    plt.close()
    return file
def plot_efficiency_MC_BDD_ratio( expr_data_list_values, title, file, column_index_MC, column_index_BDD, labels):
    #Plot for efficiency
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o","p", "*", "x", "v", "^", "+", "1", "2", "3"]
    plt.xlabel("Size percentage")
    # plt.xlabel("BDD node count percentage")
    col = file.split("_")[-1].strip(".png")
    plt.ylabel(col+" percentage")
# file = file.replace(".png", "_Mc_BDD_efficiency" + ".png")
    file = file.replace(".png", "_efficiency" + ".png")
    for index, label in enumerate(labels):
        print(label)
        expr_data = expr_data_list_values[index]
        if len(expr_data) == 0:  #no results for this label
            print("break")
            continue
            #use the below x and y to compare to initial problem
        print("~~~~~~~~~~~~~~~~~~~~~~", expr_data, column_index_MC)
        y_init = expr_data[0][column_index_MC]
        if y_init == 0:
            break
        y = [ expr_data[i][column_index_MC] / y_init for i in range(1,len(expr_data))]
        # print("-----------------------",[ expr_data[i][column_index_MC] for i in range(1,len(expr_data))])
        x_init = expr_data[0][column_index_BDD]
        print(x_init)
        x = [ expr_data[i][column_index_BDD] / x_init for i in range(1, len(expr_data))]
        ax1.scatter(x, y, c=colors[index], label=label, marker=marks[index])
        ax1.plot(x, y, c=colors[index], alpha=0.7, linewidth=1)

    plt.ylim(0,1)
    plt.xlim(1, 0)

    ax1.axline( [1, 1], [0,0], color="grey")
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print("FILENAME  ", file)
    plt.savefig(file)
    plt.clf()
    plt.close()
    return file


def evaluate_folder(folder, labels, columns, overwrite=False):
    """
    plots=["MC_efficiency", "WMC_efficiency", "MC_ratio", "WMC_ratio", "MC", "WMC", "size"]
    create plots and return their names. lables(dyn, static, etc) one line in each plot for each label
    """
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    exp_names = []
    for type in labels:
        print(type)
        if 'rand_dynamic' in folder and type == 'static':
            continue
    #     if len(type.split("_")) == 1:
    #         columns[3] = type
    #     else:
    #         columns[3] =type.split("_")[1]
        stats_file = folder + "dataset_stats_" + type + ".csv"
        # if "dynamic_ratio" in type: #this should not be the first type in an incoming lit , exp names will break
        #     stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        #     columns2 = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
        #     expr_data = ExprData(columns2)
        #     expr_data.read_stats_file(stats_file)
        #     expr_data.convert_to_columns(columns, exp_names)
        #     print(expr_data.all_expr_data)
        # else:
        expr_data = ExprData(columns)
        expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=True )
        exp_names = expr_data.exprs
        print(stats_file, columns)
        expr_datas.append(expr_data)

    allow_different_len = True
    # mc_efficiency_plots = plot_multiple(folder, expr_datas, "MC_efficiency", labels, "init", allow_different_len=allow_different_len, overwrite=overwrite)
    wmc_efficiency_plots = plot_multiple(folder, expr_datas, "WMC_efficiency", labels, "init", allow_different_len=allow_different_len,overwrite=overwrite)
    # mc_ratio_plots = plot_multiple(folder, expr_datas, "MC_ratio", labels, "init", allow_different_len=allow_different_len,overwrite=overwrite)
    wmc_ratio_plots =  plot_multiple(folder, expr_datas, "WMC_ratio", labels, "init", allow_different_len=allow_different_len,overwrite=overwrite)

    plot_type = "raw"
    # mc_plots = plot_multiple(folder,expr_datas, "MC",labels, plot_type,allow_different_len=allow_different_len, overwrite=overwrite)
    wmc_plots = plot_multiple(folder,expr_datas, "WMC",labels, plot_type, allow_different_len=allow_different_len,overwrite=overwrite)
    size_plots = plot_multiple(folder, expr_datas, "size", labels, plot_type, allow_different_len=allow_different_len,overwrite=overwrite)

    # plot_multiple(folder,expr_datas, "g2",labels, plot_type)
    # plot_multiple(folder, expr_datas, "g2_ratio", labels, "init")
    # plot_multiple(folder, expr_datas, "g2_efficiency", labels, "init")
    # plot_multiple(folder, expr_datas, "dag_size", labels, plot_type)
    # plot_multiple(folder, expr_datas, "edge_count", labels, plot_type)

    plots = { "wmc":wmc_plots, "wmc_ratio":wmc_ratio_plots,  "wmc_efficiency":wmc_efficiency_plots, "size": size_plots }
    # plots = {'size':size_plots,"wmc_efficiency":wmc_efficiency_plots} #,  "mc_ratio":mc_ratio_plots, "wmc_ratio":wmc_ratio_plots, "mc_efficiency":mc_efficiency_plots, "wmc_efficiency":wmc_efficiency_plots }
    # plots = {"mc":mc_plots, "wmc":wmc_plots,  "mc_ratio":mc_ratio_plots, "wmc_ratio":wmc_ratio_plots, "mc_efficiency":mc_efficiency_plots, "wmc_efficiency":wmc_efficiency_plots, "size": size_plots }
    return plots
def count_all_backbones():
    exprs = [ "./paper_data/DatasetA/", "./paper_data/DatasetB/",
             "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/",
            "./paper_data/Planning/blocks/",  "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/", "./paper_data/Planning/comm/",
              "./paper_data/Planning/emptyroom/",  "./paper_data/Planning/flip/", "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    #, "./paper_data/Planning/comm/"]
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/",
    #          "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]

    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    proper_backbones = []
    for folder in exprs:
        type = "init"
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        print(stats_file)
        expr_data.read_stats_file(stats_file)
        b = expr_data.count_proper_backbones()
        proper_backbones.extend(b)
    c = 0
    for x in proper_backbones:
        if x > 0:
            c+=1
    print(c)
def plot_init():
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/"]
    exprs = [   "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/", "./paper_data/iscas/iscas99/"]
    # exprs = [  "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/flip/", "./paper_data/Planning/sort/",
    #          "./paper_data/Planning/uts/"]  # , "./paper_data/Planning/comm/"]

    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']

    for folder in exprs:

        type = "init"
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        print(stats_file)
        expr_data.read_stats_file(stats_file)
        for e in expr_data.all_expr_data.keys():
            fig = plt.figure()
            d = expr_data.all_expr_data[e]
            print(len(d))
            plt.plot([i for i in range(len(d))],[x[3] for x in d])
            f = e.replace(".cnf",".initplot.png")
            print(f)
            plt.savefig(f)
            # plt.show()

def read_ratio(ratio_file):
    df = pd.read_csv(ratio_file)
    df = df.iloc[:, -3:]
    c = list(df.columns)
    print(df)
    # df[[c[1], c[1]]].sub(df[c[0]], axis=0)
    # temp = df[c[1]] - df[c[0]]
    # df = df.drop(c[0], axis=1)
    # df[c[1]]= temp
    first_half = df.iloc[20:36]
    print(df)
    first_half.plot(kind="bar")
    plt.yscale("log")
    plt.show()

def create_time_table(folders, labels):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    f = open("./paper_data/times_table.csv", "a+")
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Expr type", "Init compilation"]+labels)
    for folder in folders:
        init_compilation = 0
        label_compilations = []
        for type in labels:
            stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file)
            init_compilation += sum([v for v in expr_data.init_compilation_time.values()]) / len(expr_data.init_compilation_time)
            last_compilation = sum(list(expr_data.finish_time.values())) / len(expr_data.finish_time)
            label_compilations.append(last_compilation)
        init_compilation = init_compilation / len(labels)
        writer.writerow([folder, round(init_compilation,3)]+[round(x,3) for x in label_compilations])

def compile_folder_with_stats(folder):
    """
    Compile cnfs and put mc and bdd size in csv file. Use this to compile cnfs that have been formed during selective process
    :param folder:
    :return:
    """
    f = open(folder+"compilations.csv", "a+")
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["instance", "p", "MC", "BDD", "comp_time" ])
    files = os.listdir(folder)
    # files.sort()
    # convert = lambda text: float(text) if text.isdigit() else text
    convert = lambda text: "{:02d}".format(int(text)) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    files.sort(key=alphanum)
    # files.sort(key=lambda x: "".join(x.split("_x")[:-1]).replace("_p", ""))
    print(files)
    # for f in files:
    #     f = os.path.join(folder, f)
    #     # checking if it is a file
    #     if os.path.isfile(f) and f.endswith(".cnf"):
    #         print(f, "".join(f.split("_x")[:-1]))
    # exit(10)
    for filename in files:
        f = os.path.join(folder, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith(".cnf"):
            cnf = _cnfBDD.CNF()
            print(f)
            start_time = time.perf_counter()
            b = cnf.load_file_with_apply(f)
            load_time = time.perf_counter() - start_time
            p = cnf.instance_name.split("_p")[1].split("_")[0]
            if "cnf" in p:
                p = p.strip(".cnf")
            print(f, p, cnf.root_node.count(cnf.n), cnf.root_node.dag_size)
            writer.writerow([f, p, cnf.root_node.count(cnf.n), cnf.root_node.dag_size, load_time ])
    f.flush()
    f.close()

def generate_latex_report_per_instance(filename):
    """
    return a latex text where for an instance we put all plots together
    """
    # plots = {"mc":mc_plots, "wmc":wmc_plots, "size":size_plots, "mc_ratio":mc_ratio_plots, "wmc_ratio":wmc_ratio_plots, "mc_efficiency":mc_efficiency_plots, "wmc_efficiency":wmc_efficiency_plots, }
    alg_types = ["static", "dynamic"]#, "random_selection_1234"  ]#, "dynamic", "random_selection_1234" ]
    # expr_folders = ["./results/Benchmark_WMC/"]  #"./results/sdd/wmc2022_track2_private_WMC/"
    # expr_folders = ["./results/Dataset_preproc2_WMC/" , "./results/Dataset_preproc2_wscore_half/"]#, "./results/Benchmark_wscore_estimate/"  ]#, "./results/Benchmark_WMC/"]#, "./results/Benchmark_wscore_occratio/","./results/Benchmark_wscore_adjoccratio/"]  #"./results/sdd/wmc2022_track2_private_WMC/"
    expr_folders =  [ "./results/Dataset_preproc_WMC/",   "./results/Dataset_preproc_wscore_half/", "./results/Dataset_preproc_wscore_estimate/"]#,  "./results/Dataset_preproc_rand_dynamic/"]
    # expr_folders = ["./results/Benchmark_preproc2_WMC/" , "./results/Benchmark_preproc2_wscore_half/"]#, "./results/Benchmark_wscore_estimate/"  ]#, "./results/Benchmark_WMC/"]#, "./results/Benchmark_wscore_occratio/","./results/Benchmark_wscore_adjoccratio/"]  #"./results/sdd/wmc2022_track2_private_WMC/"
    # expr_folders = ["./results/Benchmark_wscore_half/", "./results/Benchmark_wscore_occratio/","./results/Benchmark_wscore_adjoccratio/", "./results/Benchmark_wscore_estimate/"]  #"./results/sdd/wmc2022_track2_private_WMC/"
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"] #for d4
    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time', 'WMC', "logWMC"]  # for weighted sdd
    # plot_types = ["mc_efficiency", "wmc_efficiency", "mc_ratio", "wmc_ratio", "mc", "wmc", "size", "size"]
    exp_plots = {f: None for f in expr_folders}
    for f in expr_folders:
        exp_plots[f] = evaluate_folder(f, alg_types, columns, overwrite=False)

    print("creating latex for ", exp_plots)
    latex_report_order_by_instance_name(exp_plots, expr_folders, filename)
    # latex_report_efficiency_order_by_instance_name(exp_plots, expr_folders, filename)
    # latex_report_order_by_plot_type(exp_plots, expr_folders, filename)


def latex_report_for_average(expr_folders, labels, columns):
    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time', 'WMC', "logWMC"]  # for weighted sdd
    padding = True
    obj = "WMC"
    for f in expr_folders:
        type = "Benchmark WMC d4 " #f.split("/")[-2]
        # title = "Average weighted efficiency over " + type + " instances"
        #min_n is the number of iterations/sample size we want to have, in case an experiment has less it gets expanded , not eliminated
        title = "Average weighted efficiency over " + f.split("_")[-1] +  " instances"
        # create_average_efficiency_plot([f], f+type+"_avg_weighted_efficiency", title, labels, 50, columns,obj, padding=padding)
        average_efficiency([f], f+type+"_NEWavg_weighted_efficiency", title, labels, 50, columns,obj, padding=padding)
        title = "Average weighted ratio over " + f.split("_")[-1] + " instances"
        # create_average_ratio_plot([f], f+type+"_avg_weighted_ratio", title, labels, 50, columns, obj,padding=padding)
        average_ratio([f], f+type+"_NEWavg_ratio", title, labels, 50, columns,obj, padding=padding)
        # average_column([f], f+type+"_NEWavg_ratio", title, labels, 2, columns,obj, padding=padding)
        col = "WMC"
        title = "Average weighted " + col + "for " + f.split("_")[-1] + " instances"
        average_column([f], f+type+"_avg_" + col, title, alg_types, 50, columns, "WMC", padding=True, plot_tye=col)
        col = "edge_count"
        title = "Average weighted " + col+ "for " + f.split("_")[-1] + " instances"
        average_column([f], f+type+"_avg_" + col, title, alg_types, 50, columns, "WMC", padding=True, plot_tye=col)


def latex_report_order_by_instance_name(exp_plots, exp_title, filename):
    plot_types = [ "wmc",  "wmc_efficiency","wmc_ratio", "size"]#, "size" ]
    # plot_types = ["mc", "wmc", "mc_efficiency", "wmc_efficiency", "mc_ratio", "wmc_ratio", "size"]#, "size" ]
    # plot_types = ["size",  "wmc_efficiency"]#, "size", "size" ]

    instance_names = [ iname for iname in exp_plots[exp_title[0]]["wmc_efficiency"]]
    instance_names.sort()
    ordered_instance_names = {iname: {k : None for k in plot_types} for iname in instance_names}
    doc = px.Document(font_size="small", page_numbers=False)
    i = 0
    for section_title, exp_plot in exp_plots.items():
        with doc.create(px.Section('Results by instances for '+ px.escape_latex(section_title.replace("_", ' ')))):
            for iname in instance_names:
                with doc.create(px.Subsection('Results for instance ' + px.escape_latex(iname.replace("_", ' ')))):
                    for left_fig_i in range(0, len(plot_types), 2):
                        right_fig_i = left_fig_i+1
                        print(right_fig_i)
                        if right_fig_i >= len(plot_types):
                            left_fig_i = plot_types[left_fig_i]
                            left_fig = exp_plot[left_fig_i][iname]
                            with doc.create(px.Figure(position='h!' )) as left_pic:
                                left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                left_pic.add_caption(left_fig)
                        else:
                            left_fig_i = plot_types[left_fig_i]
                            right_fig_i = plot_types[right_fig_i]
                            print(left_fig_i)
                            print( iname)
                            left_fig = exp_plot[left_fig_i][iname]
                            right_fig = exp_plot[right_fig_i][iname]
                            with doc.create(px.Figure(position='h!')) as pic:
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.43\textwidth'))) as left_pic:
                                    left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                    left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                    left_pic.add_caption(left_fig)
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.45\textwidth'))) as right_pic:
                                    right_pic.add_image(right_fig, width=px.NoEscape(r'\textwidth'))
                                    right_fig = right_fig.replace("_", ' ').replace("/", '/ ')
                                    right_pic.add_caption(right_fig)
                                pic.add_caption("plot of "+ left_fig_i + " " + right_fig_i)
                doc.append(px.NoEscape(r'\clearpage'))
                doc.append(px.NewPage())
                            # pic.add_image(pytex.escape_latex(fig_name), width=pytex.NoEscape(r'0.7\textwidth'))
                            # fig_name = fig_name.replace("_", ' ').replace("/",'/ ')
                            # pic.add_caption(fig_name)
        i+=1
    doc.generate_pdf(filename, clean_tex=False, clean=False, compiler="pdflatex")


def latex_report_efficiency_order_by_instance_name(exp_plots, exp_title, filename):
    plot_types = ["wmc", "wmc_efficiency"]
    # plot_types = ["size",  "wmc_efficiency"]#, "size", "size" ]

    instance_names = [ iname for iname in exp_plots[exp_title[0]]["wmc_efficiency"]]
    instance_names.sort()
    ordered_instance_names = {iname: {k : None for k in plot_types} for iname in instance_names}
    doc = px.Document(font_size="small", page_numbers=False)
    i = 0
    for section_title, exp_plot in exp_plots.items():
        with doc.create(px.Section('Results by instances for '+ px.escape_latex(section_title.replace("_", ' ')))):
            for iname in instance_names:
                with doc.create(px.Subsection('Results for instance ' + px.escape_latex(iname.replace("_", ' ')))):
                    for left_fig_i in range(0, len(plot_types), 2):
                        right_fig_i = left_fig_i+1
                        print(right_fig_i)
                        if right_fig_i >= len(plot_types):
                            left_fig_i = plot_types[left_fig_i]
                            left_fig = exp_plot[left_fig_i][iname]
                            with doc.create(px.Figure(position='h!' )) as left_pic:
                                left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                left_pic.add_caption(left_fig)
                        else:
                            left_fig_i = plot_types[left_fig_i]
                            right_fig_i = plot_types[right_fig_i]
                            print(left_fig_i)
                            print( iname)
                            left_fig = exp_plot[left_fig_i][iname]
                            right_fig = exp_plot[right_fig_i][iname]
                            with doc.create(px.Figure(position='h!')) as pic:
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.43\textwidth'))) as left_pic:
                                    left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                    left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                    left_pic.add_caption(left_fig)
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.45\textwidth'))) as right_pic:
                                    right_pic.add_image(right_fig, width=px.NoEscape(r'\textwidth'))
                                    right_fig = right_fig.replace("_", ' ').replace("/", '/ ')
                                    right_pic.add_caption(right_fig)
                                pic.add_caption("plot of "+ left_fig_i + " " + right_fig_i)
                doc.append(px.NoEscape(r'\clearpage'))
                doc.append(px.NewPage())
                            # pic.add_image(pytex.escape_latex(fig_name), width=pytex.NoEscape(r'0.7\textwidth'))
                            # fig_name = fig_name.replace("_", ' ').replace("/",'/ ')
                            # pic.add_caption(fig_name)
        i+=1
    doc.generate_pdf(filename, clean_tex=False, clean=False, compiler="pdflatex")

def latex_report_order_by_plot_type(exp_plots, exp_title, filename):
    # plot_types = ["mc", "wmc", "mc_efficiency", "wmc_efficiency", "mc_ratio", "wmc_ratio", "size"]#, "size" ]
    plot_types = [ "wmc", "wmc_efficiency", "wmc_ratio", "size"]#, "size" ]
    # plot_types = ["size",  "wmc_efficiency"]#, "size", "size" ]

    instance_names = [ iname for iname in exp_plots[exp_title[0]]["wmc_efficiency"]]
    instance_names.sort()
    ordered_instance_names = {iname: {k : None for k in plot_types} for iname in instance_names}
    doc = px.Document(font_size="small", page_numbers=False)
    i = 0
    # for section_title, exp_plot in exp_plots.items():
    for section_title in plot_types:
        with doc.create(px.Section('Results for '+ px.escape_latex(section_title.replace("_", ' ')))):
            for iname in instance_names:
                with doc.create(px.Subsection('Results for instance ' + px.escape_latex(iname.replace("_", ' ')))):
                    #for each objective
                    for left_fig_i in range(0, len(exp_title), 2):
                        right_fig_i = left_fig_i+1
                        print(right_fig_i)
                        if right_fig_i >= len(exp_title):
                            left_fig_i = exp_title[left_fig_i]
                            left_fig = exp_plots[left_fig_i][section_title][iname]
                            with doc.create(px.Figure(position='h!' )) as left_pic:
                                left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                left_pic.add_caption(left_fig)
                        else:
                            left_fig_i = exp_title[left_fig_i]
                            right_fig_i = exp_title[right_fig_i]
                            print(left_fig_i)
                            print( iname)
                            print(exp_plots)
                            print(left_fig_i, iname,section_title)
                            left_fig = exp_plots[left_fig_i][section_title][iname]
                            right_fig = exp_plots[right_fig_i][section_title][iname]
                            with doc.create(px.Figure(position='h!')) as pic:
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.43\textwidth'))) as left_pic:
                                    left_pic.add_image(left_fig, width=px.NoEscape(r'\textwidth'))
                                    left_fig = left_fig.replace("_", ' ').replace("/", '/ ')
                                    left_pic.add_caption(left_fig)
                                with doc.create(px.SubFigure(position='b', width=px.NoEscape(r'0.45\textwidth'))) as right_pic:
                                    right_pic.add_image(right_fig, width=px.NoEscape(r'\textwidth'))
                                    right_fig = right_fig.replace("_", ' ').replace("/", '/ ')
                                    right_pic.add_caption(right_fig)
                                pic.add_caption("plot of "+ left_fig_i + " " + right_fig_i)
                doc.append(px.NoEscape(r'\clearpage'))
                doc.append(px.NewPage())
                            # pic.add_image(pytex.escape_latex(fig_name), width=pytex.NoEscape(r'0.7\textwidth'))
                            # fig_name = fig_name.replace("_", ' ').replace("/",'/ ')
                            # pic.add_caption(fig_name)
        i+=1
    doc.generate_pdf(filename, clean_tex=False, clean=False, compiler="pdflatex")

def read_folder():
    expr_folders = ["./results/Benchmark_WMC/"]  #
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC"] #for d4
    stats_file = expr_folders[0] + "dataset_stats_init.csv"
    expr_data = ExprData(columns)
    not_compiled_exprs = []
    with open(stats_file) as csvfile:
        # reader = csv.reader(csvfile, delimiter=',')
        content = csvfile.readlines()
        exp_name  = content[0].strip()
        prev_line = content[0].strip()
        for  i in range(1, len(content)):
            line = content[i].strip().split(",")
            if len(line) == 1 or ".cnf" in line[0]:  # if first line or start of new expr
                if prev_line[0] == 'p':  # epxr compile# d:
                    not_compiled_exprs.append(exp_name)
                exp_name = line[0]
            prev_line =content[i].strip().split(",")
    print(not_compiled_exprs)


    # expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=1)

    # print("expr: ",expr_data.exprs)
    # print("expr: ",["./input/Benchmark/"+e for e in expr_data.exprs])

def plot_best_point_per_instance(folder, labels, columns, overwrite=False):
    """
        plots=["MC_efficiency", "WMC_efficiency", "MC_ratio", "WMC_ratio", "MC", "WMC", "size"]
        create plots and return their names. lables(dyn, static, etc) one line in each plot for each label
        """
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    all_exp_names = []
    for type in labels:
        print(type)
        if len(type.split("_")) == 1:
            columns[3] = type
        else:
            columns[3] = type.split("_")[1]
        stats_file = folder + "dataset_stats_" + type + ".csv"
        expr_data_per_file = ExprData(columns)
        expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
        exp_names = expr_data_per_file.exprs
        print(stats_file, columns)
        expr_datas.append(expr_data_per_file)
        for e in exp_names:
            if e not in all_exp_names:
                all_exp_names.append(e)


    # expr_names = [e.exprs for e in expr_datas]
    print(all_exp_names)
    # common_exprs = set.intersection(*map(set, expr_names))

    for i, type in enumerate(labels):
        expr_data_per_file = expr_datas[i]

    #     #remove exprs that are not common
    #     remove_expr = []
    #     for e in expr_data_per_file.exprs:
    #         if e not in common_exprs:
    #             remove_expr.append(e)
    #     for e in remove_expr:
    #         expr_data_per_file.exprs.remove(e)
    #         if e in expr_data_per_file.all_expr_data:
    #             expr_data_per_file.all_expr_data.pop(e)
    #         else:
    #             print("something is wrong", e)
    #     print(len(expr_data_per_file.all_expr_data))

        dist_type = "euclidean"
        file = folder + '/best_efficiency_'+type+"_"+dist_type+'.png'
        get_best_efficiency_per_instance(expr_data_per_file, "WMC" , file=file , dist_type="euclidean")

        dist_type = "manhattan"
        file = folder + '/best_efficiency_'+type+"_"+dist_type+'.png'
        get_best_efficiency_per_instance(expr_data_per_file, "WMC" , file=file , dist_type="manhattan")


def get_best_efficiency_per_instance(expr_data_per_file, obj, file, dist_type="euclidean", save_plot=True):
    """
    expr_data_per_file id of type ExprData class
    Given a csv file data return best efficiency wrt to obj for each experiment
    """
    max_dist = 2
    if dist_type == "euclidean":
        max_dist = math.sqrt(2)
    #calculate initial efficiency
    obj_col_index = expr_data_per_file.column_names.index(obj) #WMC
    # size_col_index = expr_data_per_file.column_names.index("SDD size")
    size_col_index = expr_data_per_file.column_names.index("edge_count")
    best_efficiency_xs = {expr_name:0 for expr_name in expr_data_per_file.exprs}
    best_efficiency_ys = {expr_name:0 for expr_name in expr_data_per_file.exprs}
    best_distances = {expr_name:0 for expr_name in expr_data_per_file.exprs}
    for expr_name in expr_data_per_file.all_expr_data:
        expr_data = expr_data_per_file.all_expr_data[expr_name]
        init_eff_x = expr_data[0][size_col_index]
        init_eff_y = expr_data[0][obj_col_index]
        best_dist = max_dist #that's the max dist, we need to minimize this
        best_point_x = 1
        best_point_y = 1
        #virtual best efficiency is in top left corner with x=1 and y=1
        #now find best effiiciency, use euclid or manhattan discance
        for data_point in expr_data[1:]:
            eff_x = data_point[size_col_index] / init_eff_x
            eff_y = data_point[obj_col_index] / init_eff_y
            dist = get_distance([[0, 1]], [[eff_x, eff_y]], dist_type)
            if dist <= best_dist: #what should happen in case of eq
                best_dist = dist
                best_point_x = eff_x
                best_point_y = eff_y
        best_efficiency_xs[expr_name] = best_point_x
        best_efficiency_ys[expr_name] = best_point_y
        best_distances[expr_name] = best_dist
        if best_point_y < 0.5 and best_point_x < 0.5:
            print(expr_name, best_point_x,best_point_y,dist)
    x_axes = [x for x in best_efficiency_xs.values()]
    x_avg = sum(x_axes) / len(x_axes)
    y_axes = [x for x in best_efficiency_ys.values()]
    y_avg = sum(y_axes) / len(y_axes)

    if save_plot:
        print(len(x_axes))
        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_axes, y_axes, c="green", label="")
        ax1.scatter([x_avg], [y_avg], c="red", label="")
        # ax1.plot(x_axes, y_axes, c="green")
        plt.ylim(0, 1)
        plt.xlim(1, 0)

        ax1.axline([1, 1], [0, 0], color="grey")
        title = "Best efficiencies per instance" + file.split("/")[0] + file.split("/")[-1]
        plt.xlabel("Size percentage")
        plt.ylabel("WMC percentage")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()
        plt.savefig(file)
        plt.clf()
    return  best_distances



def get_distance(init_eff, data_point, dist_type):
    if dist_type == "euclidean":
        return euclidean_distances(init_eff, data_point)[0][0]
    else:
        return manhattan_distances(init_eff, data_point)[0][0]

def average_area_of_efficiency(folder, labels, columns ):
    """
        plots=["MC_efficiency", "WMC_efficiency", "MC_ratio", "WMC_ratio", "MC", "WMC", "size"]
        create plots and return their names. lables(dyn, static, etc) one line in each plot for each label
        """
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    exp_names = []
    for type in labels:
        print(type)
        if len(type.split("_")) == 1:
            columns[3] = type
        else:
            columns[3] = type.split("_")[1]
        stats_file = folder + "dataset_stats_" + type + ".csv"
        expr_data_per_file = ExprData(columns)
        expr_data_per_file.read_stats_file(stats_file, full_expr_only=True, min_nb_expr=1)
        exp_names = expr_data_per_file.exprs
        print(stats_file, columns)
        expr_datas.append(expr_data_per_file)

    expr_names = [e.exprs for e in expr_datas]
    print(expr_names)
    common_exprs = set.intersection(*map(set, expr_names))

    for i, type in enumerate(labels):
        expr_data_per_file = expr_datas[i]
        #remove exprs that are not common
        remove_expr = []
        for e in expr_data_per_file.exprs:
            if e not in common_exprs:
                remove_expr.append(e)
        for e in remove_expr:
            expr_data_per_file.exprs.remove(e)
            expr_data_per_file.all_expr_data.pop(e)
        print(type, len(expr_data_per_file.all_expr_data))

        obj = "WMC"
        efficiency_area_per_instance(expr_data_per_file, obj)

def efficiency_area_per_instance(expr_data_per_file, obj):
    obj_col_index = expr_data_per_file.column_names.index(obj)
    size_col_index = expr_data_per_file.column_names.index("edge_count")
    polygons = {expr_name:None for expr_name in expr_data_per_file.all_expr_data}
    polygons_area = []
    for expr_name in expr_data_per_file.all_expr_data:
        expr_data = expr_data_per_file.all_expr_data[expr_name]
        init_eff_x = expr_data[0][size_col_index]
        init_eff_y = expr_data[0][obj_col_index]
        coords = []
        for data_point in expr_data:
            eff_x = data_point[size_col_index] / init_eff_x
            eff_y = data_point[obj_col_index] / init_eff_y
            coords.append([eff_x,eff_y])
        polygon = Polygon(coords)
        polygons[expr_name] = polygon
        polygons_area.append(polygon.area)
    print("avg: ", sum(polygons_area)/len(polygons_area))
    shapes = [polygons[e] for e in polygons]
    # generate the overlay
    overlap = list(polygonize(unary_union(list(x.exterior for x in shapes))))


def plot_percentage_of_assigned_vars(folder, labels, columns,  filter_timeout=False, filter_conflict=False):
    #filter_timeout : true => eliminate experiments that timed out - only from that label
    expr_datas = {}
    exp_names = []
    nb_vars_for_completed = {}
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + ".csv"
        expr_data_per_file = ExprData(columns)
        expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,  filter_timeout=filter_timeout, filter_conflict=filter_conflict)
        expr_datas[type] = expr_data_per_file
    xs = {l:[] for l in labels }
    ys = {l:[] for l in labels }
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"] ]

    # exit(9)
    percentage_of_completion = {l:[] for l in labels}
    cumulative_percentage_of_completion = { l: [] for l in labels}
    for i, type in enumerate(labels):
        cumulative_percentage_of_completion[type] = [0 for k in range(0,101)]
        expr_data_per_file = expr_datas[type]
        p_col_index = expr_data_per_file.column_names.index("p")
        # print(expr_data_per_file.column_names)
        nb_vars_for_completed[type] = []
        nb_vars_col_index = expr_data_per_file.column_names.index("nb_vars")
        for expr_name in expr_data_per_file.all_expr_data:
            expr_data = expr_data_per_file.all_expr_data[expr_name]
            last_assigned = expr_data_per_file.nb_completed_assignments[expr_name]
            nb_vars = expr_data[-1][nb_vars_col_index]
            # print(expr_name, last_assigned, nb_vars, p_col_index, nb_vars_col_index)
            perc =  100 * (last_assigned/nb_vars)
            if perc == 100.0 :
                # print(type, expr_name, expr_data[-1][nb_vars_col_index] )
                nb_vars_for_completed[type].append(expr_data[-1][nb_vars_col_index] )
            percentage_of_completion[type].append( perc )
            for index, v in enumerate(cumulative_percentage_of_completion[type]):
                if index <= perc:
                    cumulative_percentage_of_completion[type][index] +=1


        # print(nb_vars_for_completed[type])
        # #count nb vars , group by count for when heur finishes
        # nb_occ = {nb: nb_vars_for_completed[type].count(nb) for nb in nb_vars_for_completed[type] }
        # nb_occ_keylist = list(nb_occ.keys())
        # nb_occ_keylist.sort()
        # nb_occ_sorted_values = {k: nb_occ[k] for k in nb_occ_keylist}  # number of occurance ok k percentage of completion
        #
        # for n in nb_occ_sorted_values:
        #     print(n, nb_occ_sorted_values[n])
        # exit(9)

        # plot for non cumulaive percentage finishe
        # occurrence = {percentage: percentage_of_completion[type].count(percentage) for percentage in percentage_of_completion[type] }
        # percentage_keylist = list(occurrence.keys())
        # percentage_keylist.sort()
        # sorted_values = [occurrence[k] for k in percentage_keylist] #number of occurance ok k percentage of completion
        # xs[type] = percentage_keylist.copy()
        # ys[type] = sorted_values.copy()
        # plot for non cumulaive percentage finishe
    fig = plt.figure()
    axs = fig.add_subplot(111)

    x = [k for k in range(0,101)]
    for ind, l in enumerate(labels):

        # axs.plot( xs[l], ys[l], color=colors[ind] )
        # axs.plot( xs[l], ys[l], color=colors[ind] )
        # axs.bar( xs[type], ys[type], color="blue"  )
        axs.plot( x, cumulative_percentage_of_completion[l] , color=colors[ind] , label=l )
        # axs.scatter( x, cumulative_percentage_of_completion[l] , color=colors[ind], label=l )


        c=0
        for ind, e in enumerate(percentage_of_completion[l]):
            if e == 100.0:
                c+=1

        print(l, folder, " all vars assigned ", c)
        handles, plot_labels = axs.get_legend_handles_labels()
        axs.legend(handles, plot_labels)

    plt.ylabel("number of instances")
    plt.xlabel("Percentage of variable assignments")
    plt.xticks([i*10 for i in range(0,11)] )
    plt.ylim(0,170)
        # plt.yscale("log")
        # plt.xticks(percentage_keylist)
        # plt.yticks(sorted_values)
    # plt.title( folder.split("_")[-1]+" "+type+" percentage of completion")
    plt.title( folder.split("_")[-1]+" percentage of completion")
    fig.tight_layout()


    plt.grid()
    # file = folder + "percentage_of_completion_"+type+".png"
    file = folder + "percentage_of_completion_all.png"
    if filter_timeout:
        file = folder + "percentage_of_completion_all_timeout_filter.png"
    if filter_conflict:
        file = folder + "percentage_of_completion_all_conflict_filter.png"
    plt.savefig(file)


def plot_time(folders, outputfile, title, labels, min_n, columns, obj, padding=False, same_expr=False,filter_timeout=False,
                  filter_conflict=False, subfolder=""):
    time_data = {f: {} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 600
    if subfolder != "":
        outputfile = outputfile + subfolder
        title = title + subfolder
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder) and type == 'static':
            if 'rand_dynamic' in folder and type == 'static' :
                continue
            nb_exprs += 1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_" + subfolder + "_" + type + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=padding,
                                      filter_timeout=filter_timeout, filter_conflict=filter_conflict)
            print("========", folder, type, len(expr_data.all_expr_data))
            percentage_results_ratio, folder_smallest_n = expr_data.get_metric_wrt_initial_per_expr("weighted_ratio",
                                                                                                    obj)
            if folder_smallest_n < smallest_n:
                smallest_n = folder_smallest_n
            for expr in percentage_results_ratio.keys():
                if expr not in all_expr_names:
                    all_expr_names.append(expr)
                    all_expr_names_count[expr] = 1
                else:
                    all_expr_names_count[expr] += 1

            time_data[folder][type] = percentage_results_ratio

    if min_n > smallest_n:
        smallest_n = min_n
    print("-----------------------SMALLEST N", smallest_n)

    data_file_name = outputfile + "_pad" + str(padding) + ".csv"
    data_file = open(data_file_name, "w")
    # writer = csv.writer(data_file, delimiter=',')

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [(100 * i) / smallest_n for i in range(smallest_n + 1)]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", mcolors.CSS4_COLORS["plum"],
              mcolors.CSS4_COLORS["darkorchid"], 'red', mcolors.CSS4_COLORS["darkred"], "grey"]
    marks = ["s", "o", "p", "*", "x", "v", "^", "+", "1", "2", "3"]
    # labels = [s.replace("_1234", "") for s in labels]
    # labels[2] = "random_selection_ratio"
    plt.xlabel("Percentage of selective backbone size")
    if obj == "WCM":
        plt.ylabel("Time")
    else:
        plt.ylabel("Average of MC/size ratio percentage wrt initial ratio")

    index = 0

    for f in folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f) and l == 'static':
            if 'rand_dynamic' in f and l == 'static' :
                continue
            # writer.writerow([f, l])
            sampled_time = []
            for e in all_expr_names:
                if e in time_data[f][l]:
                    if same_expr and all_expr_names_count[e] != nb_exprs:
                        continue
                    sampled_ratio_data, finished_expr = sample_data(time_data[f][l][e], smallest_n + 1)
                    sampled_time.append(sampled_ratio_data.copy())
                    # writer.writerow([e, l] + [100 * k for k in sampled_ratio_data])

            # create average and plot
            exprs_to_avg = len(sampled_time)
            print("-------------- Expr to avg", f, l, exprs_to_avg)
            avg_time = [sum([sampled_time[j][i] for j in range(len(sampled_time))]) / exprs_to_avg for i in
                       range(len(sampled_time[0]))]

            fname = f.split("_")[-1]
            if "rand_dynamic" in f:
                fname = "random"
            ax1.scatter(x, avg_time, c=colors[index], label=fname + "-" + l, marker=marks[index])
            ax1.plot(x, avg_time, c=colors[index])

            index += 1

    data_file.close()
    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    outputfile = outputfile + "_pad" + str(padding)
    print(outputfile)
    plt.savefig(outputfile)
    plt.clf()
    plt.close()

def plot_percentage_of_assigned_backbones(folder, labels, columns,filter_timeout=False, filter_conflict=False):
    expr_datas = {}
    exp_names = []
    nb_vars_for_completed = {}
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + ".csv"
        expr_data_per_file = ExprData(columns)
        expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=filter_timeout, filter_conflict=filter_conflict)
        expr_datas[type] = expr_data_per_file
    xs = {l:[] for l in labels }
    ys = {l:[] for l in labels }
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"] ]

    percentage_of_completion = {l:[] for l in labels}
    cumulative_percentage_of_backbone = { l: [] for l in labels}
    for i, type in enumerate(labels):
        cumulative_percentage_of_backbone[type] = [0 for k in range(0,101)]
        expr_data_per_file = expr_datas[type]
        p_col_index = expr_data_per_file.column_names.index("p")
        # print(expr_data_per_file.column_names)
        nb_vars_for_completed[type] = []
        nb_vars_col_index = expr_data_per_file.column_names.index("nb_vars")
        obj_col_index = expr_data_per_file.column_names.index("obj")
        wmc_col_index = expr_data_per_file.column_names.index("WMC")
        for expr_name in expr_data_per_file.all_expr_data:
            expr_data = expr_data_per_file.all_expr_data[expr_name]
            expr_obj_count = 0
            for data_point in expr_data:
                obj = data_point[obj_col_index]
                wmc =  data_point[wmc_col_index]
                if "WMC" in folder:
                    if obj != wmc:
                        expr_obj_count += 1
                elif obj >= 100 : # value i scale backbone with
                    expr_obj_count+=1
                    # print(expr_name, expr_obj_count)


            last_assigned = expr_data_per_file.nb_completed_assignments[expr_name]
            nb_vars = expr_data[-1][nb_vars_col_index]
            perc =  100 * (expr_obj_count/nb_vars)
            print(expr_name, perc, expr_obj_count, nb_vars)
            percentage_of_completion[type].append( perc )
            for index, v in enumerate(cumulative_percentage_of_backbone[type]):
                if index <= perc:
                    cumulative_percentage_of_backbone[type][index] +=1


        # print(nb_vars_for_completed[type])
        # #count nb vars , group by count for when heur finishes
        # nb_occ = {nb: nb_vars_for_completed[type].count(nb) for nb in nb_vars_for_completed[type] }
        # nb_occ_keylist = list(nb_occ.keys())
        # nb_occ_keylist.sort()
        # nb_occ_sorted_values = {k: nb_occ[k] for k in nb_occ_keylist}  # number of occurance ok k percentage of completion
        #
        # for n in nb_occ_sorted_values:
        #     print(n, nb_occ_sorted_values[n])
        # exit(9)

        # plot for non cumulaive percentage finishe
        # occurrence = {percentage: percentage_of_completion[type].count(percentage) for percentage in percentage_of_completion[type] }
        # percentage_keylist = list(occurrence.keys())
        # percentage_keylist.sort()
        # sorted_values = [occurrence[k] for k in percentage_keylist] #number of occurance ok k percentage of completion
        # xs[type] = percentage_keylist.copy()
        # ys[type] = sorted_values.copy()
        # plot for non cumulaive percentage finishe
    fig = plt.figure()
    axs = fig.add_subplot(111)

    x = [k for k in range(0,101)]
    for ind, l in enumerate(labels):

        # axs.plot( xs[l], ys[l], color=colors[ind] )
        # axs.plot( xs[l], ys[l], color=colors[ind] )
        # axs.bar(  x, cumulative_percentage_of_backbone[l] , color=colors[ind] , label=l )
        axs.plot( x, cumulative_percentage_of_backbone[l] , color=colors[ind] , label=l )
        axs.scatter( x, cumulative_percentage_of_backbone[l] , color=colors[ind], label=l )


        c=0
        for ind, e in enumerate(percentage_of_completion[l]):
            if e == 100.0:
                c+=1

        print(l, folder, " all vars assigned backbones ", c, cumulative_percentage_of_backbone)

        handles, plot_labels = axs.get_legend_handles_labels()
        axs.legend(handles, plot_labels)

    plt.ylabel("number of instances")
    plt.xlabel("Percentage of backbone assignments")
    plt.xticks([i*10 for i in range(0,11)] )
    plt.ylim(0,200)
        # plt.yscale("log")
        # plt.xticks(percentage_keylist)
        # plt.yticks(sorted_values)
    # plt.title( folder.split("_")[-1]+" "+type+" percentage of completion")
    plt.title( folder.split("_")[-1]+" percentage of completion")
    fig.tight_layout()


    plt.grid()
    # file = folder + "percentage_of_completion_"+type+".png"
    file = folder + "percentage_of_backbone_all.png"
    plt.savefig(file)
def best_ratio_per_alg(folders, labels, columns, subfolder=""):
    algs_stats = {f : {l: [] for l in labels if not (('rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')} for f in folders }
    algs_ratios = {f : {l: [] for l in labels if not (('rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')} for f in folders }
    best_alg_count = { f.split("_")[-1] + "_" + l: 0 for f in folders for l in labels if not (('rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') }
    all_exprs = []
    all_exprs_count = {}
    nb_c = 0

    for folder in folders:
        for type in labels:
            if (('rand_dynamic' in folder or 'hybrid_wmc' in folder) and type == 'static'):
                continue
            stats_file = folder + "dataset_stats_" + type + ".csv"
            if subfolder != "":
                stats_file = folder + "dataset_stats_"+subfolder+ "_" + type + ".csv"
            expr_data_per_file = ExprData(columns)
            expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
            stats, ratios = expr_data_per_file.best_ratio_per_instance()
            algs_stats[folder][type] = stats.copy()
            algs_ratios[folder][type] = ratios.copy()
            nb_c+=1
            for e in expr_data_per_file.all_expr_data.keys():
                if e not in all_exprs and e in ecai23:
                    all_exprs.append(e)
                    all_exprs_count[e] = 1
                else:
                    if e in ecai23:
                        all_exprs_count[e]+=1

    print(len(all_exprs))
    best_ratio_per_instance = {}
    best_ratio_per_instance_location = {}
    for e in all_exprs:
        if all_exprs_count[e] != nb_c:
            continue
        best_ratio_per_instance[e] = []
        best_ratio = 0
        best_stats = {}
        best_folder = ""
        best_label = ""
        best_location = 0
        random_best = False
        duplicate = False
        for f in folders:
            for l in labels:
                if (('rand_dynamic' in f or 'hybrid_wmc' in f) and l == 'static'):
                    continue
                if e in algs_stats[f][l]:
                    ratio = algs_stats[f][l][e]["ratio"]
                    if ratio >= best_ratio:
                        best_ratio = ratio
                        best_stats = algs_stats[f][l][e].copy()
                        best_folder = f
                        best_label = l
                        best_location = algs_stats[f][l][e]['index']
        best_ratio_per_instance[e] = {'ratio': best_ratio, 'stats':best_stats, "f":[ best_folder] , "l":[ best_label], "location": [best_location] }
        best_alg_count[best_folder.split("_")[-1] + "_" + best_label] += 1
        #count if multiple best exist
        if f == "rand_dynamic" and l == "dynamic":
            print("random best")
            random_best = True
        for f in folders:
            for l in labels:
                if (('rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static'):
                    continue
                if e in algs_stats[f][l]:
                    ratio = algs_stats[f][l][e]["ratio"]
                    if ratio == best_ratio_per_instance[e]['ratio'] and (f != best_ratio_per_instance[e]['f'][0]  or l != best_ratio_per_instance[e]['l'][0] ):
                        print("duplicate", e, f, l)
                        best_alg_count[f.split("_")[-1] + "_" + l] += 1
                        best_ratio_per_instance[e]['l'].append(l)
                        best_ratio_per_instance[e]['f'].append(f)
                        best_ratio_per_instance[e]['location'].append(algs_stats[f][l][e]["index"])
                        duplicate =True
        if random_best and not duplicate:
            print("RANDOM IS THE OVERALL BEST: ", e)
            exit(6)

    for alg in best_alg_count:
        print(alg, best_alg_count[alg], round( 100*best_alg_count[alg]/168, 2))

    # print(len(best_ratio_per_instance.keys()), best_ratio_per_instance.keys())
    # print(len(all_exprs))
    # for e in all_exprs:
    #     print(e, "FOLDER: ",  best_ratio_per_instance[e]['f'], "LABEL: ", best_ratio_per_instance[e]['l'] ,  best_ratio_per_instance[e]['ratio'] ,  best_ratio_per_instance[e]['location'] ) #best_ratio_per_instance[e]['ratio'] , best_ratio_per_instance[e]['stats']['init_WMC'] /  best_ratio_per_instance[e]['stats']['init_size']  )
    # stat_count = 0
    # for e in best_ratio_per_instance.keys():
    #     best_f = best_ratio_per_instance[e]['f']
    #     best_l = best_ratio_per_instance[e]['l']
    #     if best_f == folders[1]:
    #         if best_l == "static" and algs_stats[best_f]["dynamic"][e]['ratio'] < best_ratio_per_instance[e]['ratio']:
    #                 stat_count+=1
    #                 print(e, best_ratio_per_instance[e]['ratio'], algs_stats[best_f]["dynamic"][e]['ratio'] )
    # print(stat_count)
    


def histogram_of_best_points_per_instance(folders, labels, columns, distance="euclidean"):
    max_dist = 2
    if distance == "euclidean":
        max_dist = math.sqrt(2)

    all_exp_names = []
    best_distances = {f:{l:{} for l in labels } for f in folders}
    expr_datas = {f:{l:{} for l in labels } for f in folders}
    algs_performace = {f.split("_")[-1]+"_"+l : 0 for f in folders for l in labels }
    for folder in folders:
        for type in labels:
            print(type)
            if len(type.split("_")) == 1:
                columns[3] = type
            else:
                columns[3] = type.split("_")[1]
            stats_file = folder + "dataset_stats_" + type + ".csv"
            expr_data_per_file = ExprData(columns)
            expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
            exp_names = list(expr_data_per_file.all_expr_data.keys())
            print(stats_file, columns)
            expr_datas[folder][type] = expr_data_per_file
            best_dist = get_best_efficiency_per_instance(expr_data_per_file, "WMC", file="", dist_type=distance, save_plot=False)
            best_distances[folder][type] = best_dist
            for e in exp_names:
                if e not in all_exp_names:
                    all_exp_names.append(e)

    # expr_names = [e.exprs for e in expr_datas]
    print(all_exp_names)
    # common_exprs = set.intersection(*map(set, expr_names))



    # print(percentage_of_completion[type])
    # We can set the number of bins with the *bins* keyword argument.
    # print(len(percentage_of_completion[type]), percentage_of_completion[type][len(percentage_of_completion[type])-1])
    plot_data=[]
    c = ['blue', 'cyan', 'yellow', "green", 'lime', "orange"]
    plot_label = [f.split("_")[-1]+"_"+l for f in folders for l in labels ]
    i = 0
    best_distance_per_instance = {e:max_dist for e in all_exp_names}
    best_alg_per_instance = {e:"" for e in all_exp_names}
    for e in all_exp_names:
        for f in folders:
            for l in labels:
                if e in best_distances[f][l]:
                    if best_distances[f][l][e] < best_distance_per_instance[e]:
                        best_distance_per_instance[e] = best_distances[f][l][e]
                        best_alg_per_instance[e] = f.split("_")[-1]+"_"+l
    for e in all_exp_names:
        print(e, best_alg_per_instance[e])
        algs_performace[best_alg_per_instance[e]] +=1
        for f in folders:
            for l in labels:
                if e in best_distances[f][l]:
                    if best_distances[f][l] == best_distance_per_instance[e] and best_alg_per_instance[e] != f.split("_")[-1]+"_"+l:
                        algs_performace[f.split("_")[-1]+"_"+l] +=1

    for a in algs_performace:
        print(a, algs_performace[a])

    #
    width = 0
    for f in folders:
        for l in labels:
            plots = []
            print(f, l)
            fig = plt.figure()  # figsize=(50, 10))
            axs = fig.add_subplot(111)

            temp = [ max_dist - best_distances[f][l][e] if e in best_distances[f][l] else max_dist-max_dist  for e in all_exp_names ]
            p = axs.bar([i+width for i in range(len(all_exp_names))] , temp, color=c[0] )
            plot_data.append(temp)
            plots.append(p)
            i += 1
            # width += 0.15

            plt.xlabel("Instances")
            plt.ylim(0, max_dist  )
            plt.ylabel("Best point distace to corner")
            plt.legend(plots, plot_label)
            plt.title( folder.split("_")[-1]+" "+type+" best distace")
            fig.tight_layout()
            plt.grid()
            file = folder + "best_dist_"+l+".png"
            plt.savefig(file)
            plt.clf()
            plt.close()
    #
    # data_file = "./results/best_points.csv"
    # f = open(data_file, "w")
    # writer = csv.writer(f, delimiter=',')
    # writer.writerow(["expr"] + [f.split("_")[-1] + "_" + l for l in labels for f in folders])
    # for row in plot_data:
    #     writer.writerow(row)
    # f.close()
def histogram_of_best_points(folders, labels, columns, distance="euclidean"):
    max_dist = 2
    if distance == "euclidean":
        max_dist = math.sqrt(2)

    all_exp_names = []
    best_distances = {f:{l:{} for l in labels } for f in folders}
    expr_datas = {f:{l:{} for l in labels } for f in folders}
    for folder in folders:
        for type in labels:
            print(type)
            if len(type.split("_")) == 1:
                columns[3] = type
            else:
                columns[3] = type.split("_")[1]
            stats_file = folder + "dataset_stats_" + type + ".csv"
            expr_data_per_file = ExprData(columns)
            expr_data_per_file.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
            exp_names = list(expr_data_per_file.all_expr_data.keys())
            print(stats_file, columns)
            expr_datas[folder][type] = expr_data_per_file
            best_dist = get_best_efficiency_per_instance(expr_data_per_file, "WMC", file="", dist_type=distance, save_plot=False)
            best_distances[folder][type] = best_dist
            for e in exp_names:
                if e not in all_exp_names:
                    all_exp_names.append(e)

    # expr_names = [e.exprs for e in expr_datas]
    print(all_exp_names)
    # common_exprs = set.intersection(*map(set, expr_names))



    # print(percentage_of_completion[type])
    # We can set the number of bins with the *bins* keyword argument.
    # print(len(percentage_of_completion[type]), percentage_of_completion[type][len(percentage_of_completion[type])-1])
    plot_data=[]
    c = ['blue', 'cyan', 'yellow', "green", 'lime', "orange"]
    plot_label = [f.split("_")[-1]+"_"+l for f in folders for l in labels ]
    i = 0

    width = 0
    for f in folders:
        for l in labels:
            plots = []
            print(f, l)
            fig = plt.figure()  # figsize=(50, 10))
            axs = fig.add_subplot(111)

            temp = [ max_dist - best_distances[f][l][e] if e in best_distances[f][l] else max_dist-max_dist  for e in all_exp_names ]
            p = axs.bar([i+width for i in range(len(all_exp_names))] , temp, color=c[0] )
            plot_data.append(temp)
            plots.append(p)
            i += 1
            # width += 0.15

            plt.xlabel("Instances")
            plt.ylim(0, max_dist  )
            plt.ylabel("Best point distace to corner")
            plt.legend(plots, plot_label)
            plt.title( folder.split("_")[-1]+" "+type+" best distace")
            fig.tight_layout()
            plt.grid()
            file = folder + "best_dist_"+l+".png"
            plt.savefig(file)

    data_file = "./results/best_points.csv"
    f = open(data_file, "w")
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["expr"] + [f.split("_")[-1] + "_" + l for l in labels for f in folders])
    for row in plot_data:
        writer.writerow(row)
    f.close()
def count_nb_iterations(expr_folders,labels, columns ):
    obj = "WMC"
    counts = {}
    all_exprs = []
    for f in expr_folders:
        counts[f] = {}
        for type in labels:
            counts[f][type] = {}
            stats_file = f + "dataset_stats_" + type + ".csv"
            # stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
            expr_data = ExprData(columns)
            # print(stats_file)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False)
            for expr_name, e_data in expr_data.all_expr_data.items():
                counts[f][type][expr_name] = len(e_data)
                if expr_name not in all_exprs:
                    all_exprs.append(expr_name)

    data_file =  "./results/counts.csv"
    f = open(data_file, "w")
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["expr", "nb_vars",  expr_folders[0].split("_")[-1]+"_"+labels[0], expr_folders[0].split("_")[-1]+"_"+labels[1], expr_folders[0].split("_")[-1]+"_"+labels[2],
                     expr_folders[1].split("_")[-1]+"_"+labels[0], expr_folders[1].split("_")[-1]+"_"+labels[1], expr_folders[1].split("_")[-1]+"_"+labels[2] ] )
    for e in all_exprs:
        row = [e, expr_data.all_expr_data[e][0][expr_data.column_names.index("nb_vars")] ]
        for i in range(len(expr_folders)):
            for j in range(len(labels)):
                if e in counts[expr_folders[i]][labels[j]]:
                    row.append( counts[expr_folders[i]][labels[j]][e] )
                else:
                    row.append(0)
        writer.writerow(row)
        # writer.writerow([e, counts[expr_folders[0]][labels[0]][e], counts[expr_folders[0]][labels[1]][e], counts[expr_folders[0]][labels[2]][e],
        #                  counts[expr_folders[1]][labels[0]][e], counts[expr_folders[1]][labels[1]][e], counts[expr_folders[1]][labels[2]][e]]  )
    f.close()



def group_ecai23_data(expr_folders, alg_types, columns):
    dataset_AB = {f:{} for f  in expr_folders}
    dataset_iscas = {f:{} for f  in expr_folders}
    dataset_planning = {f:{} for f  in expr_folders}
    AB = ['01_istance_K3_N15_M45_01.cnf', '01_istance_K3_N15_M45_02.cnf', '01_istance_K3_N15_M45_03.cnf',
    '01_istance_K3_N15_M45_04.cnf', '01_istance_K3_N15_M45_05.cnf', '01_istance_K3_N15_M45_06.cnf',
    '01_istance_K3_N15_M45_07.cnf', '01_istance_K3_N15_M45_08.cnf', '01_istance_K3_N15_M45_09.cnf',
    '01_istance_K3_N15_M45_10.cnf', '02_instance_K3_N30_M90_01.cnf',
    '02_instance_K3_N30_M90_02.cnf', '02_instance_K3_N30_M90_03.cnf',
    '02_instance_K3_N30_M90_04.cnf', '02_instance_K3_N30_M90_05.cnf',
    '02_instance_K3_N30_M90_06.cnf', '02_instance_K3_N30_M90_07.cnf',
    '02_instance_K3_N30_M90_08.cnf', '02_instance_K3_N30_M90_09.cnf',
    '02_instance_K3_N30_M90_10.cnf', ]
    iscas = ['04_iscas89_s400_bench.cnf', '04_iscas89_s420_1_bench.cnf', '04_iscas89_s444_bench.cnf', '04_iscas89_s526_bench.cnf', '04_iscas89_s526n_bench.cnf',
             '05_iscas93_s344_bench.cnf', '05_iscas93_s499_bench.cnf',
        '06_iscas99_b01.cnf', '06_iscas99_b02.cnf', '06_iscas99_b03.cnf', '06_iscas99_b06.cnf', '06_iscas99_b08.cnf', '06_iscas99_b09.cnf', '06_iscas99_b10.cnf']
    planning = ["07_blocks_right_2_p_t1.cnf", "07_blocks_right_2_p_t2.cnf", "07_blocks_right_2_p_t3.cnf", "07_blocks_right_2_p_t4.cnf", "07_blocks_right_2_p_t5.cnf",
                "07_blocks_right_3_p_t1.cnf", "07_blocks_right_3_p_t2.cnf", "07_blocks_right_4_p_t1.cnf",
                "08_bomb_b10_t5_p_t1.cnf", "08_bomb_b5_t1_p_t1.cnf", "08_bomb_b5_t1_p_t2.cnf", "08_bomb_b5_t1_p_t3.cnf", "08_bomb_b5_t1_p_t4.cnf", "08_bomb_b5_t1_p_t5.cnf",
                "08_bomb_b5_t5_p_t1.cnf", "08_bomb_b5_t5_p_t2.cnf",
                "09_coins_p01_p_t1.cnf", "09_coins_p02_p_t1.cnf", "09_coins_p03_p_t1.cnf", "09_coins_p04_p_t1.cnf", "09_coins_p05_p_t1.cnf", "09_coins_p05_p_t2.cnf",
                "09_coins_p10_p_t1.cnf",
                "10_comm_p01_p_t1.cnf", "10_comm_p01_p_t2.cnf", "10_comm_p02_p_t1.cnf", "10_comm_p03_p_t1.cnf",
                "11_emptyroom_d12_g6_p_t1.cnf", "11_emptyroom_d12_g6_p_t2.cnf", "11_emptyroom_d16_g8_p_t1.cnf", "11_emptyroom_d16_g8_p_t2.cnf", "11_emptyroom_d20_g10_corners_p_t1.cnf",
                "11_emptyroom_d24_g12_p_t1.cnf", "11_emptyroom_d28_g14_corners_p_t1.cnf", "11_emptyroom_d4_g2_p_t10.cnf", "11_emptyroom_d4_g2_p_t1.cnf", "11_emptyroom_d4_g2_p_t2.cnf",
                "11_emptyroom_d4_g2_p_t3.cnf", "11_emptyroom_d4_g2_p_t4.cnf", "11_emptyroom_d4_g2_p_t5.cnf", "11_emptyroom_d4_g2_p_t6.cnf", "11_emptyroom_d4_g2_p_t7.cnf",
                "11_emptyroom_d4_g2_p_t8.cnf", "11_emptyroom_d4_g2_p_t9.cnf", "11_emptyroom_d8_g4_p_t1.cnf", "11_emptyroom_d8_g4_p_t2.cnf", "11_emptyroom_d8_g4_p_t3.cnf",
                "11_emptyroom_d8_g4_p_t4.cnf",
                "12_flip_1_p_t10.cnf", "12_flip_1_p_t1.cnf", "12_flip_1_p_t2.cnf", "12_flip_1_p_t3.cnf", "12_flip_1_p_t4.cnf", "12_flip_1_p_t5.cnf", "12_flip_1_p_t6.cnf", "12_flip_1_p_t7.cnf", "12_flip_1_p_t8.cnf", "12_flip_1_p_t9.cnf", "12_flip_no_action_1_p_t10.cnf", "12_flip_no_action_1_p_t1.cnf", "12_flip_no_action_1_p_t2.cnf", "12_flip_no_action_1_p_t3.cnf", "12_flip_no_action_1_p_t4.cnf", "12_flip_no_action_1_p_t5.cnf", "12_flip_no_action_1_p_t6.cnf", "12_flip_no_action_1_p_t7.cnf", "12_flip_no_action_1_p_t8.cnf", "12_flip_no_action_1_p_t9.cnf", "13_ring2_r6_p_t1.cnf", "13_ring2_r6_p_t2.cnf", "13_ring2_r6_p_t3.cnf", "13_ring2_r8_p_t1.cnf", "13_ring2_r8_p_t2.cnf", "13_ring2_r8_p_t3.cnf", "13_ring_3_p_t1.cnf", "13_ring_3_p_t2.cnf", "13_ring_3_p_t3.cnf", "13_ring_3_p_t4.cnf", "13_ring_4_p_t1.cnf", "13_ring_4_p_t2.cnf", "13_ring_4_p_t3.cnf", "13_ring_5_p_t1.cnf", "13_ring_5_p_t2.cnf", "13_ring_5_p_t3.cnf", "14_safe_safe_10_p_t10.cnf", "14_safe_safe_10_p_t1.cnf", "14_safe_safe_10_p_t2.cnf", "14_safe_safe_10_p_t3.cnf", "14_safe_safe_10_p_t4.cnf", "14_safe_safe_10_p_t5.cnf", "14_safe_safe_10_p_t6.cnf", "14_safe_safe_10_p_t7.cnf", "14_safe_safe_10_p_t8.cnf", "14_safe_safe_10_p_t9.cnf", "14_safe_safe_30_p_t1.cnf", "14_safe_safe_30_p_t2.cnf", "14_safe_safe_30_p_t3.cnf", "14_safe_safe_30_p_t4.cnf", "14_safe_safe_30_p_t5.cnf", "14_safe_safe_30_p_t6.cnf", "14_safe_safe_5_p_t10.cnf", "14_safe_safe_5_p_t1.cnf", "14_safe_safe_5_p_t2.cnf", "14_safe_safe_5_p_t3.cnf", "14_safe_safe_5_p_t4.cnf", "14_safe_safe_5_p_t5.cnf", "14_safe_safe_5_p_t6.cnf", "14_safe_safe_5_p_t7.cnf", "14_safe_safe_5_p_t8.cnf", "14_safe_safe_5_p_t9.cnf", "15_sort_num_s_3_p_t10.cnf", "15_sort_num_s_3_p_t1.cnf", "15_sort_num_s_3_p_t2.cnf", "15_sort_num_s_3_p_t3.cnf", "15_sort_num_s_3_p_t4.cnf", "15_sort_num_s_3_p_t5.cnf", "15_sort_num_s_3_p_t6.cnf", "15_sort_num_s_3_p_t7.cnf", "15_sort_num_s_3_p_t8.cnf", "15_sort_num_s_3_p_t9.cnf", "15_sort_num_s_4_p_t1.cnf", "16_uts_k1_p_t10.cnf", "16_uts_k1_p_t1.cnf", "16_uts_k1_p_t2.cnf", "16_uts_k1_p_t3.cnf", "16_uts_k1_p_t4.cnf", "16_uts_k1_p_t5.cnf", "16_uts_k1_p_t6.cnf", "16_uts_k1_p_t7.cnf", "16_uts_k1_p_t8.cnf", "16_uts_k1_p_t9.cnf", "16_uts_k2_p_t1.cnf", "16_uts_k2_p_t2.cnf", "16_uts_k3_p_t1.cnf" ]
    for f in expr_folders:
        for l in alg_types:
            if 'rand_dynamic' in f and l == 'static' :
                continue
            dataset_AB[f][l] = {}
            dataset_iscas[f][l] = {}
            dataset_planning[f][l] = {}
            stats_file = f + "dataset_stats_" + l + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False, filter_conflict=False)
            for e in expr_data.all_expr_data.keys():
                if e in AB:
                    dataset_AB[f][l][e] = expr_data.all_expr_data[e]
                elif e in iscas:
                    dataset_iscas[f][l][e] = expr_data.all_expr_data[e]
                elif e in planning:
                    dataset_planning[f][l][e] = expr_data.all_expr_data[e]
                else:
                    print("Something is strange")

    for folder in expr_folders:
        for type in alg_types:
            if 'rand_dynamic' in folder and type == 'static':
                continue
            file = open(folder + "dataset_stats_AB_" + type + ".csv" , "w")
            writer = csv.writer(file, delimiter=',')
            for e in dataset_AB[folder][type]:
                writer.writerow([e])
                writer.writerow(columns)
                for d in dataset_AB[folder][type][e]:
                    writer.writerow(d)
            file.flush()
            file.close()

            file = open(folder + "dataset_stats_iscas_" + type + ".csv", "w")
            writer = csv.writer(file, delimiter=',')
            for e in dataset_iscas[folder][type]:
                writer.writerow([e])
                writer.writerow(columns)
                for d in dataset_iscas[folder][type][e]:
                    writer.writerow(d)
            file.flush()
            file.close()

            file = open(folder + "dataset_stats_planning_" + type + ".csv", "w")
            writer = csv.writer(file, delimiter=',')
            for e in dataset_planning[folder][type]:
                writer.writerow([e])
                writer.writerow(columns)
                for d in dataset_planning[folder][type][e]:
                    writer.writerow(d)
            file.flush()
            file.close()

def count_conflicts_timeout(expr_folders, labels,columns, subfolder):
    conflict_instances = {f:{} for f in expr_folders}
    timeout_instances = {f:{} for f in expr_folders}

    for f in expr_folders:
        for l in labels:
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
            if 'rand_dynamic' in f and l == 'static' :
                continue
            total = 0
            conflict_instances[f][l] = []
            timeout_instances[f][l] = []
            stats_file = f + "dataset_stats_" + l + ".csv"
            if subfolder != "":
                stats_file = f + "dataset_stats_" +subfolder +"_" + l + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False, filter_conflict=False)

            mc_index = expr_data.column_names.index("MC")
            nb_vars_index = expr_data.column_names.index("nb_vars")
            p_index = expr_data.column_names.index("p")
            for expr in expr_data.all_expr_data.keys():
                total +=1
                last_row = expr_data.all_expr_data[expr][-1]
                if last_row[p_index] != last_row[nb_vars_index] and last_row[mc_index] == 0:
                    conflict_instances[f][l].append(expr)
                if last_row[p_index] != last_row[nb_vars_index] and last_row[mc_index] != 0:  # timeout
                    timeout_instances[f][l].append(expr)
            print("nb expr: ",total)

    print("conflict")
    for f in expr_folders:
        for l in labels:
            if 'rand_dynamic' in f and l == 'static' :
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :

                continue
            print(f.split("_")[-1], l, len(conflict_instances[f][l]))
            # print(len(conflict_instances[f][l]))
    print("timeout")
    for f in expr_folders:
        for l in labels:
            if 'rand_dynamic' in f and l == 'static' :
            # if ('rand_dynamic' in f or 'wscore_half' in f or 'wscore_estimate' in f ) and l == 'static' :
                continue
            print(f.split("_")[-1], l, len(timeout_instances[f][l]))
            # print(len(timeout_instances[f][l]))


def check_benchmark_preproc2():
    #eliminated : 'par8-1-c.cnf','par8-4-c.cnf', 'par8-2-c.cnf', 'par8-5-c.cnf',par8-3-c.cnf, 'par16-3.cnf', 'par16-1.cnf','par16-4.cnf' 'par16-5.cnf','par16-2.cnf',
    #'medium.cnf','par8-1.cnf', 'par8-2.cnf','par8-4.cnf','par8-3.cnf', 'par8-5.cnf','par32-2.cnf',
    # 'par32-5.cnf', 'par32-3.cnf',
    #missing from half static: or-70-10-8.cnf
    labels = ["static", "dynamic"]  # ,  "random_selection_1234" ]
    folders = ["./results/Benchmark_preproc2_WMC/", "./results/Benchmark_preproc2_wscore_half/",
                    "./results/Benchmark_preproc2_wscore_estimate/", "./results/Benchmark_preproc2_rand_dynamic/"]
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
               "obj"]  # for d4
    completed_exprs = {f: {} for f in folders}
    last_expr = {f: {} for f in folders}
    last_expr_var_count = {f: {} for f in folders}

    for f in folders:
        for l in labels:
            if 'rand_dynamic' in f and l == 'static':
                continue
            completed_exprs[f][l] = 0
            last_expr[f][l] = 0
            last_expr_var_count[f][l] = 0
            stats_file = f + "dataset_stats_" + l + ".csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,
                                      filter_timeout=False, filter_conflict=False)
            nb_vars_index = expr_data.column_names.index("nb_vars")
            for i, e in enumerate(all_files):
                if e.count(".") > 1:
                    e = e.replace(".", "_", e.count(".") - 1)  # actually first . will always be ./input so should skipp that
                # if "wscore_half" in f and "static" == l and e =='or-70-10-8.cnf': #TODO this is missing
                #     continue
                if e in expr_data.all_expr_data:
                    print(e)
                    completed_exprs[f][l] +=1
                    last_expr[f][l]  = i
                    last_expr_var_count[f][l] = expr_data.all_expr_data[e][0][nb_vars_index]
                    # if continuous == False:
                    #     print("gap", i, e)
                        # exit(12)
                # else:
                #     missing_exprs[f][l] += 1
                #     if continuous == False:
                #         print("missing", e)
                #     print("missing ", f, l, e , i)
                #     continuous = False
    for f in folders:
        for l in labels:
            if 'rand_dynamic' in f and l == 'static':
                continue
            print(f,l,completed_exprs[f][l], " last expr: ", last_expr[f][l], last_expr_var_count[f][l] )

def temp():
    compile_expr = ExprData(columns)
    stats_file = "./results/Dataset_preproc_final_wscore_estimate/dataset_stats_dynamic.csv"
    # stats_file = "./results/Dataset_preproc_final_hybrid_wmc/dataset_stats_dynamic.csv"
    compile_expr.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False, filter_conflict=False)

    no_compile_expr = ExprData(columns)
    # stats_file = "./results/Dataset_preproc_NO_COMPILE_hybrid_wmc/dataset_stats_dynamic_NO_COMPILE.csv"
    stats_file = "./results/Dataset_preproc_NO_COMPILE_wscore_estimate/dataset_stats_dynamic.csv"
    no_compile_expr.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False, filter_timeout=False,
                                 filter_conflict=False)
    c = 0
    t = 0
    for e in no_compile_expr.all_expr_data:
        if len(no_compile_expr.all_expr_data[e]) != len(compile_expr.all_expr_data[e]):
            print("diff len ", e)
            c +=1
        else:
            t+=1
    print(c,t)

def create_time_table_d4(folders, labels, columns, nocompile=False, cutoff={}):
    #cutoff is used for the no compiled setting to only count time for when we actually have looked results for
    import statistics
    # f = open("./results/times_table.csv", "w")
    # f = open("./results_aaai/times_table.csv", "w")
    if nocompile:
        # f = open("./results/times_table_NO_COMPILE_2.csv", "w")
        f = open("./results_aaai_final/times_table_NO_COMPILE_aaai.csv", "w")
    writer = csv.writer(f, delimiter=',')
    header = ["Expr" ]
    for f in folders:
        for l in labels:
            if 'rand_dynamic' in f or 'hybrid_wmc' in f:
                if l == 'dynamic':
                    header.append( f.split("_")[-2]+"_"+f.split("_")[-1] )
            else:
                header.append(f.split("_")[-1] + "_" + l)
    writer.writerow(header)
    print(header)
    time_data = {f: {} for f in folders}
    nb_assigned_vars_data = {f: {} for f in folders}
    nb_vars_data = {}
    nb_backbones_data = {f: {} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 60000
    all_exprs = []
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder) and type == 'static':
            if ('rand_dynamic' in folder and type == 'static') or ( "hybrid_wmc" in folder and type == 'static'):
                continue
            nb_exprs += 1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            expr_data = ExprData(columns)
            if nocompile:
                expr_cutoff = {}
                print("in cutoff? ", folder.split("_")[-1] )
                if folder.split("_")[-1] in cutoff:
                    if type in cutoff[folder.split("_")[-1]]:
                        expr_cutoff = cutoff[folder.split("_")[-1]][type]
                        print("cutoff: ", folder)
                expr_data.read_nocompile_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,
                                                    filter_timeout=False, filter_conflict=False,cutoff=expr_cutoff)
            else:
                expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,
                                      filter_timeout=False, filter_conflict=False)
            print("========", folder, type, len(expr_data.all_expr_data))
            time_data[folder][type] = expr_data.get_finishing_times()
            nb_assigned_vars_data[folder][type] = expr_data.nb_completed_assignments
            nb_vars_col_index = expr_data.column_names.index("nb_vars")
            if len(nb_vars_data) == 0:
                nb_vars_data = { e : expr_data.all_expr_data[e][0][nb_vars_col_index] for e in expr_data.all_expr_data}
            #count backbones - only for compiled version
            if not nocompile:
                nb_backbones_data[folder][type] = {}

                obj_col_index = expr_data.column_names.index("obj")
                wmc_col_index = expr_data.column_names.index("WMC")
                for expr_name in expr_data.all_expr_data:
                    edata = expr_data.all_expr_data[expr_name]
                    expr_backbone_count = 0
                    for data_point in edata[1:]:
                        obj = data_point[obj_col_index]
                        wmc = data_point[wmc_col_index]
                        if "WMC" in folder:
                            if obj != wmc:
                                expr_backbone_count += 1
                        elif obj >= 100:  # value I scale backbone with
                            expr_backbone_count += 1
                    nb_backbones_data[folder][type][expr_name] = expr_backbone_count


            print("times: ", folder, type, len(time_data[folder][type] ))

    all_exprs = list(expr_data.all_expr_data.keys())
    for e in all_exprs:
        writer.writerow([e ]+ [ time_data[f][l][e] for f in folders for l in labels if  not ( ( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') ])
    mins = []
    maxs = []
    avgs = []
    medians = []
    for f in folders:
        for l in labels:
            if ('rand_dynamic' in f or 'hybrid_wmc' in f) and l == 'static':
                continue
            all = [time_data[f][l][e] for e in all_exprs]
            print(f, l)
            mins.append(round(min(all), 3))
            maxs.append(round(max(all), 3))
            avgs.append(round(sum(all) / len(all), 3))
            medians.append(round(statistics.median(all), 3))
    print(len(mins))
    writer.writerow(["min:"] + mins)
    writer.writerow(["max:"] + maxs)
    writer.writerow(["avg:"] + avgs)
    writer.writerow(["median:"] + medians)

    writer.writerow(["Nb assigned vars"])
    writer.writerow(["expr"]+[ f.split("_")[-1] + "_" + l  for f in folders for l in labels if not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') ]+ ["number of variables"])
    for e in all_exprs:
        writer.writerow([e ]+ [ nb_assigned_vars_data[f][l][e] for f in folders for l in labels if not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') ] + [nb_vars_data[e]]  )
    mins = []
    maxs = []
    avgs = []
    medians = []
    for f in folders:
        for l in labels:
            if ('rand_dynamic' in f or 'hybrid_wmc' in f) and l == 'static':
                continue
            all = [ 100 * nb_assigned_vars_data[f][l][e]/nb_vars_data[e] for e in all_exprs]
            print(f, l)
            mins.append(round(min(all), 3))
            maxs.append(round(max(all), 3))
            avgs.append(round(sum(all) / len(all), 3))
            medians.append(round(statistics.median(all), 3))
    print(len(mins))
    writer.writerow(["min:"] + mins)
    writer.writerow(["max:"] + maxs)
    writer.writerow(["avg:"] + avgs)
    writer.writerow(["median:"] + medians)


    if not nocompile:
        writer.writerow(["Nb unit clauses assigned"])
        writer.writerow(["expr"] + [f.split("_")[-1] + "_" + l for f in folders for l in labels if
                                    not ( ( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')] + ["number of variables"])
        for e in all_exprs:
            writer.writerow([e] + [nb_backbones_data[f][l][e] for f in folders for l in labels if
                                   not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')] + [nb_vars_data[e]])



    print(nb_assigned_vars_data)

def create_percent_of_assigned_table_d4(folders, labels, columns, nocompile=False, cutoff={}):
    #cutoff is used for the no compiled setting to only count time for when we actually have looked results for
    import statistics
    f = open("./results_aaai_final/percent_assigned_table_aaai.csv", "w")
    writer = csv.writer(f, delimiter=',')
    header = ["Expr" ]
    for f in folders:
        for l in labels:
            if 'rand_dynamic' in f or 'hybrid_wmc' in f:
                if l == 'dynamic':
                    header.append( f.split("_")[-2]+"_"+f.split("_")[-1] )
            else:
                header.append(f.split("_")[-1] + "_" + l)
    writer.writerow(header)
    print(header)
    nb_assigned_vars_data = {f: {} for f in folders}
    nb_vars_data = {}
    nb_backbones_data = {f: {} for f in folders}
    all_expr_names = []
    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 60000
    all_exprs = []
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder) and type == 'static':
            if ('rand_dynamic' in folder and type == 'static') or ( "hybrid_wmc" in folder and type == 'static'):
                continue
            nb_exprs += 1
            stats_file = folder + "dataset_stats_" + type + ".csv"
            expr_data = ExprData(columns)
            if nocompile:
                expr_cutoff = {}
                if folder.split("_")[-1] in cutoff:
                    expr_cutoff = cutoff[folder.split("_")[-1]][type]
                    print("cutoff: ", folder)
                expr_data.read_nocompile_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,
                                                    filter_timeout=False, filter_conflict=False,cutoff=expr_cutoff)
            else:
                expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=0, padding=False,
                                      filter_timeout=False, filter_conflict=False)
            print("========", folder, type, len(expr_data.all_expr_data))
            nb_assigned_vars_data[folder][type] = expr_data.nb_completed_assignments
            if type == "static" and "_WMC" in folder:
                print(folder)
                print(expr_data.nb_completed_assignments)
                exit(88)
            nb_vars_col_index = expr_data.column_names.index("nb_vars")
            if len(nb_vars_data) == 0:
                nb_vars_data = { e : expr_data.all_expr_data[e][0][nb_vars_col_index] for e in expr_data.all_expr_data}
            #count backbones - only for compiled version
            if not nocompile:
                nb_backbones_data[folder][type] = {}

                obj_col_index = expr_data.column_names.index("obj")
                wmc_col_index = expr_data.column_names.index("WMC")
                for expr_name in expr_data.all_expr_data:
                    edata = expr_data.all_expr_data[expr_name]
                    expr_backbone_count = 0
                    for data_point in edata[1:]:
                        obj = data_point[obj_col_index]
                        wmc = data_point[wmc_col_index]
                        if "WMC" in folder:
                            if obj != wmc:
                                expr_backbone_count += 1
                        elif obj >= 100:  # value I scale backbone with
                            expr_backbone_count += 1
                    nb_backbones_data[folder][type][expr_name] = expr_backbone_count



    all_exprs = list(expr_data.all_expr_data.keys())

    writer.writerow(["Nb assigned vars"])
    writer.writerow(["expr"]+[ f.split("_")[-1] + "_" + l  for f in folders for l in labels if not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') ]+ ["number of variables"])
    for e in all_exprs:
        writer.writerow([e ]+ [ nb_assigned_vars_data[f][l][e] for f in folders for l in labels if not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static') ] + [nb_vars_data[e]]  )
    mins = []
    maxs = []
    avgs = []
    medians = []
    for f in folders:
        for l in labels:
            if ('rand_dynamic' in f or 'hybrid_wmc' in f) and l == 'static':
                continue
            all = [ 100 * nb_assigned_vars_data[f][l][e]/nb_vars_data[e] for e in all_exprs]
            print("=----------------------------", f, l, all.count(100))
            mins.append(round(min(all), 2))
            maxs.append(round(max(all), 2))
            avgs.append(round(sum(all) / len(all), 2))
            medians.append(round(statistics.median(all), 2))
    print(len(mins))
    writer.writerow(["min:"] + mins)
    writer.writerow(["max:"] + maxs)
    writer.writerow(["avg:"] + avgs)
    writer.writerow(["median:"] + medians)


    if not nocompile:
        writer.writerow(["Nb unit clauses assigned"])
        writer.writerow(["expr"] + [f.split("_")[-1] + "_" + l for f in folders for l in labels if
                                    not ( ( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')] + ["number of variables"])
        for e in all_exprs:
            writer.writerow([e] + [nb_backbones_data[f][l][e] for f in folders for l in labels if
                                   not (( 'rand_dynamic' in f or 'hybrid_wmc' in f ) and l == 'static')] + [nb_vars_data[e]])

        mins = []
        maxs = []
        avgs = []
        medians = []
        for f in folders:
            for l in labels:
                if ('rand_dynamic' in f or 'hybrid_wmc' in f) and l == 'static':
                    continue
                all = [ 100*nb_backbones_data[f][l][e]/nb_vars_data[e] for e in all_exprs]
                mins.append(round(min(all), 2))
                maxs.append(round(max(all), 2))
                avgs.append(round(sum(all) / len(all), 2))
                medians.append(round(statistics.median(all), 2))
        print(len(mins))
        writer.writerow(["min:"] + mins)
        writer.writerow(["max:"] + maxs)
        writer.writerow(["avg:"] + avgs)
        writer.writerow(["median:"] + medians)

    print(nb_assigned_vars_data)

def filer_instances():
    FOLDER = "Dataset_preproc"
    nb_vars_data = {}
    columns = [ "p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]  # for d4

    all_expr_names_count = {}
    nb_exprs = 0
    smallest_n = 600
    all_exprs = []
    # folders = ["./results_aaai/dataset_preproc_wmc/"]
    folders = ["./results_aaai/Dataset_preproc_wscore_estimate/"]
    # labels = ["static"]
    labels = ["dynamic"]
    nb_vars_data = {}
    for folder in folders:
        for type in labels:
            # if ('rand_dynamic' in folder or 'wscore_half' in folder or 'wscore_estimate' in folder) and type == 'static':
            if 'rand_dynamic' in folder and type == 'static':
                continue
            nb_exprs += 1
            stats_file = folder + "dataset_stats_" + type + ".csv"

            with (open(stats_file) as csvfile):
                prev_line_expr_name = False
                reader = csv.reader(csvfile, delimiter=',')
                for line in reader:
                    if len(line) == 1 or ".cnf" in line[0]:  # if first line or start of new expr
                        print("expr:", line)
                        expr_file = line[0].split("/")[-1]
                        save_expr_name = line[0]
                        if "_bench"  in expr_file:
                            expr_file = expr_file.replace("_bench", ".bench")
                        if save_expr_name.count(".") > 1:
                            save_expr_name = save_expr_name.replace(".", "_", save_expr_name.count( ".") - 1)  # actually first . will always be ./input so should skipp that
                        save_expr_name = save_expr_name.split("/")[-1]
                        if save_expr_name not in nb_vars_data:

                            with open('./input/Dataset_preproc/'+expr_file, "r") as f:
                                content = f.readlines()
                                nb_vars = int(content[0].strip().split(" ")[2])
                            nb_vars_data[save_expr_name] = nb_vars
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file, full_expr_only=False, min_nb_expr=-1, padding=False,
                                      filter_timeout=False, filter_conflict=False)
            print("========", folder, type, len(expr_data.all_expr_data))
            nb_assigned_vars_data = expr_data.nb_completed_assignments

    #-----------------------
    count_less300 = 0
    count_less900 = 0
    count_morethen_900 = 0
    expr_morethen_900 = []
    exprs = []
    expr_time = []
    for e, nb_var in nb_vars_data.items():
        print(e, nb_var)
        if nb_var <= 300:
            count_less300+=1
        elif nb_var > 300 and nb_var <=900:
            count_less900+=1
            exprs.append(e)
            if e in expr_data.all_expr_data:
                expr_time.append( expr_data.all_expr_data[e][-1][-4] )
            else:
                print("no init", e,nb_var)
        else:
            count_morethen_900 +=1
            if e in expr_data.all_expr_data:
                expr_morethen_900.append(e)
            else:
                print("no init", e, nb_var)
            
    print("count_less300 ",count_less300)
    print("count_less900 ",count_less900)
    print("more then 900 ",count_morethen_900)
    print(expr_morethen_900)
    # print(exprs)
    # print(len(expr_time), sum(expr_time), max(expr_time))
    exit(9)

    #------

    sorted_nb_vars_data = sorted(nb_vars_data.items(), key=lambda kv: kv[1])
    print(sorted_nb_vars_data)
    print( len(sorted_nb_vars_data))
    # f = open("./results_aaai/instances.txt","w")
    no_init_comp = []
    for item in sorted_nb_vars_data:
        assigned = -1
        if item[0] in nb_assigned_vars_data:
            assigned = nb_assigned_vars_data[ item[0] ]
        else: #no init compilation
            no_init_comp.append(item[0])
            print(item[0])
    print(no_init_comp)
    #     f.write(str(item[0])+ " "+str(item[1])+ " "+ str(assigned) +"\n")
    # f.flush()
    # f.close()
    #get unique values
    nb_vars = [item[1] for item in sorted_nb_vars_data]
    keys = sorted(list(set(nb_vars)))

    occurance =[]
    for k in keys:
        occurance.append(nb_vars.count(k))


    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # print(len(keys), len(occurance))
    # ax1.bar(keys, occurance )
    # # plt.xticks(keys)
    # plt.show()
    # print(keys)
    # print(occurance)


columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
           "obj"]  # for d4
cutoff = {'half/': {
       'static': {'01_istance_K3_N15_M45_01.cnf': 6.0, '01_istance_K3_N15_M45_02.cnf': 3.0,
                   '01_istance_K3_N15_M45_03.cnf': 10.0, '01_istance_K3_N15_M45_04.cnf': 2.0,
                   '01_istance_K3_N15_M45_05.cnf': 5.0, '01_istance_K3_N15_M45_06.cnf': 6.0,
                   '01_istance_K3_N15_M45_07.cnf': 7.0, '01_istance_K3_N15_M45_08.cnf': 7.0,
                   '01_istance_K3_N15_M45_09.cnf': 9.0, '01_istance_K3_N15_M45_10.cnf': 5.0,
                   '02_instance_K3_N30_M90_01.cnf': 10.0, '02_instance_K3_N30_M90_02.cnf': 5.0,
                   '02_instance_K3_N30_M90_03.cnf': 8.0, '02_instance_K3_N30_M90_04.cnf': 8.0,
                   '02_instance_K3_N30_M90_05.cnf': 12.0, '02_instance_K3_N30_M90_06.cnf': 13.0,
                   '02_instance_K3_N30_M90_07.cnf': 8.0, '02_instance_K3_N30_M90_08.cnf': 6.0,
                   '02_instance_K3_N30_M90_09.cnf': 12.0, '02_instance_K3_N30_M90_10.cnf': 4.0,
                   '04_iscas89_s400_bench.cnf': 6.0, '04_iscas89_s420_1_bench.cnf': 8.0,
                   '04_iscas89_s444_bench.cnf': 12.0, '04_iscas89_s526_bench.cnf': 12.0,
                   '04_iscas89_s526n_bench.cnf': 4.0, '05_iscas93_s344_bench.cnf': 7.0,
                   '05_iscas93_s499_bench.cnf': 3.0, '06_iscas99_b01.cnf': 4.0, '06_iscas99_b02.cnf': 5.0,
                   '06_iscas99_b03.cnf': 9.0, '06_iscas99_b06.cnf': 6.0, '06_iscas99_b08.cnf': 9.0,
                   '06_iscas99_b09.cnf': 8.0, '06_iscas99_b10.cnf': 3.0, '07_blocks_right_2_p_t1.cnf': 3.0,
                   '07_blocks_right_2_p_t2.cnf': 6.0, '07_blocks_right_2_p_t3.cnf': 2.0,
                   '07_blocks_right_2_p_t4.cnf': 7.0, '07_blocks_right_2_p_t5.cnf': 9.0,
                   '07_blocks_right_3_p_t1.cnf': 3.0, '07_blocks_right_3_p_t2.cnf': 3.0,
                   '07_blocks_right_4_p_t1.cnf': 2.0, '08_bomb_b10_t5_p_t1.cnf': 18.0, '08_bomb_b5_t1_p_t1.cnf': 16.0,
                   '08_bomb_b5_t1_p_t2.cnf': 6.0, '08_bomb_b5_t1_p_t3.cnf': 12.0, '08_bomb_b5_t1_p_t4.cnf': 8.0,
                   '08_bomb_b5_t1_p_t5.cnf': 10.0, '08_bomb_b5_t5_p_t1.cnf': 13.0, '08_bomb_b5_t5_p_t2.cnf': 10.0,
                   '09_coins_p01_p_t1.cnf': 13.0, '09_coins_p02_p_t1.cnf': 13.0, '09_coins_p03_p_t1.cnf': 13.0,
                   '09_coins_p04_p_t1.cnf': 13.0, '09_coins_p05_p_t1.cnf': 13.0, '09_coins_p05_p_t2.cnf': 16.0,
                   '09_coins_p10_p_t1.cnf': 13.0, '10_comm_p01_p_t1.cnf': 9.0, '10_comm_p01_p_t2.cnf': 8.0,
                   '10_comm_p02_p_t1.cnf': 12.0, '10_comm_p03_p_t1.cnf': 13.0, '11_emptyroom_d12_g6_p_t1.cnf': 4.0,
                   '11_emptyroom_d12_g6_p_t2.cnf': 3.0, '11_emptyroom_d16_g8_p_t1.cnf': 4.0,
                   '11_emptyroom_d16_g8_p_t2.cnf': 7.0, '11_emptyroom_d20_g10_corners_p_t1.cnf': 7.0,
                   '11_emptyroom_d24_g12_p_t1.cnf': 6.0, '11_emptyroom_d28_g14_corners_p_t1.cnf': 7.0,
                   '11_emptyroom_d4_g2_p_t10.cnf': 10.0, '11_emptyroom_d4_g2_p_t1.cnf': 5.0,
                   '11_emptyroom_d4_g2_p_t2.cnf': 4.0, '11_emptyroom_d4_g2_p_t3.cnf': 6.0,
                   '11_emptyroom_d4_g2_p_t4.cnf': 3.0, '11_emptyroom_d4_g2_p_t5.cnf': 4.0,
                   '11_emptyroom_d4_g2_p_t6.cnf': 10.0, '11_emptyroom_d4_g2_p_t7.cnf': 7.0,
                   '11_emptyroom_d4_g2_p_t8.cnf': 6.0, '11_emptyroom_d4_g2_p_t9.cnf': 5.0,
                   '11_emptyroom_d8_g4_p_t1.cnf': 2.0, '11_emptyroom_d8_g4_p_t2.cnf': 4.0,
                   '11_emptyroom_d8_g4_p_t3.cnf': 7.0, '11_emptyroom_d8_g4_p_t4.cnf': 9.0, '12_flip_1_p_t10.cnf': 11.0,
                   '12_flip_1_p_t1.cnf': 3.0, '12_flip_1_p_t2.cnf': 3.0, '12_flip_1_p_t3.cnf': 4.0,
                   '12_flip_1_p_t4.cnf': 6.0, '12_flip_1_p_t5.cnf': 8.0, '12_flip_1_p_t6.cnf': 10.0,
                   '12_flip_1_p_t7.cnf': 12.0, '12_flip_1_p_t8.cnf': 10.0, '12_flip_1_p_t9.cnf': 10.0,
                   '12_flip_no_action_1_p_t10.cnf': 7.0, '12_flip_no_action_1_p_t1.cnf': 3.0,
                   '12_flip_no_action_1_p_t2.cnf': 5.0, '12_flip_no_action_1_p_t3.cnf': 4.0,
                   '12_flip_no_action_1_p_t4.cnf': 8.0, '12_flip_no_action_1_p_t5.cnf': 4.0,
                   '12_flip_no_action_1_p_t6.cnf': 2.0, '12_flip_no_action_1_p_t7.cnf': 4.0,
                   '12_flip_no_action_1_p_t8.cnf': 5.0, '12_flip_no_action_1_p_t9.cnf': 6.0,
                   '13_ring2_r6_p_t1.cnf': 3.0, '13_ring2_r6_p_t2.cnf': 4.0, '13_ring2_r6_p_t3.cnf': 4.0,
                   '13_ring2_r8_p_t1.cnf': 4.0, '13_ring2_r8_p_t2.cnf': 3.0, '13_ring2_r8_p_t3.cnf': 6.0,
                   '13_ring_3_p_t1.cnf': 2.0, '13_ring_3_p_t2.cnf': 7.0, '13_ring_3_p_t3.cnf': 8.0,
                   '13_ring_3_p_t4.cnf': 16.0, '13_ring_4_p_t1.cnf': 18.0, '13_ring_4_p_t2.cnf': 7.0,
                   '13_ring_4_p_t3.cnf': 8.0, '13_ring_5_p_t1.cnf': 7.0, '13_ring_5_p_t2.cnf': 13.0,
                   '13_ring_5_p_t3.cnf': 3.0, '14_safe_safe_10_p_t10.cnf': 4.0, '14_safe_safe_10_p_t1.cnf': 7.0,
                   '14_safe_safe_10_p_t2.cnf': 2.0, '14_safe_safe_10_p_t3.cnf': 4.0, '14_safe_safe_10_p_t4.cnf': 4.0,
                   '14_safe_safe_10_p_t5.cnf': 4.0, '14_safe_safe_10_p_t6.cnf': 4.0, '14_safe_safe_10_p_t7.cnf': 7.0,
                   '14_safe_safe_10_p_t8.cnf': 7.0, '14_safe_safe_10_p_t9.cnf': 11.0, '14_safe_safe_30_p_t1.cnf': 6.0,
                   '14_safe_safe_30_p_t2.cnf': 5.0, '14_safe_safe_30_p_t3.cnf': 3.0, '14_safe_safe_30_p_t4.cnf': 4.0,
                   '14_safe_safe_30_p_t5.cnf': 5.0, '14_safe_safe_30_p_t6.cnf': 6.0, '14_safe_safe_5_p_t10.cnf': 8.0,
                   '14_safe_safe_5_p_t1.cnf': 4.0, '14_safe_safe_5_p_t2.cnf': 9.0, '14_safe_safe_5_p_t3.cnf': 4.0,
                   '14_safe_safe_5_p_t4.cnf': 2.0, '14_safe_safe_5_p_t5.cnf': 3.0, '14_safe_safe_5_p_t6.cnf': 6.0,
                   '14_safe_safe_5_p_t7.cnf': 6.0, '14_safe_safe_5_p_t8.cnf': 4.0, '14_safe_safe_5_p_t9.cnf': 8.0,
                   '15_sort_num_s_3_p_t10.cnf': 6.0, '15_sort_num_s_3_p_t1.cnf': 3.0, '15_sort_num_s_3_p_t2.cnf': 2.0,
                   '15_sort_num_s_3_p_t3.cnf': 2.0, '15_sort_num_s_3_p_t4.cnf': 9.0, '15_sort_num_s_3_p_t5.cnf': 4.0,
                   '15_sort_num_s_3_p_t6.cnf': 4.0, '15_sort_num_s_3_p_t7.cnf': 6.0, '15_sort_num_s_3_p_t8.cnf': 7.0,
                   '15_sort_num_s_3_p_t9.cnf': 8.0, '15_sort_num_s_4_p_t1.cnf': 4.0, '16_uts_k1_p_t10.cnf': 3.0,
                   '16_uts_k1_p_t1.cnf': 4.0, '16_uts_k1_p_t2.cnf': 1.0, '16_uts_k1_p_t3.cnf': 4.0,
                   '16_uts_k1_p_t4.cnf': 4.0, '16_uts_k1_p_t5.cnf': 4.0, '16_uts_k1_p_t6.cnf': 3.0,
                   '16_uts_k1_p_t7.cnf': 3.0, '16_uts_k1_p_t8.cnf': 9.0, '16_uts_k1_p_t9.cnf': 8.0,
                   '16_uts_k2_p_t1.cnf': 6.0, '16_uts_k2_p_t2.cnf': 7.0, '16_uts_k3_p_t1.cnf': 4.0},
        'dynamic': {'01_istance_K3_N15_M45_01.cnf': 15.0, '01_istance_K3_N15_M45_02.cnf': 5.0,
                    '01_istance_K3_N15_M45_03.cnf': 15.0, '01_istance_K3_N15_M45_04.cnf': 2.0,
                    '01_istance_K3_N15_M45_05.cnf': 15.0, '01_istance_K3_N15_M45_06.cnf': 15.0,
                    '01_istance_K3_N15_M45_07.cnf': 13.0, '01_istance_K3_N15_M45_08.cnf': 9.0,
                    '01_istance_K3_N15_M45_09.cnf': 15.0, '01_istance_K3_N15_M45_10.cnf': 7.0,
                    '02_instance_K3_N30_M90_01.cnf': 30.0, '02_instance_K3_N30_M90_02.cnf': 7.0,
                    '02_instance_K3_N30_M90_03.cnf': 30.0, '02_instance_K3_N30_M90_04.cnf': 24.0,
                    '02_instance_K3_N30_M90_05.cnf': 30.0, '02_instance_K3_N30_M90_06.cnf': 30.0,
                    '02_instance_K3_N30_M90_07.cnf': 30.0, '02_instance_K3_N30_M90_08.cnf': 15.0,
                    '02_instance_K3_N30_M90_09.cnf': 30.0, '02_instance_K3_N30_M90_10.cnf': 5.0,
                    '04_iscas89_s400_bench.cnf': 44.0, '04_iscas89_s420_1_bench.cnf': 252.0,
                    '04_iscas89_s444_bench.cnf': 205.0, '04_iscas89_s526_bench.cnf': 217.0,
                    '04_iscas89_s526n_bench.cnf': 218.0, '05_iscas93_s344_bench.cnf': 184.0,
                    '05_iscas93_s499_bench.cnf': 175.0, '06_iscas99_b01.cnf': 45.0, '06_iscas99_b02.cnf': 26.0,
                    '06_iscas99_b03.cnf': 156.0, '06_iscas99_b06.cnf': 44.0, '06_iscas99_b08.cnf': 46.0,
                    '06_iscas99_b09.cnf': 169.0, '06_iscas99_b10.cnf': 201.0, '07_blocks_right_2_p_t1.cnf': 90.0,
                    '07_blocks_right_2_p_t2.cnf': 169.0, '07_blocks_right_2_p_t3.cnf': 2.0,
                    '07_blocks_right_2_p_t4.cnf': 170.0, '07_blocks_right_2_p_t5.cnf': 221.0,
                    '07_blocks_right_3_p_t1.cnf': 176.0, '07_blocks_right_3_p_t2.cnf': 8.0,
                    '07_blocks_right_4_p_t1.cnf': 2.0, '08_bomb_b10_t5_p_t1.cnf': 268.0,
                    '08_bomb_b5_t1_p_t1.cnf': 120.0, '08_bomb_b5_t1_p_t2.cnf': 23.0, '08_bomb_b5_t1_p_t3.cnf': 254.0,
                    '08_bomb_b5_t1_p_t4.cnf': 241.0, '08_bomb_b5_t1_p_t5.cnf': 103.0, '08_bomb_b5_t5_p_t1.cnf': 280.0,
                    '08_bomb_b5_t5_p_t2.cnf': 51.0, '09_coins_p01_p_t1.cnf': 175.0, '09_coins_p02_p_t1.cnf': 175.0,
                    '09_coins_p03_p_t1.cnf': 175.0, '09_coins_p04_p_t1.cnf': 175.0, '09_coins_p05_p_t1.cnf': 175.0,
                    '09_coins_p05_p_t2.cnf': 194.0, '09_coins_p10_p_t1.cnf': 101.0, '10_comm_p01_p_t1.cnf': 170.0,
                    '10_comm_p01_p_t2.cnf': 145.0, '10_comm_p02_p_t1.cnf': 282.0, '10_comm_p03_p_t1.cnf': 270.0,
                    '11_emptyroom_d12_g6_p_t1.cnf': 73.0, '11_emptyroom_d12_g6_p_t2.cnf': 37.0,
                    '11_emptyroom_d16_g8_p_t1.cnf': 188.0, '11_emptyroom_d16_g8_p_t2.cnf': 173.0,
                    '11_emptyroom_d20_g10_corners_p_t1.cnf': 108.0, '11_emptyroom_d24_g12_p_t1.cnf': 284.0,
                    '11_emptyroom_d28_g14_corners_p_t1.cnf': 298.0, '11_emptyroom_d4_g2_p_t10.cnf': 228.0,
                    '11_emptyroom_d4_g2_p_t1.cnf': 25.0, '11_emptyroom_d4_g2_p_t2.cnf': 53.0,
                    '11_emptyroom_d4_g2_p_t3.cnf': 84.0, '11_emptyroom_d4_g2_p_t4.cnf': 18.0,
                    '11_emptyroom_d4_g2_p_t5.cnf': 31.0, '11_emptyroom_d4_g2_p_t6.cnf': 83.0,
                    '11_emptyroom_d4_g2_p_t7.cnf': 94.0, '11_emptyroom_d4_g2_p_t8.cnf': 45.0,
                    '11_emptyroom_d4_g2_p_t9.cnf': 105.0, '11_emptyroom_d8_g4_p_t1.cnf': 92.0,
                    '11_emptyroom_d8_g4_p_t2.cnf': 46.0, '11_emptyroom_d8_g4_p_t3.cnf': 65.0,
                    '11_emptyroom_d8_g4_p_t4.cnf': 47.0, '12_flip_1_p_t10.cnf': 29.0, '12_flip_1_p_t1.cnf': 5.0,
                    '12_flip_1_p_t2.cnf': 3.0, '12_flip_1_p_t3.cnf': 13.0, '12_flip_1_p_t4.cnf': 13.0,
                    '12_flip_1_p_t5.cnf': 21.0, '12_flip_1_p_t6.cnf': 19.0, '12_flip_1_p_t7.cnf': 29.0,
                    '12_flip_1_p_t8.cnf': 22.0, '12_flip_1_p_t9.cnf': 26.0, '12_flip_no_action_1_p_t10.cnf': 71.0,
                    '12_flip_no_action_1_p_t1.cnf': 8.0, '12_flip_no_action_1_p_t2.cnf': 15.0,
                    '12_flip_no_action_1_p_t3.cnf': 22.0, '12_flip_no_action_1_p_t4.cnf': 29.0,
                    '12_flip_no_action_1_p_t5.cnf': 36.0, '12_flip_no_action_1_p_t6.cnf': 43.0,
                    '12_flip_no_action_1_p_t7.cnf': 28.0, '12_flip_no_action_1_p_t8.cnf': 57.0,
                    '12_flip_no_action_1_p_t9.cnf': 64.0, '13_ring2_r6_p_t1.cnf': 76.0, '13_ring2_r6_p_t2.cnf': 8.0,
                    '13_ring2_r6_p_t3.cnf': 44.0, '13_ring2_r8_p_t1.cnf': 100.0, '13_ring2_r8_p_t2.cnf': 34.0,
                    '13_ring2_r8_p_t3.cnf': 58.0, '13_ring_3_p_t1.cnf': 2.0, '13_ring_3_p_t2.cnf': 116.0,
                    '13_ring_3_p_t3.cnf': 36.0, '13_ring_3_p_t4.cnf': 208.0, '13_ring_4_p_t1.cnf': 92.0,
                    '13_ring_4_p_t2.cnf': 13.0, '13_ring_4_p_t3.cnf': 212.0, '13_ring_5_p_t1.cnf': 114.0,
                    '13_ring_5_p_t2.cnf': 188.0, '13_ring_5_p_t3.cnf': 14.0, '14_safe_safe_10_p_t10.cnf': 321.0,
                    '14_safe_safe_10_p_t1.cnf': 42.0, '14_safe_safe_10_p_t2.cnf': 73.0,
                    '14_safe_safe_10_p_t3.cnf': 104.0, '14_safe_safe_10_p_t4.cnf': 135.0,
                    '14_safe_safe_10_p_t5.cnf': 166.0, '14_safe_safe_10_p_t6.cnf': 197.0,
                    '14_safe_safe_10_p_t7.cnf': 228.0, '14_safe_safe_10_p_t8.cnf': 259.0,
                    '14_safe_safe_10_p_t9.cnf': 290.0, '14_safe_safe_30_p_t1.cnf': 122.0,
                    '14_safe_safe_30_p_t2.cnf': 213.0, '14_safe_safe_30_p_t3.cnf': 304.0,
                    '14_safe_safe_30_p_t4.cnf': 395.0, '14_safe_safe_30_p_t5.cnf': 486.0,
                    '14_safe_safe_30_p_t6.cnf': 577.0, '14_safe_safe_5_p_t10.cnf': 166.0,
                    '14_safe_safe_5_p_t1.cnf': 22.0, '14_safe_safe_5_p_t2.cnf': 38.0, '14_safe_safe_5_p_t3.cnf': 54.0,
                    '14_safe_safe_5_p_t4.cnf': 70.0, '14_safe_safe_5_p_t5.cnf': 86.0, '14_safe_safe_5_p_t6.cnf': 102.0,
                    '14_safe_safe_5_p_t7.cnf': 118.0, '14_safe_safe_5_p_t8.cnf': 134.0,
                    '14_safe_safe_5_p_t9.cnf': 150.0, '15_sort_num_s_3_p_t10.cnf': 177.0,
                    '15_sort_num_s_3_p_t1.cnf': 39.0, '15_sort_num_s_3_p_t2.cnf': 18.0,
                    '15_sort_num_s_3_p_t3.cnf': 99.0, '15_sort_num_s_3_p_t4.cnf': 129.0,
                    '15_sort_num_s_3_p_t5.cnf': 100.0, '15_sort_num_s_3_p_t6.cnf': 189.0,
                    '15_sort_num_s_3_p_t7.cnf': 94.0, '15_sort_num_s_3_p_t8.cnf': 149.0,
                    '15_sort_num_s_3_p_t9.cnf': 93.0, '15_sort_num_s_4_p_t1.cnf': 110.0, '16_uts_k1_p_t10.cnf': 3.0,
                    '16_uts_k1_p_t1.cnf': 36.0, '16_uts_k1_p_t2.cnf': 1.0, '16_uts_k1_p_t3.cnf': 42.0,
                    '16_uts_k1_p_t4.cnf': 24.0, '16_uts_k1_p_t5.cnf': 49.0, '16_uts_k1_p_t6.cnf': 31.0,
                    '16_uts_k1_p_t7.cnf': 50.0, '16_uts_k1_p_t8.cnf': 176.0, '16_uts_k1_p_t9.cnf': 56.0,
                    '16_uts_k2_p_t1.cnf': 102.0, '16_uts_k2_p_t2.cnf': 191.0, '16_uts_k3_p_t1.cnf': 200.0}},
     'estimate/': {
         'static': {'01_istance_K3_N15_M45_01.cnf': 15.0, '01_istance_K3_N15_M45_02.cnf': 13.0,
                    '01_istance_K3_N15_M45_03.cnf': 15.0, '01_istance_K3_N15_M45_04.cnf': 9.0,
                    '01_istance_K3_N15_M45_05.cnf': 14.0, '01_istance_K3_N15_M45_06.cnf': 15.0,
                    '01_istance_K3_N15_M45_07.cnf': 14.0, '01_istance_K3_N15_M45_08.cnf': 12.0,
                    '01_istance_K3_N15_M45_09.cnf': 15.0, '01_istance_K3_N15_M45_10.cnf': 10.0,
                    '02_instance_K3_N30_M90_01.cnf': 22.0, '02_instance_K3_N30_M90_02.cnf': 15.0,
                    '02_instance_K3_N30_M90_03.cnf': 24.0, '02_instance_K3_N30_M90_04.cnf': 27.0,
                    '02_instance_K3_N30_M90_05.cnf': 17.0, '02_instance_K3_N30_M90_06.cnf': 17.0,
                    '02_instance_K3_N30_M90_07.cnf': 21.0, '02_instance_K3_N30_M90_08.cnf': 24.0,
                    '02_instance_K3_N30_M90_09.cnf': 18.0, '02_instance_K3_N30_M90_10.cnf': 8.0,
                    '04_iscas89_s400_bench.cnf': 36.0, '04_iscas89_s420_1_bench.cnf': 19.0,
                    '04_iscas89_s444_bench.cnf': 19.0, '04_iscas89_s526_bench.cnf': 31.0,
                    '04_iscas89_s526n_bench.cnf': 45.0, '05_iscas93_s344_bench.cnf': 9.0,
                    '05_iscas93_s499_bench.cnf': 7.0, '06_iscas99_b01.cnf': 9.0, '06_iscas99_b02.cnf': 7.0,
                    '06_iscas99_b03.cnf': 5.0, '06_iscas99_b06.cnf': 6.0, '06_iscas99_b08.cnf': 8.0,
                    '06_iscas99_b09.cnf': 8.0, '06_iscas99_b10.cnf': 18.0, '07_blocks_right_2_p_t1.cnf': 30.0,
                    '07_blocks_right_2_p_t2.cnf': 38.0, '07_blocks_right_2_p_t3.cnf': 14.0,
                    '07_blocks_right_2_p_t4.cnf': 26.0, '07_blocks_right_2_p_t5.cnf': 10.0,
                    '07_blocks_right_3_p_t1.cnf': 31.0, '07_blocks_right_3_p_t2.cnf': 26.0,
                    '07_blocks_right_4_p_t1.cnf': 68.0, '08_bomb_b10_t5_p_t1.cnf': 99.0, '08_bomb_b5_t1_p_t1.cnf': 52.0,
                    '08_bomb_b5_t1_p_t2.cnf': 52.0, '08_bomb_b5_t1_p_t3.cnf': 55.0, '08_bomb_b5_t1_p_t4.cnf': 79.0,
                    '08_bomb_b5_t1_p_t5.cnf': 158.0, '08_bomb_b5_t5_p_t1.cnf': 52.0, '08_bomb_b5_t5_p_t2.cnf': 52.0,
                    '09_coins_p01_p_t1.cnf': 22.0, '09_coins_p02_p_t1.cnf': 22.0, '09_coins_p03_p_t1.cnf': 22.0,
                    '09_coins_p04_p_t1.cnf': 22.0, '09_coins_p05_p_t1.cnf': 22.0, '09_coins_p05_p_t2.cnf': 51.0,
                    '09_coins_p10_p_t1.cnf': 117.0, '10_comm_p01_p_t1.cnf': 54.0, '10_comm_p01_p_t2.cnf': 88.0,
                    '10_comm_p02_p_t1.cnf': 52.0, '10_comm_p03_p_t1.cnf': 83.0, '11_emptyroom_d12_g6_p_t1.cnf': 39.0,
                    '11_emptyroom_d12_g6_p_t2.cnf': 37.0, '11_emptyroom_d16_g8_p_t1.cnf': 62.0,
                    '11_emptyroom_d16_g8_p_t2.cnf': 56.0, '11_emptyroom_d20_g10_corners_p_t1.cnf': 27.0,
                    '11_emptyroom_d24_g12_p_t1.cnf': 82.0, '11_emptyroom_d28_g14_corners_p_t1.cnf': 11.0,
                    '11_emptyroom_d4_g2_p_t10.cnf': 67.0, '11_emptyroom_d4_g2_p_t1.cnf': 13.0,
                    '11_emptyroom_d4_g2_p_t2.cnf': 28.0, '11_emptyroom_d4_g2_p_t3.cnf': 20.0,
                    '11_emptyroom_d4_g2_p_t4.cnf': 25.0, '11_emptyroom_d4_g2_p_t5.cnf': 25.0,
                    '11_emptyroom_d4_g2_p_t6.cnf': 29.0, '11_emptyroom_d4_g2_p_t7.cnf': 31.0,
                    '11_emptyroom_d4_g2_p_t8.cnf': 43.0, '11_emptyroom_d4_g2_p_t9.cnf': 53.0,
                    '11_emptyroom_d8_g4_p_t1.cnf': 16.0, '11_emptyroom_d8_g4_p_t2.cnf': 19.0,
                    '11_emptyroom_d8_g4_p_t3.cnf': 21.0, '11_emptyroom_d8_g4_p_t4.cnf': 44.0,
                    '12_flip_1_p_t10.cnf': 18.0, '12_flip_1_p_t1.cnf': 3.0, '12_flip_1_p_t2.cnf': 5.0,
                    '12_flip_1_p_t3.cnf': 5.0, '12_flip_1_p_t4.cnf': 15.0, '12_flip_1_p_t5.cnf': 12.0,
                    '12_flip_1_p_t6.cnf': 11.0, '12_flip_1_p_t7.cnf': 10.0, '12_flip_1_p_t8.cnf': 15.0,
                    '12_flip_1_p_t9.cnf': 16.0, '12_flip_no_action_1_p_t10.cnf': 30.0,
                    '12_flip_no_action_1_p_t1.cnf': 4.0, '12_flip_no_action_1_p_t2.cnf': 9.0,
                    '12_flip_no_action_1_p_t3.cnf': 12.0, '12_flip_no_action_1_p_t4.cnf': 18.0,
                    '12_flip_no_action_1_p_t5.cnf': 22.0, '12_flip_no_action_1_p_t6.cnf': 17.0,
                    '12_flip_no_action_1_p_t7.cnf': 21.0, '12_flip_no_action_1_p_t8.cnf': 25.0,
                    '12_flip_no_action_1_p_t9.cnf': 26.0, '13_ring2_r6_p_t1.cnf': 21.0, '13_ring2_r6_p_t2.cnf': 22.0,
                    '13_ring2_r6_p_t3.cnf': 45.0, '13_ring2_r8_p_t1.cnf': 31.0, '13_ring2_r8_p_t2.cnf': 18.0,
                    '13_ring2_r8_p_t3.cnf': 34.0, '13_ring_3_p_t1.cnf': 7.0, '13_ring_3_p_t2.cnf': 17.0,
                    '13_ring_3_p_t3.cnf': 19.0, '13_ring_3_p_t4.cnf': 23.0, '13_ring_4_p_t1.cnf': 28.0,
                    '13_ring_4_p_t2.cnf': 19.0, '13_ring_4_p_t3.cnf': 16.0, '13_ring_5_p_t1.cnf': 21.0,
                    '13_ring_5_p_t2.cnf': 21.0, '13_ring_5_p_t3.cnf': 16.0, '14_safe_safe_10_p_t10.cnf': 41.0,
                    '14_safe_safe_10_p_t1.cnf': 20.0, '14_safe_safe_10_p_t2.cnf': 13.0,
                    '14_safe_safe_10_p_t3.cnf': 19.0, '14_safe_safe_10_p_t4.cnf': 20.0,
                    '14_safe_safe_10_p_t5.cnf': 22.0, '14_safe_safe_10_p_t6.cnf': 34.0,
                    '14_safe_safe_10_p_t7.cnf': 36.0, '14_safe_safe_10_p_t8.cnf': 37.0,
                    '14_safe_safe_10_p_t9.cnf': 45.0, '14_safe_safe_30_p_t1.cnf': 32.0,
                    '14_safe_safe_30_p_t2.cnf': 30.0, '14_safe_safe_30_p_t3.cnf': 37.0,
                    '14_safe_safe_30_p_t4.cnf': 54.0, '14_safe_safe_30_p_t5.cnf': 59.0,
                    '14_safe_safe_30_p_t6.cnf': 66.0, '14_safe_safe_5_p_t10.cnf': 26.0, '14_safe_safe_5_p_t1.cnf': 11.0,
                    '14_safe_safe_5_p_t2.cnf': 13.0, '14_safe_safe_5_p_t3.cnf': 18.0, '14_safe_safe_5_p_t4.cnf': 13.0,
                    '14_safe_safe_5_p_t5.cnf': 14.0, '14_safe_safe_5_p_t6.cnf': 21.0, '14_safe_safe_5_p_t7.cnf': 22.0,
                    '14_safe_safe_5_p_t8.cnf': 21.0, '14_safe_safe_5_p_t9.cnf': 18.0, '15_sort_num_s_3_p_t10.cnf': 37.0,
                    '15_sort_num_s_3_p_t1.cnf': 11.0, '15_sort_num_s_3_p_t2.cnf': 25.0,
                    '15_sort_num_s_3_p_t3.cnf': 19.0, '15_sort_num_s_3_p_t4.cnf': 31.0,
                    '15_sort_num_s_3_p_t5.cnf': 31.0, '15_sort_num_s_3_p_t6.cnf': 31.0,
                    '15_sort_num_s_3_p_t7.cnf': 38.0, '15_sort_num_s_3_p_t8.cnf': 46.0,
                    '15_sort_num_s_3_p_t9.cnf': 68.0, '15_sort_num_s_4_p_t1.cnf': 19.0, '16_uts_k1_p_t10.cnf': 10.0,
                    '16_uts_k1_p_t1.cnf': 20.0, '16_uts_k1_p_t2.cnf': 10.0, '16_uts_k1_p_t3.cnf': 5.0,
                    '16_uts_k1_p_t4.cnf': 19.0, '16_uts_k1_p_t5.cnf': 28.0, '16_uts_k1_p_t6.cnf': 43.0,
                    '16_uts_k1_p_t7.cnf': 61.0, '16_uts_k1_p_t8.cnf': 54.0, '16_uts_k1_p_t9.cnf': 19.0,
                    '16_uts_k2_p_t1.cnf': 46.0, '16_uts_k2_p_t2.cnf': 31.0, '16_uts_k3_p_t1.cnf': 95.0},
         'dynamic': {'01_istance_K3_N15_M45_01.cnf': 15.0, '01_istance_K3_N15_M45_02.cnf': 15.0,
                     '01_istance_K3_N15_M45_03.cnf': 15.0, '01_istance_K3_N15_M45_04.cnf': 15.0,
                     '01_istance_K3_N15_M45_05.cnf': 11.0, '01_istance_K3_N15_M45_06.cnf': 15.0,
                     '01_istance_K3_N15_M45_07.cnf': 15.0, '01_istance_K3_N15_M45_08.cnf': 15.0,
                     '01_istance_K3_N15_M45_09.cnf': 15.0, '01_istance_K3_N15_M45_10.cnf': 15.0,
                     '02_instance_K3_N30_M90_01.cnf': 30.0, '02_instance_K3_N30_M90_02.cnf': 15.0,
                     '02_instance_K3_N30_M90_03.cnf': 30.0, '02_instance_K3_N30_M90_04.cnf': 30.0,
                     '02_instance_K3_N30_M90_05.cnf': 30.0, '02_instance_K3_N30_M90_06.cnf': 30.0,
                     '02_instance_K3_N30_M90_07.cnf': 30.0, '02_instance_K3_N30_M90_08.cnf': 30.0,
                     '02_instance_K3_N30_M90_09.cnf': 30.0, '02_instance_K3_N30_M90_10.cnf': 30.0,
                     '04_iscas89_s400_bench.cnf': 189.0, '04_iscas89_s420_1_bench.cnf': 252.0,
                     '04_iscas89_s444_bench.cnf': 57.0, '04_iscas89_s526_bench.cnf': 99.0,
                     '04_iscas89_s526n_bench.cnf': 218.0, '05_iscas93_s344_bench.cnf': 184.0,
                     '05_iscas93_s499_bench.cnf': 18.0, '06_iscas99_b01.cnf': 45.0, '06_iscas99_b02.cnf': 26.0,
                     '06_iscas99_b03.cnf': 156.0, '06_iscas99_b06.cnf': 44.0, '06_iscas99_b08.cnf': 180.0,
                     '06_iscas99_b09.cnf': 169.0, '06_iscas99_b10.cnf': 18.0, '07_blocks_right_2_p_t1.cnf': 90.0,
                     '07_blocks_right_2_p_t2.cnf': 169.0, '07_blocks_right_2_p_t3.cnf': 102.0,
                     '07_blocks_right_2_p_t4.cnf': 52.0, '07_blocks_right_2_p_t5.cnf': 20.0,
                     '07_blocks_right_3_p_t1.cnf': 176.0, '07_blocks_right_3_p_t2.cnf': 240.0,
                     '07_blocks_right_4_p_t1.cnf': 290.0, '08_bomb_b10_t5_p_t1.cnf': 570.0,
                     '08_bomb_b5_t1_p_t1.cnf': 120.0, '08_bomb_b5_t1_p_t2.cnf': 222.0, '08_bomb_b5_t1_p_t3.cnf': 150.0,
                     '08_bomb_b5_t1_p_t4.cnf': 426.0, '08_bomb_b5_t1_p_t5.cnf': 528.0, '08_bomb_b5_t5_p_t1.cnf': 280.0,
                     '08_bomb_b5_t5_p_t2.cnf': 530.0, '09_coins_p01_p_t1.cnf': 175.0, '09_coins_p02_p_t1.cnf': 175.0,
                     '09_coins_p03_p_t1.cnf': 175.0, '09_coins_p04_p_t1.cnf': 175.0, '09_coins_p05_p_t1.cnf': 175.0,
                     '09_coins_p05_p_t2.cnf': 178.0, '09_coins_p10_p_t1.cnf': 349.0, '10_comm_p01_p_t1.cnf': 170.0,
                     '10_comm_p01_p_t2.cnf': 299.0, '10_comm_p02_p_t1.cnf': 282.0, '10_comm_p03_p_t1.cnf': 418.0,
                     '11_emptyroom_d12_g6_p_t1.cnf': 140.0, '11_emptyroom_d12_g6_p_t2.cnf': 256.0,
                     '11_emptyroom_d16_g8_p_t1.cnf': 188.0, '11_emptyroom_d16_g8_p_t2.cnf': 344.0,
                     '11_emptyroom_d20_g10_corners_p_t1.cnf': 236.0, '11_emptyroom_d24_g12_p_t1.cnf': 214.0,
                     '11_emptyroom_d28_g14_corners_p_t1.cnf': 332.0, '11_emptyroom_d4_g2_p_t10.cnf': 78.0,
                     '11_emptyroom_d4_g2_p_t1.cnf': 44.0, '11_emptyroom_d4_g2_p_t2.cnf': 80.0,
                     '11_emptyroom_d4_g2_p_t3.cnf': 116.0, '11_emptyroom_d4_g2_p_t4.cnf': 118.0,
                     '11_emptyroom_d4_g2_p_t5.cnf': 54.0, '11_emptyroom_d4_g2_p_t6.cnf': 65.0,
                     '11_emptyroom_d4_g2_p_t7.cnf': 67.0, '11_emptyroom_d4_g2_p_t8.cnf': 253.0,
                     '11_emptyroom_d4_g2_p_t9.cnf': 78.0, '11_emptyroom_d8_g4_p_t1.cnf': 92.0,
                     '11_emptyroom_d8_g4_p_t2.cnf': 168.0, '11_emptyroom_d8_g4_p_t3.cnf': 244.0,
                     '11_emptyroom_d8_g4_p_t4.cnf': 206.0, '12_flip_1_p_t10.cnf': 41.0, '12_flip_1_p_t1.cnf': 5.0,
                     '12_flip_1_p_t2.cnf': 9.0, '12_flip_1_p_t3.cnf': 13.0, '12_flip_1_p_t4.cnf': 17.0,
                     '12_flip_1_p_t5.cnf': 21.0, '12_flip_1_p_t6.cnf': 25.0, '12_flip_1_p_t7.cnf': 29.0,
                     '12_flip_1_p_t8.cnf': 33.0, '12_flip_1_p_t9.cnf': 37.0, '12_flip_no_action_1_p_t10.cnf': 71.0,
                     '12_flip_no_action_1_p_t1.cnf': 8.0, '12_flip_no_action_1_p_t2.cnf': 15.0,
                     '12_flip_no_action_1_p_t3.cnf': 22.0, '12_flip_no_action_1_p_t4.cnf': 29.0,
                     '12_flip_no_action_1_p_t5.cnf': 36.0, '12_flip_no_action_1_p_t6.cnf': 30.0,
                     '12_flip_no_action_1_p_t7.cnf': 50.0, '12_flip_no_action_1_p_t8.cnf': 52.0,
                     '12_flip_no_action_1_p_t9.cnf': 55.0, '13_ring2_r6_p_t1.cnf': 76.0, '13_ring2_r6_p_t2.cnf': 134.0,
                     '13_ring2_r6_p_t3.cnf': 192.0, '13_ring2_r8_p_t1.cnf': 100.0, '13_ring2_r8_p_t2.cnf': 63.0,
                     '13_ring2_r8_p_t3.cnf': 136.0, '13_ring_3_p_t1.cnf': 70.0, '13_ring_3_p_t2.cnf': 58.0,
                     '13_ring_3_p_t3.cnf': 162.0, '13_ring_3_p_t4.cnf': 208.0, '13_ring_4_p_t1.cnf': 92.0,
                     '13_ring_4_p_t2.cnf': 85.0, '13_ring_4_p_t3.cnf': 26.0, '13_ring_5_p_t1.cnf': 114.0,
                     '13_ring_5_p_t2.cnf': 188.0, '13_ring_5_p_t3.cnf': 45.0, '14_safe_safe_10_p_t10.cnf': 321.0,
                     '14_safe_safe_10_p_t1.cnf': 42.0, '14_safe_safe_10_p_t2.cnf': 73.0,
                     '14_safe_safe_10_p_t3.cnf': 104.0, '14_safe_safe_10_p_t4.cnf': 135.0,
                     '14_safe_safe_10_p_t5.cnf': 166.0, '14_safe_safe_10_p_t6.cnf': 197.0,
                     '14_safe_safe_10_p_t7.cnf': 228.0, '14_safe_safe_10_p_t8.cnf': 259.0,
                     '14_safe_safe_10_p_t9.cnf': 290.0, '14_safe_safe_30_p_t1.cnf': 122.0,
                     '14_safe_safe_30_p_t2.cnf': 213.0, '14_safe_safe_30_p_t3.cnf': 304.0,
                     '14_safe_safe_30_p_t4.cnf': 395.0, '14_safe_safe_30_p_t5.cnf': 486.0,
                     '14_safe_safe_30_p_t6.cnf': 577.0, '14_safe_safe_5_p_t10.cnf': 166.0,
                     '14_safe_safe_5_p_t1.cnf': 22.0, '14_safe_safe_5_p_t2.cnf': 38.0, '14_safe_safe_5_p_t3.cnf': 54.0,
                     '14_safe_safe_5_p_t4.cnf': 70.0, '14_safe_safe_5_p_t5.cnf': 86.0, '14_safe_safe_5_p_t6.cnf': 102.0,
                     '14_safe_safe_5_p_t7.cnf': 118.0, '14_safe_safe_5_p_t8.cnf': 134.0,
                     '14_safe_safe_5_p_t9.cnf': 150.0, '15_sort_num_s_3_p_t10.cnf': 309.0,
                     '15_sort_num_s_3_p_t1.cnf': 39.0, '15_sort_num_s_3_p_t2.cnf': 69.0,
                     '15_sort_num_s_3_p_t3.cnf': 99.0, '15_sort_num_s_3_p_t4.cnf': 129.0,
                     '15_sort_num_s_3_p_t5.cnf': 159.0, '15_sort_num_s_3_p_t6.cnf': 189.0,
                     '15_sort_num_s_3_p_t7.cnf': 219.0, '15_sort_num_s_3_p_t8.cnf': 249.0,
                     '15_sort_num_s_3_p_t9.cnf': 190.0, '15_sort_num_s_4_p_t1.cnf': 110.0, '16_uts_k1_p_t10.cnf': 26.0,
                     '16_uts_k1_p_t1.cnf': 36.0, '16_uts_k1_p_t2.cnf': 65.0, '16_uts_k1_p_t3.cnf': 8.0,
                     '16_uts_k1_p_t4.cnf': 123.0, '16_uts_k1_p_t5.cnf': 62.0, '16_uts_k1_p_t6.cnf': 153.0,
                     '16_uts_k1_p_t7.cnf': 152.0, '16_uts_k1_p_t8.cnf': 167.0, '16_uts_k1_p_t9.cnf': 67.0,
                     '16_uts_k2_p_t1.cnf': 102.0, '16_uts_k2_p_t2.cnf': 54.0, '16_uts_k3_p_t1.cnf': 168.0}},
     'dynamic/': {
         'dynamic': {'01_istance_K3_N15_M45_01.cnf': 3.0, '01_istance_K3_N15_M45_02.cnf': 15.0,
                     '01_istance_K3_N15_M45_03.cnf': 15.0, '01_istance_K3_N15_M45_04.cnf': 15.0,
                     '01_istance_K3_N15_M45_05.cnf': 15.0, '01_istance_K3_N15_M45_06.cnf': 15.0,
                     '01_istance_K3_N15_M45_07.cnf': 5.0, '01_istance_K3_N15_M45_08.cnf': 15.0,
                     '01_istance_K3_N15_M45_09.cnf': 6.0, '01_istance_K3_N15_M45_10.cnf': 15.0,
                     '02_instance_K3_N30_M90_01.cnf': 30.0, '02_instance_K3_N30_M90_02.cnf': 30.0,
                     '02_instance_K3_N30_M90_03.cnf': 30.0, '02_instance_K3_N30_M90_04.cnf': 30.0,
                     '02_instance_K3_N30_M90_05.cnf': 23.0, '02_instance_K3_N30_M90_06.cnf': 30.0,
                     '02_instance_K3_N30_M90_07.cnf': 19.0, '02_instance_K3_N30_M90_08.cnf': 30.0,
                     '02_instance_K3_N30_M90_09.cnf': 25.0, '02_instance_K3_N30_M90_10.cnf': 10.0,
                     '04_iscas89_s400_bench.cnf': 157.0, '04_iscas89_s420_1_bench.cnf': 5.0,
                     '04_iscas89_s444_bench.cnf': 205.0, '04_iscas89_s526_bench.cnf': 217.0,
                     '04_iscas89_s526n_bench.cnf': 218.0, '05_iscas93_s344_bench.cnf': 184.0,
                     '05_iscas93_s499_bench.cnf': 175.0, '06_iscas99_b01.cnf': 45.0, '06_iscas99_b02.cnf': 8.0,
                     '06_iscas99_b03.cnf': 40.0, '06_iscas99_b06.cnf': 44.0, '06_iscas99_b08.cnf': 180.0,
                     '06_iscas99_b09.cnf': 169.0, '06_iscas99_b10.cnf': 201.0, '07_blocks_right_2_p_t1.cnf': 90.0,
                     '07_blocks_right_2_p_t2.cnf': 169.0, '07_blocks_right_2_p_t3.cnf': 8.0,
                     '07_blocks_right_2_p_t4.cnf': 231.0, '07_blocks_right_2_p_t5.cnf': 7.0,
                     '07_blocks_right_3_p_t1.cnf': 1.0, '07_blocks_right_3_p_t2.cnf': 333.0,
                     '07_blocks_right_4_p_t1.cnf': 290.0, '08_bomb_b10_t5_p_t1.cnf': 570.0,
                     '08_bomb_b5_t1_p_t1.cnf': 120.0, '08_bomb_b5_t1_p_t2.cnf': 142.0, '08_bomb_b5_t1_p_t3.cnf': 238.0,
                     '08_bomb_b5_t1_p_t4.cnf': 161.0, '08_bomb_b5_t1_p_t5.cnf': 209.0, '08_bomb_b5_t5_p_t1.cnf': 280.0,
                     '08_bomb_b5_t5_p_t2.cnf': 303.0, '09_coins_p01_p_t1.cnf': 143.0, '09_coins_p02_p_t1.cnf': 143.0,
                     '09_coins_p03_p_t1.cnf': 158.0, '09_coins_p04_p_t1.cnf': 143.0, '09_coins_p05_p_t1.cnf': 124.0,
                     '09_coins_p05_p_t2.cnf': 323.0, '09_coins_p10_p_t1.cnf': 75.0, '10_comm_p01_p_t1.cnf': 104.0,
                     '10_comm_p01_p_t2.cnf': 299.0, '10_comm_p02_p_t1.cnf': 57.0, '10_comm_p03_p_t1.cnf': 121.0,
                     '11_emptyroom_d12_g6_p_t1.cnf': 2.0, '11_emptyroom_d12_g6_p_t2.cnf': 121.0,
                     '11_emptyroom_d16_g8_p_t1.cnf': 188.0, '11_emptyroom_d16_g8_p_t2.cnf': 44.0,
                     '11_emptyroom_d20_g10_corners_p_t1.cnf': 53.0, '11_emptyroom_d24_g12_p_t1.cnf': 284.0,
                     '11_emptyroom_d28_g14_corners_p_t1.cnf': 332.0, '11_emptyroom_d4_g2_p_t10.cnf': 217.0,
                     '11_emptyroom_d4_g2_p_t1.cnf': 26.0, '11_emptyroom_d4_g2_p_t2.cnf': 80.0,
                     '11_emptyroom_d4_g2_p_t3.cnf': 92.0, '11_emptyroom_d4_g2_p_t4.cnf': 63.0,
                     '11_emptyroom_d4_g2_p_t5.cnf': 80.0, '11_emptyroom_d4_g2_p_t6.cnf': 71.0,
                     '11_emptyroom_d4_g2_p_t7.cnf': 92.0, '11_emptyroom_d4_g2_p_t8.cnf': 88.0,
                     '11_emptyroom_d4_g2_p_t9.cnf': 146.0, '11_emptyroom_d8_g4_p_t1.cnf': 92.0,
                     '11_emptyroom_d8_g4_p_t2.cnf': 90.0, '11_emptyroom_d8_g4_p_t3.cnf': 102.0,
                     '11_emptyroom_d8_g4_p_t4.cnf': 60.0, '12_flip_1_p_t10.cnf': 41.0, '12_flip_1_p_t1.cnf': 5.0,
                     '12_flip_1_p_t2.cnf': 9.0, '12_flip_1_p_t3.cnf': 13.0, '12_flip_1_p_t4.cnf': 17.0,
                     '12_flip_1_p_t5.cnf': 21.0, '12_flip_1_p_t6.cnf': 25.0, '12_flip_1_p_t7.cnf': 29.0,
                     '12_flip_1_p_t8.cnf': 33.0, '12_flip_1_p_t9.cnf': 37.0, '12_flip_no_action_1_p_t10.cnf': 51.0,
                     '12_flip_no_action_1_p_t1.cnf': 8.0, '12_flip_no_action_1_p_t2.cnf': 15.0,
                     '12_flip_no_action_1_p_t3.cnf': 22.0, '12_flip_no_action_1_p_t4.cnf': 29.0,
                     '12_flip_no_action_1_p_t5.cnf': 36.0, '12_flip_no_action_1_p_t6.cnf': 43.0,
                     '12_flip_no_action_1_p_t7.cnf': 50.0, '12_flip_no_action_1_p_t8.cnf': 57.0,
                     '12_flip_no_action_1_p_t9.cnf': 64.0, '13_ring2_r6_p_t1.cnf': 2.0, '13_ring2_r6_p_t2.cnf': 7.0,
                     '13_ring2_r6_p_t3.cnf': 36.0, '13_ring2_r8_p_t1.cnf': 100.0, '13_ring2_r8_p_t2.cnf': 34.0,
                     '13_ring2_r8_p_t3.cnf': 47.0, '13_ring_3_p_t1.cnf': 70.0, '13_ring_3_p_t2.cnf': 116.0,
                     '13_ring_3_p_t3.cnf': 74.0, '13_ring_3_p_t4.cnf': 208.0, '13_ring_4_p_t1.cnf': 92.0,
                     '13_ring_4_p_t2.cnf': 152.0, '13_ring_4_p_t3.cnf': 10.0, '13_ring_5_p_t1.cnf': 114.0,
                     '13_ring_5_p_t2.cnf': 188.0, '13_ring_5_p_t3.cnf': 262.0, '14_safe_safe_10_p_t10.cnf': 321.0,
                     '14_safe_safe_10_p_t1.cnf': 42.0, '14_safe_safe_10_p_t2.cnf': 73.0,
                     '14_safe_safe_10_p_t3.cnf': 104.0, '14_safe_safe_10_p_t4.cnf': 135.0,
                     '14_safe_safe_10_p_t5.cnf': 166.0, '14_safe_safe_10_p_t6.cnf': 197.0,
                     '14_safe_safe_10_p_t7.cnf': 228.0, '14_safe_safe_10_p_t8.cnf': 259.0,
                     '14_safe_safe_10_p_t9.cnf': 290.0, '14_safe_safe_30_p_t1.cnf': 122.0,
                     '14_safe_safe_30_p_t2.cnf': 213.0, '14_safe_safe_30_p_t3.cnf': 304.0,
                     '14_safe_safe_30_p_t4.cnf': 395.0, '14_safe_safe_30_p_t5.cnf': 486.0,
                     '14_safe_safe_30_p_t6.cnf': 577.0, '14_safe_safe_5_p_t10.cnf': 166.0,
                     '14_safe_safe_5_p_t1.cnf': 22.0, '14_safe_safe_5_p_t2.cnf': 38.0, '14_safe_safe_5_p_t3.cnf': 54.0,
                     '14_safe_safe_5_p_t4.cnf': 70.0, '14_safe_safe_5_p_t5.cnf': 86.0, '14_safe_safe_5_p_t6.cnf': 102.0,
                     '14_safe_safe_5_p_t7.cnf': 118.0, '14_safe_safe_5_p_t8.cnf': 134.0,
                     '14_safe_safe_5_p_t9.cnf': 150.0, '15_sort_num_s_3_p_t10.cnf': 309.0,
                     '15_sort_num_s_3_p_t1.cnf': 39.0, '15_sort_num_s_3_p_t2.cnf': 5.0,
                     '15_sort_num_s_3_p_t3.cnf': 99.0, '15_sort_num_s_3_p_t4.cnf': 103.0,
                     '15_sort_num_s_3_p_t5.cnf': 159.0, '15_sort_num_s_3_p_t6.cnf': 110.0,
                     '15_sort_num_s_3_p_t7.cnf': 35.0, '15_sort_num_s_3_p_t8.cnf': 166.0,
                     '15_sort_num_s_3_p_t9.cnf': 133.0, '15_sort_num_s_4_p_t1.cnf': 16.0, '16_uts_k1_p_t10.cnf': 73.0,
                     '16_uts_k1_p_t1.cnf': 4.0, '16_uts_k1_p_t2.cnf': 65.0, '16_uts_k1_p_t3.cnf': 55.0,
                     '16_uts_k1_p_t4.cnf': 48.0, '16_uts_k1_p_t5.cnf': 5.0, '16_uts_k1_p_t6.cnf': 1.0,
                     '16_uts_k1_p_t7.cnf': 25.0, '16_uts_k1_p_t8.cnf': 102.0, '16_uts_k1_p_t9.cnf': 2.0,
                     '16_uts_k2_p_t1.cnf': 102.0, '16_uts_k2_p_t2.cnf': 37.0, '16_uts_k3_p_t1.cnf': 200.0}},
    'WMC/':{'static': {'01_istance_K3_N15_M45_01.cnf': 15.0, '01_istance_K3_N15_M45_02.cnf': 13.0, '01_istance_K3_N15_M45_03.cnf': 15.0, '01_istance_K3_N15_M45_04.cnf': 15.0, '01_istance_K3_N15_M45_05.cnf': 15.0, '01_istance_K3_N15_M45_06.cnf': 15.0, '01_istance_K3_N15_M45_07.cnf': 15.0, '01_istance_K3_N15_M45_08.cnf': 15.0, '01_istance_K3_N15_M45_09.cnf': 15.0, '01_istance_K3_N15_M45_10.cnf': 15.0, '02_instance_K3_N30_M90_01.cnf': 30.0, '02_instance_K3_N30_M90_02.cnf': 30.0, '02_instance_K3_N30_M90_03.cnf': 30.0, '02_instance_K3_N30_M90_04.cnf': 30.0, '02_instance_K3_N30_M90_05.cnf': 30.0, '02_instance_K3_N30_M90_06.cnf': 30.0, '02_instance_K3_N30_M90_07.cnf': 30.0, '02_instance_K3_N30_M90_08.cnf': 30.0, '02_instance_K3_N30_M90_09.cnf': 30.0, '02_instance_K3_N30_M90_10.cnf': 30.0, '04_iscas89_s400_bench.cnf': 189.0, '04_iscas89_s420_1_bench.cnf': 251.0, '04_iscas89_s444_bench.cnf': 205.0, '04_iscas89_s526_bench.cnf': 217.0, '04_iscas89_s526n_bench.cnf': 202.0, '05_iscas93_s344_bench.cnf': 184.0, '05_iscas93_s499_bench.cnf': 173.0, '06_iscas99_b01.cnf': 30.0, '06_iscas99_b02.cnf': 26.0, '06_iscas99_b03.cnf': 152.0, '06_iscas99_b06.cnf': 44.0, '06_iscas99_b08.cnf': 180.0, '06_iscas99_b09.cnf': 169.0, '06_iscas99_b10.cnf': 142.0, '07_blocks_right_2_p_t1.cnf': 83.0, '07_blocks_right_2_p_t2.cnf': 169.0, '07_blocks_right_2_p_t3.cnf': 248.0, '07_blocks_right_2_p_t4.cnf': 326.0, '07_blocks_right_2_p_t5.cnf': 406.0, '07_blocks_right_3_p_t1.cnf': 176.0, '07_blocks_right_3_p_t2.cnf': 324.0, '07_blocks_right_4_p_t1.cnf': 290.0, '08_bomb_b10_t5_p_t1.cnf': 512.0, '08_bomb_b5_t1_p_t1.cnf': 109.0, '08_bomb_b5_t1_p_t2.cnf': 178.0, '08_bomb_b5_t1_p_t3.cnf': 324.0, '08_bomb_b5_t1_p_t4.cnf': 408.0, '08_bomb_b5_t1_p_t5.cnf': 528.0, '08_bomb_b5_t5_p_t1.cnf': 271.0, '08_bomb_b5_t5_p_t2.cnf': 515.0, '09_coins_p01_p_t1.cnf': 175.0, '09_coins_p02_p_t1.cnf': 175.0, '09_coins_p03_p_t1.cnf': 175.0, '09_coins_p04_p_t1.cnf': 175.0, '09_coins_p05_p_t1.cnf': 171.0, '09_coins_p05_p_t2.cnf': 323.0, '09_coins_p10_p_t1.cnf': 354.0, '10_comm_p01_p_t1.cnf': 165.0, '10_comm_p01_p_t2.cnf': 289.0, '10_comm_p02_p_t1.cnf': 277.0, '10_comm_p03_p_t1.cnf': 414.0, '11_emptyroom_d12_g6_p_t1.cnf': 140.0, '11_emptyroom_d12_g6_p_t2.cnf': 247.0, '11_emptyroom_d16_g8_p_t1.cnf': 180.0, '11_emptyroom_d16_g8_p_t2.cnf': 344.0, '11_emptyroom_d20_g10_corners_p_t1.cnf': 236.0, '11_emptyroom_d24_g12_p_t1.cnf': 281.0, '11_emptyroom_d28_g14_corners_p_t1.cnf': 297.0, '11_emptyroom_d4_g2_p_t10.cnf': 321.0, '11_emptyroom_d4_g2_p_t1.cnf': 44.0, '11_emptyroom_d4_g2_p_t2.cnf': 80.0, '11_emptyroom_d4_g2_p_t3.cnf': 116.0, '11_emptyroom_d4_g2_p_t4.cnf': 152.0, '11_emptyroom_d4_g2_p_t5.cnf': 180.0, '11_emptyroom_d4_g2_p_t6.cnf': 224.0, '11_emptyroom_d4_g2_p_t7.cnf': 251.0, '11_emptyroom_d4_g2_p_t8.cnf': 249.0, '11_emptyroom_d4_g2_p_t9.cnf': 304.0, '11_emptyroom_d8_g4_p_t1.cnf': 92.0, '11_emptyroom_d8_g4_p_t2.cnf': 158.0, '11_emptyroom_d8_g4_p_t3.cnf': 235.0, '11_emptyroom_d8_g4_p_t4.cnf': 296.0, '12_flip_1_p_t10.cnf': 36.0, '12_flip_1_p_t1.cnf': 5.0, '12_flip_1_p_t2.cnf': 9.0, '12_flip_1_p_t3.cnf': 13.0, '12_flip_1_p_t4.cnf': 17.0, '12_flip_1_p_t5.cnf': 21.0, '12_flip_1_p_t6.cnf': 25.0, '12_flip_1_p_t7.cnf': 27.0, '12_flip_1_p_t8.cnf': 29.0, '12_flip_1_p_t9.cnf': 32.0, '12_flip_no_action_1_p_t10.cnf': 57.0, '12_flip_no_action_1_p_t1.cnf': 6.0, '12_flip_no_action_1_p_t2.cnf': 15.0, '12_flip_no_action_1_p_t3.cnf': 22.0, '12_flip_no_action_1_p_t4.cnf': 29.0, '12_flip_no_action_1_p_t5.cnf': 36.0, '12_flip_no_action_1_p_t6.cnf': 41.0, '12_flip_no_action_1_p_t7.cnf': 42.0, '12_flip_no_action_1_p_t8.cnf': 57.0, '12_flip_no_action_1_p_t9.cnf': 52.0, '13_ring2_r6_p_t1.cnf': 64.0, '13_ring2_r6_p_t2.cnf': 129.0, '13_ring2_r6_p_t3.cnf': 181.0, '13_ring2_r8_p_t1.cnf': 100.0, '13_ring2_r8_p_t2.cnf': 163.0, '13_ring2_r8_p_t3.cnf': 252.0, '13_ring_3_p_t1.cnf': 70.0, '13_ring_3_p_t2.cnf': 71.0, '13_ring_3_p_t3.cnf': 134.0, '13_ring_3_p_t4.cnf': 195.0, '13_ring_4_p_t1.cnf': 92.0, '13_ring_4_p_t2.cnf': 136.0, '13_ring_4_p_t3.cnf': 211.0, '13_ring_5_p_t1.cnf': 111.0, '13_ring_5_p_t2.cnf': 161.0, '13_ring_5_p_t3.cnf': 247.0, '14_safe_safe_10_p_t10.cnf': 306.0, '14_safe_safe_10_p_t1.cnf': 37.0, '14_safe_safe_10_p_t2.cnf': 65.0, '14_safe_safe_10_p_t3.cnf': 93.0, '14_safe_safe_10_p_t4.cnf': 129.0, '14_safe_safe_10_p_t5.cnf': 146.0, '14_safe_safe_10_p_t6.cnf': 197.0, '14_safe_safe_10_p_t7.cnf': 199.0, '14_safe_safe_10_p_t8.cnf': 256.0, '14_safe_safe_10_p_t9.cnf': 290.0, '14_safe_safe_30_p_t1.cnf': 115.0, '14_safe_safe_30_p_t2.cnf': 213.0, '14_safe_safe_30_p_t3.cnf': 283.0, '14_safe_safe_30_p_t4.cnf': 395.0, '14_safe_safe_30_p_t5.cnf': 486.0, '14_safe_safe_30_p_t6.cnf': 577.0, '14_safe_safe_5_p_t10.cnf': 166.0, '14_safe_safe_5_p_t1.cnf': 22.0, '14_safe_safe_5_p_t2.cnf': 38.0, '14_safe_safe_5_p_t3.cnf': 38.0, '14_safe_safe_5_p_t4.cnf': 70.0, '14_safe_safe_5_p_t5.cnf': 86.0, '14_safe_safe_5_p_t6.cnf': 102.0, '14_safe_safe_5_p_t7.cnf': 118.0, '14_safe_safe_5_p_t8.cnf': 134.0, '14_safe_safe_5_p_t9.cnf': 150.0, '15_sort_num_s_3_p_t10.cnf': 309.0, '15_sort_num_s_3_p_t1.cnf': 39.0, '15_sort_num_s_3_p_t2.cnf': 69.0, '15_sort_num_s_3_p_t3.cnf': 99.0, '15_sort_num_s_3_p_t4.cnf': 119.0, '15_sort_num_s_3_p_t5.cnf': 159.0, '15_sort_num_s_3_p_t6.cnf': 178.0, '15_sort_num_s_3_p_t7.cnf': 216.0, '15_sort_num_s_3_p_t8.cnf': 243.0, '15_sort_num_s_3_p_t9.cnf': 255.0, '15_sort_num_s_4_p_t1.cnf': 85.0, '16_uts_k1_p_t10.cnf': 297.0, '16_uts_k1_p_t1.cnf': 36.0, '16_uts_k1_p_t2.cnf': 65.0, '16_uts_k1_p_t3.cnf': 94.0, '16_uts_k1_p_t4.cnf': 120.0, '16_uts_k1_p_t5.cnf': 152.0, '16_uts_k1_p_t6.cnf': 181.0, '16_uts_k1_p_t7.cnf': 210.0, '16_uts_k1_p_t8.cnf': 239.0, '16_uts_k1_p_t9.cnf': 268.0, '16_uts_k2_p_t1.cnf': 102.0, '16_uts_k2_p_t2.cnf': 191.0, '16_uts_k3_p_t1.cnf': 190.0}
}
}

ecai23 = ['01_istance_K3_N15_M45_01.cnf', '01_istance_K3_N15_M45_02.cnf', '01_istance_K3_N15_M45_03.cnf',
              '01_istance_K3_N15_M45_04.cnf', '01_istance_K3_N15_M45_05.cnf', '01_istance_K3_N15_M45_06.cnf',
              '01_istance_K3_N15_M45_07.cnf', '01_istance_K3_N15_M45_08.cnf', '01_istance_K3_N15_M45_09.cnf',
              '01_istance_K3_N15_M45_10.cnf', '02_instance_K3_N30_M90_01.cnf',
              '02_instance_K3_N30_M90_02.cnf', '02_instance_K3_N30_M90_03.cnf',
              '02_instance_K3_N30_M90_04.cnf', '02_instance_K3_N30_M90_05.cnf',
              '02_instance_K3_N30_M90_06.cnf', '02_instance_K3_N30_M90_07.cnf',
              '02_instance_K3_N30_M90_08.cnf', '02_instance_K3_N30_M90_09.cnf',
              '02_instance_K3_N30_M90_10.cnf', '04_iscas89_s400_bench.cnf', '04_iscas89_s420_1_bench.cnf',
              '04_iscas89_s444_bench.cnf',
              '04_iscas89_s526_bench.cnf', '04_iscas89_s526n_bench.cnf', '05_iscas93_s344_bench.cnf',
              '05_iscas93_s499_bench.cnf', '06_iscas99_b01.cnf', '06_iscas99_b02.cnf', '06_iscas99_b03.cnf',
              '06_iscas99_b06.cnf',
              '06_iscas99_b08.cnf', '06_iscas99_b09.cnf', '06_iscas99_b10.cnf', "07_blocks_right_2_p_t1.cnf",
              "07_blocks_right_2_p_t1.cnf", "07_blocks_right_2_p_t2.cnf", "07_blocks_right_2_p_t3.cnf",
              "07_blocks_right_2_p_t4.cnf", "07_blocks_right_2_p_t5.cnf", "07_blocks_right_3_p_t1.cnf",
              "07_blocks_right_3_p_t2.cnf", "07_blocks_right_4_p_t1.cnf", "08_bomb_b10_t5_p_t1.cnf",
              "08_bomb_b5_t1_p_t1.cnf", "08_bomb_b5_t1_p_t2.cnf", "08_bomb_b5_t1_p_t3.cnf", "08_bomb_b5_t1_p_t4.cnf",
              "08_bomb_b5_t1_p_t5.cnf", "08_bomb_b5_t5_p_t1.cnf", "08_bomb_b5_t5_p_t2.cnf", "09_coins_p01_p_t1.cnf",
              "09_coins_p02_p_t1.cnf", "09_coins_p03_p_t1.cnf", "09_coins_p04_p_t1.cnf", "09_coins_p05_p_t1.cnf",
              "09_coins_p05_p_t2.cnf", "09_coins_p10_p_t1.cnf", "10_comm_p01_p_t1.cnf", "10_comm_p01_p_t2.cnf",
              "10_comm_p02_p_t1.cnf", "10_comm_p03_p_t1.cnf", "11_emptyroom_d12_g6_p_t1.cnf",
              "11_emptyroom_d12_g6_p_t2.cnf", "11_emptyroom_d16_g8_p_t1.cnf", "11_emptyroom_d16_g8_p_t2.cnf",
              "11_emptyroom_d20_g10_corners_p_t1.cnf", "11_emptyroom_d24_g12_p_t1.cnf",
              "11_emptyroom_d28_g14_corners_p_t1.cnf", "11_emptyroom_d4_g2_p_t10.cnf", "11_emptyroom_d4_g2_p_t1.cnf",
              "11_emptyroom_d4_g2_p_t2.cnf", "11_emptyroom_d4_g2_p_t3.cnf", "11_emptyroom_d4_g2_p_t4.cnf",
              "11_emptyroom_d4_g2_p_t5.cnf", "11_emptyroom_d4_g2_p_t6.cnf", "11_emptyroom_d4_g2_p_t7.cnf",
              "11_emptyroom_d4_g2_p_t8.cnf", "11_emptyroom_d4_g2_p_t9.cnf", "11_emptyroom_d8_g4_p_t1.cnf",
              "11_emptyroom_d8_g4_p_t2.cnf", "11_emptyroom_d8_g4_p_t3.cnf", "11_emptyroom_d8_g4_p_t4.cnf",
              "12_flip_1_p_t10.cnf", "12_flip_1_p_t1.cnf", "12_flip_1_p_t2.cnf", "12_flip_1_p_t3.cnf",
              "12_flip_1_p_t4.cnf", "12_flip_1_p_t5.cnf", "12_flip_1_p_t6.cnf", "12_flip_1_p_t7.cnf",
              "12_flip_1_p_t8.cnf", "12_flip_1_p_t9.cnf", "12_flip_no_action_1_p_t10.cnf",
              "12_flip_no_action_1_p_t1.cnf", "12_flip_no_action_1_p_t2.cnf", "12_flip_no_action_1_p_t3.cnf",
              "12_flip_no_action_1_p_t4.cnf", "12_flip_no_action_1_p_t5.cnf", "12_flip_no_action_1_p_t6.cnf",
              "12_flip_no_action_1_p_t7.cnf", "12_flip_no_action_1_p_t8.cnf", "12_flip_no_action_1_p_t9.cnf",
              "13_ring2_r6_p_t1.cnf", "13_ring2_r6_p_t2.cnf", "13_ring2_r6_p_t3.cnf", "13_ring2_r8_p_t1.cnf",
              "13_ring2_r8_p_t2.cnf", "13_ring2_r8_p_t3.cnf", "13_ring_3_p_t1.cnf", "13_ring_3_p_t2.cnf",
              "13_ring_3_p_t3.cnf", "13_ring_3_p_t4.cnf", "13_ring_4_p_t1.cnf", "13_ring_4_p_t2.cnf",
              "13_ring_4_p_t3.cnf", "13_ring_5_p_t1.cnf", "13_ring_5_p_t2.cnf", "13_ring_5_p_t3.cnf",
              "14_safe_safe_10_p_t10.cnf", "14_safe_safe_10_p_t1.cnf", "14_safe_safe_10_p_t2.cnf",
              "14_safe_safe_10_p_t3.cnf", "14_safe_safe_10_p_t4.cnf", "14_safe_safe_10_p_t5.cnf",
              "14_safe_safe_10_p_t6.cnf", "14_safe_safe_10_p_t7.cnf", "14_safe_safe_10_p_t8.cnf",
              "14_safe_safe_10_p_t9.cnf", "14_safe_safe_30_p_t1.cnf", "14_safe_safe_30_p_t2.cnf",
              "14_safe_safe_30_p_t3.cnf", "14_safe_safe_30_p_t4.cnf", "14_safe_safe_30_p_t5.cnf",
              "14_safe_safe_30_p_t6.cnf", "14_safe_safe_5_p_t10.cnf", "14_safe_safe_5_p_t1.cnf",
              "14_safe_safe_5_p_t2.cnf", "14_safe_safe_5_p_t3.cnf", "14_safe_safe_5_p_t4.cnf",
              "14_safe_safe_5_p_t5.cnf", "14_safe_safe_5_p_t6.cnf", "14_safe_safe_5_p_t7.cnf",
              "14_safe_safe_5_p_t8.cnf", "14_safe_safe_5_p_t9.cnf", "15_sort_num_s_3_p_t10.cnf",
              "15_sort_num_s_3_p_t1.cnf", "15_sort_num_s_3_p_t2.cnf", "15_sort_num_s_3_p_t3.cnf",
              "15_sort_num_s_3_p_t4.cnf", "15_sort_num_s_3_p_t5.cnf", "15_sort_num_s_3_p_t6.cnf",
              "15_sort_num_s_3_p_t7.cnf", "15_sort_num_s_3_p_t8.cnf", "15_sort_num_s_3_p_t9.cnf",
              "15_sort_num_s_4_p_t1.cnf", "16_uts_k1_p_t10.cnf", "16_uts_k1_p_t1.cnf", "16_uts_k1_p_t2.cnf",
              "16_uts_k1_p_t3.cnf", "16_uts_k1_p_t4.cnf", "16_uts_k1_p_t5.cnf", "16_uts_k1_p_t6.cnf",
              "16_uts_k1_p_t7.cnf", "16_uts_k1_p_t8.cnf", "16_uts_k1_p_t9.cnf", "16_uts_k2_p_t1.cnf",
              "16_uts_k2_p_t2.cnf", "16_uts_k3_p_t1.cnf"]

if __name__ == "__main__":

      # alg_types = [ "static", "dynamic",  "random_selection_1234" ]
    # alg_types = [ "rand_dynamic" ]# ,  "random_selection_1234" ]
    alg_types = [ "static", "dynamic" ]
    # alg_types = [  "dynamic" ]
    # alg_types = [  "dynamic" , "static"]
    # FOLDER = "Dataset_preproc"
    # result_folder = "./results_aaai2/"
    result_folder = "./results/"
    # FOLDER = "Dataset_preproc_final"
    FOLDER = "Dataset_preproc_NO_COMPILE"
    HEUR_NAMES = {"MC/": "actual_MC", "WMC/": "actual_WMC", "half/": "relative_weight", "estimate/": "estimated_WMC", "random":"random", "hybrid_wmc/": "hybrid_wmc"}
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

    # what_more_to_run(expr_folders, alg_types, columns)
    # group_ecai23_data(expr_folders, alg_types, columns)
    # check_benchmark_preproc2()
    # exit(6)
    # best_ratio_per_alg(expr_folders, alg_types, columns)

    # plot_best_point_per_instance(expr_folders[0], alg_types, columns)
    # plot_best_point_per_instance(expr_folders[1], alg_types, columns)

    # plot_percentage_of_assigned_vars(expr_folders[2], alg_types, columns, filter_timeout=False, filter_conflict=False)
    # plot_percentage_of_assigned_vars(expr_folders[1], alg_types, columns, filter_timeout=True, filter_conflict=False)
    # plot_percentage_of_assigned_vars(expr_folders[2], alg_types, columns, filter_timeout=True, filter_conflict=False)

    # plot_time(expr_folders[2], alg_types, columns, filter_timeout=False, filter_conflict=False)


    # plot_percentage_of_assigned_backbones(expr_folders[0], alg_types, columns, filter_timeout=False, filter_conflict=False)
    # plot_percentage_of_assigned_backbones(expr_folders[1], alg_types, columns, filter_timeout=False, filter_conflict=False)
    # plot_percentage_of_assigned_backbones(expr_folders[2], alg_types, columns)
    # exit(8)

    # histogram_of_best_points_per_instance(expr_folders, alg_types, columns, "manhattan")
    # histogram_of_best_points_per_instance(expr_folders, alg_types, columns, "euclidean")
    # average_area_of_efficiency(expr_folders, alg_types, columns)

    # filename=("ResultsByInstanceDataset_all_by_instance")
    # generate_latex_report_per_instance(filename)
    # latex_report_for_average([expr_folders[0]], alg_types, columns)
    # latex_report_for_average([expr_folders[1]], alg_types, columns)
    # latex_report_for_average([expr_folders[2]], alg_types, columns)
    # count_nb_iterations(expr_folders, alg_types, columns)
    # read_folder()

    # eval_progress(expr_folders, out_file+"efficiency", "title", alg_types, 50, columns, "WMC", padding=True, same_length=same_length)
    # exit(4)

    # subfolder = "planning"
    # subfolder = "iscas"
    # count_conflicts_timeout(expr_folders, alg_types, columns, subfolder)
    # exit(9)


    #----------------------time table ---------------------------
    # result_folder = "./results_aaai_final/"
    # FOLDER = "Dataset_preproc_NO_COMPILE"
    # FOLDER2 = "Dataset_preproc_NO_COMPILE_2"
    # expr_folders = [result_folder + FOLDER + "_WMC/", result_folder + FOLDER2 + "_wscore_estimate/",
    #                   result_folder + FOLDER2 + "_wscore_half/", result_folder + FOLDER + "_hybrid_wmc/",
    #                   result_folder + FOLDER2 + "_rand_dynamic/"]
    #
    # create_time_table_d4(expr_folders, alg_types, columns, nocompile=True, cutoff=cutoff)
    # exit(4)
    #----------------------time table ---------------------------

    # ----------------------percent of vars assinged table ---------------------------
    result_folder = "./results_aaai_final/"
    FOLDER = "Dataset_preproc_final"
    expr_folders = [result_folder + FOLDER + "_WMC/", result_folder + FOLDER + "_wscore_estimate/",
                      result_folder + FOLDER + "_wscore_half/",
                    result_folder + FOLDER + "_hybrid_wmc/",
                      result_folder + FOLDER + "_rand_dynamic/"]

    # create_percent_of_assigned_table_d4(expr_folders, alg_types, columns, nocompile=False, cutoff={})
    # exit(4)

    best_ratio_per_alg(expr_folders, alg_types, columns, "")
    exit(5)
    # ----------------------percent of vars assinged  table ---------------------------

    subfolder = ""
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
         out_file = result_folder+"Dataset_preproc_median_WMC_"

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

    exit(11)

    # read_ratio("paper_data/ratio_table.csv")
    # plot_init()
    # count_all_backbones()
    # exit(666)

    folder = "./aaai_data/DatasetA/"
    # compile_folder_with_stats(folder)
    # exit(9)


    labels = ["random_1234",  "random_selection_1234",  "random_ratio_selection_1234", "static", "static_ratio", "dynamic", "dynamic_ratio"]


    ################ paper results ###################

    # plt.rcParams.update({'font.size': 16})
    ################### avg dataset A and B
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/"]
    # outfile = "./paper_data/AB_avg_"
    # title = "Average efficiency over Dataset A and B"
    # create_average_efficiency_plot( exprs , outfile+"efficiency.png", title,  labels, 1)
    # title = "Average ratio over Dataset A and B"
    # create_average_ratio_plot(  exprs ,  outfile+"ratio.png", title, labels, 1)

    ################### avg iscas
    # exprs = ["./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/"]
    # outfile = "./paper_data/iscas_avg50_"
    # title = "Average efficiency over iscas instances"
    # create_average_efficiency_plot(exprs, outfile + "efficiency.png", title, labels, 50)
    # title = "Average ratio over iscas instances"
    # create_average_ratio_plot(exprs, outfile + "ratio.png", title, labels, 50)

    ################### avg planning
    # exprs = ["./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/comm/",
    #          "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
    #          "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    # outfile = "./paper_data/planning_avg50_"
    # title = "Average efficiency over planning instances"
    # create_average_efficiency_plot(exprs, outfile + "efficiency.png", title, labels, 50)
    # title = "Average ratio over planning instances"
    # create_average_ratio_plot(exprs, outfile + "ratio.png", title, labels, 50)

    ################### evaluate folders and average between folders
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    # exprs = ["./results/data_sdd/DatasetA/", "./results/data_sdd/DatasetB/" ]
    # exprs = ["./results/data/DatasetA/", "./results/data/DatasetB/",  "./results/data/iscas89/", "./results/data/iscas93/", "./results/data/iscas99/" ]
    # columns = ["p", "var", "value",  'n_vars', "MC", "BDD len", 'n_nodes', 'n_reorderings', 'dag_size', 'time']

    # labels = [ "static", "static_ratio", "dynamic", "dynamic_ratio"]#, "random_1234", "random_selection_1234" ]
    labels = [ "static", "dynamic", "random_selection_1234"]#, "dynamic", "random_selection_1234" ]
    # exprs = ["./results/wmc2022_track2_private_count/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    exprs = ["./results/sdd/wmc2022_track2_private_WMC/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # exprs = ["./results/sdd/wmc2022_track2_private_comp/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # exprs = ["./results/wmc2022_track2_private_MC/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # exprs = ["./results/wmc2022_track2_private_count/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # exprs = ["./results/test/f1g1/", "./results/test/f1g2/","./results/test/f2g1/", "./results/test/f2g2/"]#, "./results/data_sdd/DatasetB/",  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # exprs = [  "./results/data_sdd/iscas89/", "./results/data_sdd/iscas93/", "./results/data_sdd/iscas99/" ]
    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time', 'WMC', 'g2']
    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "g2"]

    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC"] #for d4
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time', 'WMC', "logWMC"] #for weighted sdd
    for f in exprs:
        type = f.split("/")[-2]
        # title = "Average efficiency over "+type+" instances"
        # create_average_efficiency_plot([f], f+type+"_avg_efficiency", title, labels, 1, columns)
        # title = "Average ratio over "+type+" instances"
        # create_average_ratio_plot([f], f+type+"_avg_ratio", title, labels, 1, columns)
        title = "Average weighted efficiency over " + type + " instances"
        create_average_efficiency_plot([f], f+type+"_avg_weighted_efficiency", title, labels, 1, columns)
        title = "Average weighted ratio over " + type + " instances"
        create_average_ratio_plot([f], f+type+"_avg_weighted_ratio", title, labels, 1, columns)
        evaluate_folder( f, labels, columns )
    exit(1)

    ##################
    # columns =  ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    # type = "dynamic_ratio"
    # stats_file = "./paper_data/Planning/uts/" + "dataset_stats_" + type + "_reorder.csv"
    # expr_data_dynamic2 = ExprData(columns)
    # expr_data_dynamic2.read_stats_file(stats_file)
    # expr_data_dynamic2.best_ratio_table_per_alg()

    ###################
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    #     "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/", "./paper_data/Planning/comm/",
    #          "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
    #               "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/" ]
    # labels = ["random_1234", "random_selection_1234",  "random_ratio_selection_1234", "static", "static_ratio", "dynamic", "dynamic_ratio"]
    # # create_best_ratio_table("paper_data/ratio_table.csv", exprs, labels, aggregate=True)

    # create_time_table(exprs, labels)
    # metric = "mc"
    # read_ratio_table("paper_data/base_ratio_table.csv", metric)
    # exprs = ["./paper_data/DatasetA/"]
    # for f in exprs:
    #     evaluate_folder(f, labels)

    ################ end paper results ###################


    # look at ratio table

    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/comm/",
    #          "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
    #          "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    # create_best_ratio_tables(exprs)

    # count_all_backbones()
    # folder = "./paper_data/DatasetA/"




    # alg_types = [ "init", "random", "random_selection", "static","static_ratio", "dynamic","dynamic_ratio"]
    #
    # exprs = ["./paper_data/BayesianNetwork/","./paper_data/DatasetA/","./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/base/", "./paper_data/Planning/blocks/",  "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/flip/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/", "./paper_data/Planning/comm/"]

    # exprs = ["./paper_data/Planning/comm/", "./paper_data/Planning/coins/"]
    # exprs = ["./aaai_data/datasetA/"]
    # for e in exprs:
    #     evaluate_folder(e, ["lit", "opp"])

    ############################ AAAI eval #############################

    # exprs = [   "./aaai_data/output/Planning/blocks/",    "./aaai_data/output/Planning/ring/",
    #     "./aaai_data/output/Planning/sort/" ]

    # exprs = ["./aaai_data/output/Planning/coins/"]

    # exprs = [  "./aaai_data/output/DatasetA/", "./aaai_data/output/DatasetB/" ,
    #"./aaai_data/output/Planning/coins/",
    # "./aaai_data/output/DatasetB/" # , "./aaai_data/output/DatasetB/",
    #    "./aaai_data/output/iscas/iscas89/",
    # exprs = ["./aaai_data/output/iscas/iscas93/","./aaai_data/output/iscas/iscas99/"]
    # labels = ["dynamic_opp", "dynamic_lit"]
    # labels = [ "dlisTieBreakPurity", "StrictDlisTieBreakPurity", "dynamic_lit"] #, "dynamic_opp","dynamic_wopp", "dynamic_lit", "dynamic_wlit", "dynamic_ratio", "dynamic", "random_ratio_selection"]
    # colors = ["blue", "cyan", mcolors.CSS4_COLORS["gold"], "orange", "green", "olive", 'red', "grey"]


    for exp_folder in expr_folders:
        labels = ["dynamic_opp","dynamic_wopp", "dynamic_lit", "dynamic_wlit", "static_opp","static_wopp", "static_lit", "static_wlit", "dynamic_ratio", "dynamic", "random_ratio_selection"]
        # labels = ["dynamic_opp","dynamic_wopp", "dynamic_lit", "dynamic_wlit", "dynamic_ratio", "dynamic", "random_ratio_selection"]
        # labels = ["dynamic_opp","dynamic_wopp", "dynamic_lit", "dynamic_wlit",  "dynamic_ratio", "dynamic", "random_ratio_selection", "StrictDlisTieBreakPurity", "dlisRelativePurity"]
        columns = ["p", "var", "value", "nb_vars", "nb_cls", "obj", 'time', "MC", "dag_size", "compilation_time"] #this variable is getting modified somewhere in the code below
        type = exp_folder.split("/")[-2]
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",type)
        title = "Average efficiency over "+type+" instances"
        create_average_efficiency_plot([exp_folder], exp_folder+type+"_avg_efficiency", title, labels, 1, columns)
        title = "Average ratio over "+type+" instances"
        create_average_ratio_plot([exp_folder], exp_folder+type+"_avg_ratio", title, labels, 1, columns)
        evaluate_folder( exp_folder, labels, columns)




