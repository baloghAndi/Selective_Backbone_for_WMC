import decimal

import CSP
from pysdd.sdd import SddManager, Vtree
import time
import graphviz
import os
import subprocess
import math
import datetime
class WCNF:
    def __init__(self,logger=None, scalar=0, NO_COMPILE=False):
        self.variables = {}  # name:domain
        self.literals = []
        self.cls = []
        self.partial_assignment = PartialAssigment()
        self.logger = logger
        self.instance_name = ""
        self.p = 0
        self.literal_weights = None
        self.weight_file = None
        self.obj_type = ""
        self.scalar = scalar
        self.init_MC = -1
        self.init_WMC = -1
        self.trivial_backbone = []
        self.NO_COMPILE = NO_COMPILE
        self.literal_clause_map = {} #lit:[len of clauses it appears in]

    def load_file(self, filename, obj_type=None, heur_type=None):
        self.obj_type = obj_type
        self.heur_type = heur_type
        self.instance_name = filename

        self.weight_file = filename.replace(".cnf", '.w') #for  non weighted
        if self.scalar > 0 :
            self.weight_file = filename.replace(".cnf", "_w"+str(self.scalar)+".w") #for weighted


        with open(filename, "r") as f:
            content = f.readlines()
            # remove init comments
            # nb_vars = 0
            init_nb_lines = 1
            for c in content:
                if c.startswith("c"):
                    init_nb_lines += 1
                    continue
                else:
                    print(c)
                    nb_vars = int(c.strip().split(" ")[2])
                    break
            # if nb_vars > 300:
            #     print("Nb vars more then 300")
                # exit(111)
                # return False
            # nb_vars = int(content[0].strip().split(" ")[2])
            print("NB VARS", nb_vars)
            for i in range(1, nb_vars + 1):
                self.literal_clause_map[-i]= []
                self.literal_clause_map[i] = []
            # self.literal_weights = {0: nb_vars * [1], 1: nb_vars * [1]}
            for str_clause in content[init_nb_lines:]:
                # if "c p weight" in str_clause:
                #     content = str_clause.strip().split(" ")
                #     weight = content[-2]
                #     literal = content[-3]
                #     print(content, weight, literal)
                #     literal = int(literal)
                #     if literal < 0:
                #         self.literal_weights[0][abs(literal)-1] = float(weight)
                #     else:
                #         self.literal_weights[1][abs(literal)-1] = float(weight)
                if 'c' in str_clause:
                    continue
                # str_clause.replace("-","~")

                else:
                    lits = [int(i) for i in str_clause.strip().split(" ")[:-1] if i != '']
                    if len(lits) == 0:
                        continue
                    if len(lits) == 1:
                        self.trivial_backbone.append(lits[0])
                    self.cls.append(lits)
                    for l in lits:
                        self.literal_clause_map[l].append(len(lits))

            self.literals = [i for i in range(1, nb_vars + 1)]
            if self.literal_weights == None:
                self.literal_weights = {0: nb_vars * [1], 1: nb_vars * [1]}
            #if weight file does not exists take 1 for each weight - will be equivalent to MC

            if os.path.exists(self.weight_file):
                print(self.weight_file)
                with open(self.weight_file, "r") as f:
                    content = f.readlines()
                    for str_line in content:
                        literal, weight = str_line.strip().split(" ")
                        # print(literal_weights, nb_vars)
                        literal = int(literal)
                        if literal < 0:
                            self.literal_weights[0][abs(literal) - 1] = float(weight)
                        else:
                            self.literal_weights[1][abs(literal) - 1] = float(weight)
            else:
                print("No weights give, default is 1, eq to MC")
                # self.write_weights(self.weight_file)

            print("finished reading")
            self.variables = {i: [0, 1] for i in self.literals}
            self.n = len(self.literals)

            # self.write_weights()
            if self.logger:
                if self.NO_COMPILE:
                    elapsed = self.logger.get_time_elapsed()
                    self.logger.log([0, "-1", "-1", self.n, len(self.cls), '-1', '-1',  '-1', elapsed, '-1', "-1", "-1" ])
                    return True
                if self.logger.compile:
                    nb_nodes, nb_edges, wmc, comp_time = self.compile_d4_wmc(self.instance_name, self.weight_file)
                    _, _, mc, _ = self.compile_d4_mc(self.instance_name)
                    # columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC"]
                    print( [0, "-1", "-1", self.n, len(self.cls),  mc, nb_edges, nb_nodes, comp_time, wmc])
                    if wmc == 0.0:
                        self.logger.log( [0, "-1", "-1", self.n, len(self.cls),  mc, nb_edges, nb_nodes, comp_time, wmc, 0, 0])
                        self.init_MC = 0
                        self.init_WMC = 0
                        print("WMC is 0")
                        return False
                    else:
                        logWMC = math.log10(wmc)
                        self.logger.log( [0, "-1", "-1", self.n, len(self.cls),  mc, nb_edges, nb_nodes, comp_time, wmc, logWMC, 0])
                        self.init_MC = mc
                        self.init_WMC = wmc
                        if type(wmc)==float and math.isinf(wmc):
                            print("WMC is inf")
                            return False
                else:
                    self.logger.log( [0, "-1", "-1", self.n, len(self.cls),  '-1', '-1', '-1', '-1', '-1', "-1"])

        # print(self.literals)

        return True

    def load_wcnf_file(self, filename):
        self.instance_name = filename
        self.weight_file = filename.replace(".cnf", '.w')

        with open(filename, "r") as f:
            content = f.readlines()
            #remove init comments
            nb_vars = 0
            for c in content:
                if c.startswith("c"):
                    continue
                else:
                    nb_vars = int(c.strip().split(" ")[2])
                    break
            print("NB VARS", nb_vars)
            self.literal_weights = {0: nb_vars * [1], 1: nb_vars * [1]}
            for str_clause in content[1:]:
                if "c p weight" in str_clause :
                    content = str_clause.strip().split(" ")
                    weight = content[-2]
                    literal = content[-3]
                    print(content, weight, literal)
                    literal = int(literal)
                    if literal < 0:
                        self.literal_weights[0][abs(literal)-1] = float(weight)
                    else:
                        self.literal_weights[1][abs(literal)-1] = float(weight)
                elif str_clause.startswith("w "):
                    content = str_clause.strip().split(" ")
                    weight = content[2]
                    literal = content[1]
                    print(content, weight, literal)
                    literal = int(literal)
                    if literal < 0:
                        self.literal_weights[0][abs(literal) - 1] = float(weight)
                    else:
                        self.literal_weights[1][abs(literal) - 1] = float(weight)
                elif 'c' in str_clause:
                    continue
                # str_clause.replace("-","~")

                else:
                    lits = [int(i) for i in str_clause.strip().split(" ")[:-1] if i != '']
                    if len(lits) == 0:
                        continue
                    self.cls.append(lits)
            self.literals = [i for i in range(1, nb_vars + 1)]
            if self.literal_weights == None:
                self.literal_weights = {0: nb_vars * [1], 1: nb_vars * [1]}
            print("finished reading")
            self.variables = {i: [0, 1] for i in self.literals}
            self.n = len(self.literals)
            # self.write_weights()
        self.n = len(self.literals)
        return True

    def compile_d4_mc(self, cnf_file):
        res = subprocess.run(["./d4", "-dDNNF", cnf_file ], stdout=subprocess.PIPE, text=True)
        output = res.stdout
        output = output.split("\n")
        # print(output)
        nb_nodes = 0
        nb_edges = 0
        comp_time = 0
        # wmc = 0
        for line in output:
            # print(line)
            if "Number of nodes:" in line:
                nb_nodes = int(line.split(" ")[-1].strip())
            elif "Number of edges:" in line:
                nb_edges = int(line.split(" ")[-1].strip())
            elif line.startswith("s "):
                mc = int(line.split(" ")[-1].strip())
            elif "Final time:" in line:
                comp_time = float(line.split(" ")[-1].strip())
        return nb_nodes, nb_edges, mc, comp_time

    def get_d4_wmc(self, cnf_file, weights_file):
        """
        Call d4 on command line and return relevant statistics
        """
        #( timeout 1m  time ../../../d4/d4-main/d4 -dDNNF $i -out="./d4/"$outfile.nnf ) 2>&1 | tee  "./d4/"$outfile"_stats".txt`
        # stats_file = weights_file.replace(".w", "_stats.txt")
        # f = open(stats_file, "w")
        # res = subprocess.run(["./d4", "-dDNNF", cnf_file, "-wFile="+weights_file ], stdout=f, text=True, timeout=600)
        print("running: ", "./d4 -no-dDNNF"+ cnf_file+ "-wFile="+weights_file+ "-mc" )
        res = subprocess.run(["./d4", "-no-dDNNF", cnf_file, "-wFile="+weights_file, "-mc"], stdout=subprocess.PIPE, text=True)
        output = res.stdout
        output = output.split("\n")
        solve_time = -1
        wmc = -1
        for line in output:
            if line.startswith("s "):
                scaled_wmc = float(line.split(" ")[-1].strip())
                if math.isinf(scaled_wmc):
                    scaled_wmc = int(line.split(" ")[-1].strip().split(".")[0])
                wmc = scaled_wmc
                # print("WMC", wmc)
                # if self.scaled_weights:
                #     wmc = scaled_wmc / math.pow(2, self.n)
            elif "Final time:" in line:
                solve_time = float(line.split(" ")[-1].strip())
        if wmc == -1:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            self.write_cnf(file_name=cnf_file.replace(".cnf", "_error"+timestamp+".cnf"))
            print(output)
            self.logger.log_error(cnf_file,output)
        return  wmc, solve_time

    def compile_d4_wmc(self, cnf_file, weights_file):
        """
        Call d4 on command line and return relevant statistics
        """
        #( timeout 1m  time ../../../d4/d4-main/d4 -dDNNF $i -out="./d4/"$outfile.nnf ) 2>&1 | tee  "./d4/"$outfile"_stats".txt`
        # stats_file = weights_file.replace(".w", "_stats.txt")
        # f = open(stats_file, "w")
        # res = subprocess.run(["./d4", "-dDNNF", cnf_file, "-wFile="+weights_file ], stdout=f, text=True, timeout=600)
        res = subprocess.run(["./d4", "-dDNNF", cnf_file, "-wFile="+weights_file], stdout=subprocess.PIPE, text=True)
        output = res.stdout
        output = output.split("\n")
        # print(output)
        nb_nodes = 0
        nb_edges = 0
        comp_time = 0
        # wmc = 0
        for line in output:
            #print(line)
            if "Number of nodes:" in line:
                nb_nodes = int(line.split(" ")[-1].strip())
            elif "Number of edges:" in line:
                nb_edges = int(line.split(" ")[-1].strip())
            elif line.startswith("s "):
                scaled_wmc = float(line.split(" ")[-1].strip())
                if math.isinf(scaled_wmc):
                    scaled_wmc = int(line.split(" ")[-1].strip().split(".")[0])
                wmc = scaled_wmc
                # print("WMC", wmc)
                # if self.scaled_weights:
                #     wmc = scaled_wmc / math.pow(2, self.n)
            elif "Final time:" in line:
                comp_time = float(line.split(" ")[-1].strip())

        # res = subprocess.run(["./d4", "-dDNNF", cnf_file ], stdout=subprocess.PIPE, text=True)
        # output = res.stdout
        # output = output.split("\n")
        # # print(output[-2])
        # mc = int(output[-2].split(" ")[-1].strip())

        # print("nb_nodes, nb_edges, wmc, mc", nb_nodes, nb_edges, wmc, mc)
        return nb_nodes, nb_edges, wmc, comp_time

    def check_wmc_of(self, var, value, compile=True):
        #LOOK OUT : sdd removes node if its values are interchangeable, looking at global to include both solutions, conditioning doesn't work
        # this way of conjoining  adds another node
        #overwtite cnf file with added clause
        if value == 0:
            var = -var
        # if "_final" in self.instance_name:
        #     cnf_f = self.instance_name.replace("_final", "")
        cnf_file_name = self.instance_name.replace(".cnf","_temp"+self.obj_type+self.heur_type+".cnf")
        self.write_cnf_extend(cnf_file_name, [[var]])
        if compile:
            nb_nodes, nb_edges, wmc, comp_time = self.compile_d4_wmc(cnf_file_name, self.weight_file)
        else:
            wmc, comp_time = self.get_d4_wmc(cnf_file_name, self.weight_file)
            nb_nodes = -1
            nb_edges = -1
        # self.write_cnf()
        return nb_nodes, nb_edges,  wmc,  comp_time

    def check_mc_of(self, var, value):
        # LOOK OUT : sdd removes node if its values are interchangeable, looking at global to include both solutions, conditioning doesn't work
        # this way of conjoining  adds another node
        # overwtite cnf file with added clause
        if value == 0:
            var = -var
        # if "_final" in self.instance_name:
        #     cnf_f = self.instance_name.replace("_final", "")
        cnf_file_name = self.instance_name.replace(".cnf", "_temp" + self.obj_type + self.heur_type + ".cnf")
        self.write_cnf_extend(cnf_file_name, [[var]])
        nb_nodes, nb_edges, mc, comp_time = self.compile_d4_mc(cnf_file_name)
        # self.write_cnf()
        return nb_nodes, nb_edges, mc, comp_time

    def check_mc_bdd_ratio_of(self, var, value):
        if value == 0:
            var = -var
        # temp_root = self.root_node.condition(self.sdd_manager.literal(var))
        self.sdd_manager.auto_gc_and_minimize_off()
        temp_root = self.root_node & self.sdd_manager.get_vars(var)

        # !!!! use ref if minimizing vtree
        # print(self.sdd_manager.get_vars(var))
        # temp_root = self.root_node & self.sdd_manager.get_vars(var)  # same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))
        # print("condition_node ", var, value, condition_node.manager.global_model_count(condition_node), condition_node.model_count(), condition_node.size(), condition_node.count(), condition_node)
        # temp_root.ref()
        # temp_root.manager.minimize()

        model_count = temp_root.manager.global_model_count(temp_root) #temp_root.model_count()
        # temp_root = self.root_node.conjoin(self.sdd_manager.literal(var))
        if temp_root.size() == 0:
            r = 0
        else:
            r = model_count/temp_root.size()
        # print( var, value, r, model_count, temp_root.size(), temp_root.count())
        #score_of_assignment, size, node_count, temp_root
        # self.sdd_manager.auto_gc_and_minimize_on()
        return r, temp_root.size(), temp_root.count(), temp_root #TODO need to send back ratio but also log mc


    def extend_assignment(self, var, value, score, propagate=False):
        #write out new cnf file with the uploaded assignment, change instance name
        #extend self n cls etc
        opp = -var
        if value == 0:
            var = -var
            opp = abs(var)
        new_cls = [var]
        self.literal_clause_map[var].append(1)
        #call this if you want to save intermediate cnf files
        # fname = self.instance_name.replace(".cnf", "_x"+str(var) + ".cnf")
        # self.write_cnf_extend(fname, [new_cls])

        self.cls.append(new_cls)
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[abs(var)] = value
        self.partial_assignment.score = score

        if propagate:
            # remove clauses that contain var - they are already true
            # remove the opposite of the literal from other clauses
            # do not change number of variables
            csp_clauses = []
            csp_clauses.append([var]) #use this for type 2 extend
            self.trivial_backbone = [] #eliminate all trivial backbones as we go through the whole cnf again here
            for c in self.cls:
                if var not in c:
                    updated_c = []
                    for i in c:
                        if i != opp:
                            updated_c.append(i)
                        #remove count for i - it is removed from cls
                    if len(updated_c) != len(c):
                        for temp in c:
                            self.literal_clause_map[temp].remove(len(c))
                        for temp in updated_c:
                            self.literal_clause_map[temp].append(len(updated_c))
                    # print(c, updated_c, variable, value)
                    csp_clauses.append(updated_c)

                    if len(updated_c) == 1 and abs(updated_c[0]) not in self.partial_assignment.assigned:
                        #check that variable has not been assigned already
                        self.trivial_backbone.append(updated_c[0])
                # if var in c we should just eliminate it so not adding to new clauses
                else: #remove clause c because it is satisfied by var - need to decrease cls count for literals in it
                    for l in c:
                        self.literal_clause_map[l].remove(len(c))
            self.cls = csp_clauses.copy()
            if self.obj_type == "WMC":
                cnf_file_name = self.instance_name.replace(".cnf", "_temp"+self.obj_type+self.heur_type+".cnf")
                self.print_clauses( cnf_file_name, csp_clauses, self.n)



    def print_clauses(self, cnf_file, cls, n):
        """
        Print current cnf - with last self.p clauses as fist ones
        :param cnf_file:
        :return:
        """
        print("writing file: ", cnf_file)
        f = open(cnf_file, "w")
        f.write("p cnf "+ str(n) + " " +str(len(cls))+"\n" )
        for c in cls:
            f.write(" ".join([str(x) for x in c])+" 0 \n")
        f.close()
    def copy_sdd(self):
        print("copy cnf")
        copy_cnf = CNF()
        copy_cnf.variables = self.variables
        copy_cnf.literals =  self.literals
        copy_cnf.cls = self.cls
        copy_cnf.partial_assignment =  self.partial_assignment
        copy_cnf.logger = self.logger
        copy_cnf.instance_name =  self.instance_name
        copy_cnf.p = self.p
        copy_cnf.n = self.n
        print(self.sdd_manager == self.root_node.manager)
        copy_cnf.root_node = self.root_node.copy()
        copy_cnf.sdd_manager = self.sdd_manager.copy([self.root_node])
        print(copy_cnf.sdd_manager == copy_cnf.root_node.manager)
        print(copy_cnf.root_node.size(), copy_cnf.root_node.model_count(), self.root_node.size(), self.root_node.model_count())
        # copy_cnf.sdd_manager.auto_gc_and_minimize_off()
        # v = copy_cnf.sdd_manager.get_vars(1)
        # beta = copy_cnf.root_node & v
        # print("got beta")
        return copy_cnf

    def write_weights(self, file_name=None, force=False):
        if file_name == None:
            file_name = self.weight_file
        if not os.path.exists(file_name) or force:
            print("write file ", file_name)
            f = open(file_name, "w")
            i = 1
            for negl, l  in zip(self.literal_weights[0], self.literal_weights[1]):

                f.write(str(i)+" "+str(l)+"\n")
                f.write("-"+str(i)+" "+str(negl)+"\n")
                i += 1
            f.close()
            return True
        return False

    def write_scaled_weights(self, file_name=None):
        if file_name == None:
            file_name = self.weight_file
        if not os.path.exists(file_name):
            f = open(file_name, "w")
            i = 1
            for negl, l  in zip(self.literal_weights[0], self.literal_weights[1]):

                f.write(str(i)+" "+str(2* l)+"\n")
                f.write("-"+str(i)+" "+str(2 * negl )+"\n")
                i += 1
            f.close()
            return True
        return False

    def write_cnf(self, file_name=None):
        # if "_final.cnf" not in self.instance_name:
        #     self.instance_name.replace(".cnf", "_final.cnf")
        print(file_name)
        if file_name == None:
            file_name = self.instance_name
        f = open(file_name, "w")
        f.write("p cnf " + str(self.n) + " " + str(len(self.cls)) + "\n")
        for c in self.cls:
            f.write(" ".join([str(x) for x in c]) + " 0 \n")
        f.flush()
        f.close()

    def write_cnf_extend(self, cnf_file_name, extra_cls):
        # if "_final.cnf" not in self.instance_name:
        #     self.instance_name.replace(".cnf", "_final.cnf")
        f = open(cnf_file_name, "w")
        cls_len = len(extra_cls) + len(self.cls)

        f.write("p cnf " + str(self.n) + " " + str(cls_len) + "\n")
        for c in self.cls:
            f.write(" ".join([str(x) for x in c]) + " 0 \n")
        for c in extra_cls:
            f.write(" ".join([str(x) for x in c]) + " 0 \n")
        f.flush()
        f.close()

    def opposite_occurance(self, var, value):
        """
        The opp score of a literal is the number of times its opposite occures in the instance
        the smaller the value the better
        :param var:
        :param value:
        :return:
        """
        opp_literal = var
        if value == 1: #1 is we want opposite count need to minimize in this case
            opp_literal = -var
        # print(var, value, opp_literal)
        count = 0
        for cls in self.cls:
            if opp_literal in cls:
                count += 1
        return count

    def occurance(self, var, value):
        """
        The  score of a literal is the number of times it occures in the instance
        :param var:
        :param value:
        :return:
        """
        if value == 0:
            lit = -var
        else:
            lit = var
        return len(self.literal_clause_map[lit])
        # print(var, value, opp_literal)
        # count = 0
        # for cls in self.cls:
        #     if lit in cls:
        #         count += 1
        # return count

    def adjusted_occurance(self, var, value):
        """
        sum of the inverses of the cardinalities of the clauses that l appears in.So for example, if appears in one clause with 4 literals,
        in three clauses with 5 literals, and in two clauses with 8 literals, then its adjusted number of occurrences is
        AdjOcc(l) = 1/4 + 3/5 + 2/8 = 1.1
        """
        adjsum = 0
        cls_len_occurance = {} # eg: 4:1 , 5:3, 8:2
        if value == 0:
            lit = -var
        else:
            lit = var
        # print(var, value, opp_literal)
        for m in self.literal_clause_map[lit]:
            if m not in cls_len_occurance:
                cls_len_occurance[m] = 0
            cls_len_occurance[m] += 1
        count = sum([v/k for k,v in cls_len_occurance.items()] )
        # for cls in self.cls:
        #     if lit in cls:
        #         for m in self.literal_clause_map[lit]:
        #             m = len(cls)
        #             if m not in cls_len_occurance:
        #                 cls_len_occurance[m] = 0
        #             cls_len_occurance[m] += 1
        # count = sum([v / k for k, v in self.cls_len_occurance.items()])
        return count

    def calculate_score(self, var, value, score_type, weighted=True):
        weight = self.literal_weights[value][var-1] / (self.literal_weights[0][var-1]+self.literal_weights[1][var-1])
        opp_lit = var
        if value == 0:
            lit = -var
            opp_lit = var
        else:
            lit = var
            opp_lit = -var
        if lit in self.trivial_backbone:
            print("backbone: ", lit)
            if weighted:
                return weight * 100000
            else:
                return 100000
        score = 0
        #TODO: create prerosessing that calculated the counts so we don't run throught the clauses more then once
        if score_type == "half":
            score = 1
        elif score_type == "occratio":
            score = (1 + self.occurance(var, value)) / (2 + self.occurance(var, 0) + self.occurance(var, 1))
            # print("occratio", weight * score)
        elif score_type == "adjoccratio":
            score = (1 + self.adjusted_occurance(var, value, )) / (2 + self.adjusted_occurance(var, 0) + self.adjusted_occurance(var, 1))
        elif score_type == "estimate":
            # opp_occurance_list = [ cls for cls in self.cls if opp_lit in cls]
            # occurance_list = [ cls for cls in self.cls if lit in cls]
            # if occurance_list == [] or opp_occurance_list == []:
            #     print("UNCONSTRAINED VAR CHOSEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     score =  0
            # else:
            est_lit = math.prod([1-math.pow(0.5, cls_len-1 ) for cls_len in self.literal_clause_map[opp_lit]])
            est_opp_lit = math.prod([1-math.pow(0.5, cls_len-1 ) for cls_len in self.literal_clause_map[lit]])
            # est_lit = math.prod([1 - math.pow(0.5, len(cls) - 1) for cls in opp_occurance_list])
            # est_opp_lit = math.prod([1 - math.pow(0.5, len(cls) - 1) for cls in occurance_list])
            score = est_lit / (est_lit + est_opp_lit)
                # score = math.prod([1-math.pow(0.5, len(cls)-1 ) for cls in occurance_list])
        elif score_type == "otherwg":
            # opp_occurance_list = [cls for cls in self.cls if opp_lit in cls]
            score = 1
            for cls in self.cls:
                if opp_lit in cls:
                    score *= (1-math.prod([self.literal_weights[0 if cvar<0 else 1][abs(cvar)-1] for cvar in cls if cvar!=opp_lit]))
        else:
            score = 0
        if weighted:
            return weight * score
        else:
            return score


    def count_irrelevant_lits(self):
        lit_count = {}
        for l in self.literals:
            lit_count[l] = 0
            lit_count[-l] = 0
        for c in self.cls:
            for l in c:
                lit_count[l] += 1
        irrelevant = []
        for l,count in lit_count.items():
            if count == 0 :
                irrelevant.append(l)
        print(len(irrelevant)/2,2*len(self.literals), irrelevant)


class PartialAssigment:
    def __init__(self):
        self.score = 0
        self.assigned = {} # dict (variable, value)



if __name__ == "__main__":
    cnf = WCNF()
    cnf.get_d4_wmc("./input/Dataset_preproc/01_istance_K3_N15_M45_01.cnf", "./input/Dataset_preproc/01_istance_K3_N15_M45_01_w3.w")
    # get_d4_wmc
    exit(99)


    # vtree = Vtree(var_count=120, vtree_type='balanced')
    # sdd_manager = SddManager.from_vtree(vtree)
    # sdd_manager.auto_gc_and_minimize_off()
    # print("reading...")
    # root_node = sdd_manager.read_cnf_file("./input/wmc2020_track2_all/track2_005.mcc2020.cnf".encode('utf-8'))
    # print("mc ", root_node.model_count())
    # exit(100)
    cnf =CNF()
    cnf.load_file("./test/nqueens_4_modified.cnf")
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # for m in cnf.root_node.models():
    #     print(m)
    #
    # cnf = CNF()
    # cnf.load_file("../nqueens_4.cnf")
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # for m in cnf.root_node.models():
    #     print(m)

    # d = cnf.root_node.condition(2)
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # print("new :", d.model_count(), d.size(), d.count())

    # g = graphviz.Source(cnf.sdd_manager.dot(d))
    # cnf_file_name = "./condition2_"
    # g.render(view=False, format='png', filename=cnf_file_name + "_sdd")
    # g = graphviz.Source(cnf.sdd_manager.vtree().dot())
    # g.render(view=False, format='png', filename=cnf_file_name + "_vtree")

    # #
    # d = cnf.root_node.condition(-2)
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # print("new :", d.model_count(), d.size(), d.count())

    # print("conjoin")
    # d = cnf.root_node.conjoin(cnf.sdd_manager.literal(2))
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # print("new :", d.model_count(), d.size(), d.count())
    #
    # g = graphviz.Source(cnf.sdd_manager.dot(d))
    # cnf_file_name = "./conjoin2_"
    # g.render(view=False, format='png', filename=cnf_file_name + "_sdd")
    # g = graphviz.Source(cnf.sdd_manager.vtree().dot())
    # g.render(view=False, format='png', filename=cnf_file_name + "_vtree")

    # d = cnf.root_node.conjoin(cnf.sdd_manager.literal(-2))
    # print("root:", cnf.root_node.model_count(), cnf.root_node.size(), cnf.root_node.count())
    # print("new :", d.model_count(), d.size(), d.count())
    #
    var_count = 3
    var_order = [1,2,3]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)
    # manager.set_prevent_transformation( prevent=True)

    print("constructing SDD ... ")
    a, b, c = [manager.literal(i) for i in range(1, 4)]
    alpha =  (~a & ~b & ~c ) | (~a & b & ~c ) | ( a & b & c ) | (a & ~b & c )

    # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha) )
    # beta = alpha.condition(c)
    # print(beta.manager.var_count(), beta.manager.count(), beta.manager.size())
    # print("----")
    # m=beta.manager
    # print(m.count(), manager.count())
    #
    # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha) , beta.model_count(), manager.model_count(beta), beta.manager.global_model_count(alpha) )
    # # beta = alpha.condition(~c)
    # # # beta.manager.
    # # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha), beta.model_count(),
    # # manager.model_count(beta), beta.manager.global_model_count(beta))
    #
    # r = manager.exists(3,alpha)
    # print(r.model_count(), r.manager.model_count(r), r.manager.global_model_count(r))

    # vtree = Vtree(var_count, var_order, vtree_type)
    # manager = SddManager.from_vtree(vtree)
    # manager.set_prevent_transformation(prevent=True)

    # print("constructing SDD ... ")
    # a, b, c = [manager.literal(i) for i in range(1, 4)]
    # alpha = (~a & ~b & ~c) | (~a & b & ~c) | (a & b & c) | (a & ~b & c)
    alpha.ref()
    manager.auto_gc_and_minimize_on()
    v = manager.get_vars(3)
    beta = alpha & v
    beta.ref()
    print("alpha", alpha.model_count(), alpha.manager.global_model_count(alpha), alpha.size())
    print("beta", beta.model_count(), beta.manager.global_model_count(beta), beta.size())

    #
    # print(alpha.manager == manager, alpha.literal)
    # #copy alpha
    # alpha_manager_copy = alpha.manager.copy([alpha])
    # alpha_copy = alpha.copy()
    # print(alpha_copy.manager == alpha_manager_copy)
    #
    # alpha_manager_copy = SddManager.from_vtree(alpha.manager.vtree())
    # alpha_copy = alpha.copy(alpha_manager_copy)
    # print(alpha_copy.manager == alpha_manager_copy)
    # exit(8)
    #
    # v_copy = alpha_manager_copy.get_vars(3)
    # print(v.model_count())
    # alpha_manager_copy.auto_gc_and_minimize_on()
    # print(alpha.model_count(), alpha.size(), manager.global_model_count(alpha), alpha_copy.model_count(), alpha_copy.size(), alpha_manager_copy.global_model_count(alpha))
    #
    # beta_copy = alpha_copy & v_copy
    #
    #
    # # manager.minimize()
    # print("beta", beta.model_count(),beta.manager.global_model_count(beta), beta.size())
    # print("beta_copy", beta_copy.model_count(),beta_copy.manager.global_model_count(beta_copy), beta_copy.size())
    # print(alpha.model_count(), alpha.size(), manager.global_model_count(alpha))
    # print(alpha_copy.model_count(), alpha_copy.size(), alpha_manager_copy.global_model_count(alpha_copy))

    # alpha_copy.size(), alpha_manager_copy.global_model_count(alpha))



    # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha), beta.model_count(),
    #       manager.model_count(beta), beta.manager.global_model_count(beta))
    # print(beta.manager.var_count(), beta.manager.count(), beta.manager.size())
    #
    #
    # beta_mng_copy = beta.manager.copy([beta])
    # # beta_mng_copy = beta.manager.copy(beta.manager.var_order())
    # # beta.manager.vtree_minimize(beta.manager.vtree())
    # print("new: ", beta_mng_copy.var_count(), beta_mng_copy.count(), beta_mng_copy.size())
    # beta_mng_copy.minimize()
    # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha) , beta.model_count(), manager.model_count(beta), beta.manager.global_model_count(beta) )
    # print("old ", beta.manager.var_count(), beta.manager.count(),  beta.manager.size())
    # print("new: ", beta_mng_copy.var_count(), beta_mng_copy.count(),  beta_mng_copy.size())

    # vtree = Vtree(var_count, var_order, vtree_type)
    # manager = SddManager.from_vtree(vtree)
    # manager.set_prevent_transformation(prevent=True)
    #
    # print("constructing SDD ... ")
    # a, b, c = [manager.literal(i) for i in range(1, 4)]
    # alpha = (~a & ~b & ~c) | (~a & b & ~c) | (a & b & c) | (a & ~b & c)
    # v = manager.get_vars(3)
    # beta = manager.conjoin(alpha, v)
    # print(alpha.model_count(), manager.model_count(alpha), manager.global_model_count(alpha), beta.model_count(),
    #       manager.model_count(beta), beta.manager.global_model_count(beta))
    # print(beta.manager.var_count(), beta.manager.count(), beta.manager.size())
