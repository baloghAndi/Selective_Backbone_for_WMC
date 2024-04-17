from pysdd.sdd import SddManager, Vtree

from CNFmodelSDD import CNF
from CSP import PartialAssigment
import numpy as np
import itertools
import math
from array import array
import time
import os


SOLUTIONS = {
        1:  [1, 1, 1, 0, 1, 0, 1, 0, 1],
        2:  [1, 1, 0, 0, 0, 0, 1, 1, 1],
        3:  [0, 1, 1, 0, 1, 0, 1, 0, 0],
        4:  [1, 0, 0, 0, 1, 0, 1, 0, 0],
        5:  [0, 0, 1, 1, 0, 0, 1, 1, 0],
        6:  [0, 0, 0, 0, 1, 0, 1, 0, 0],
        7:  [1, 0, 1, 0, 1, 1, 1, 0, 1],
        8:  [0, 0, 0, 1, 0, 1, 1, 0, 1],
        9:  [1, 0, 1, 1, 0, 1, 1, 0, 1],
        10: [1, 0, 1, 1, 0, 1, 1, 1, 1]

        }
class WCNF(CNF):

    def __init__(self, logger=None, scalar=0):
        self.variables = {}  # name:domain
        self.literals = []
        self.cls = []
        self.partial_assignment = PartialAssigment()
        self.logger = logger
        self.instance_name = ""
        self.p = 0
        self.n= 0
        self.global_all_solutions = {}
        self.global_all_solution_costs = []
        self.literal_weights = None
        self.weight_file = None
        self.obj_type = ""
        self.scalar = scalar

    def load_file(self, filename, obj_type=None, heur_type=None):

        self.instance_name = filename
        self.weight_file = filename.replace(".cnf", '.w')  # for  non weighted
        if self.scalar > 0:
            self.weight_file = filename.replace(".cnf", "_w" + str(self.scalar) + ".w")  # for weighted

        with open(filename, "r") as f:
            content = f.readlines()
            # remove init comments
            init_nb_lines = 1
            for c in content:
                if c.startswith("c"):
                    init_nb_lines += 1
                    continue
                else:
                    print(c)
                    nb_vars = int(c.strip().split(" ")[2])
                    break
            if nb_vars > 300:
                print("Nb vars more then 300")
                exit(111)
                return False
            # nb_vars = int(content[0].strip().split(" ")[2])
            print("NB VARS", nb_vars)
            # nb_clauses = content[0].strip().split(" ")[3]
            # self.literals = [i for i in range(1, nb_vars + 1)]
            for str_clause in content[init_nb_lines:]:
                if 'c' in str_clause:
                    continue
                else:
                    lits = [int(i) for i in str_clause.strip().split(" ")[:-1] if i != '']
                    if len(lits) == 0:
                        continue
                    self.cls.append(lits)

            self.literals = [i for i in range(1, nb_vars + 1)]
            self.variables = {i: [0, 1] for i in self.literals}
            self.n = len(self.literals)

            if self.literal_weights == None:
                self.literal_weights = {0: nb_vars * [1], 1: nb_vars * [1]}

            # if weight file does not exists take 1 for each weight - will be equivalent to MC
            if os.path.exists(self.weight_file):
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
            # print(self.literal_weights)
            start_time = time.perf_counter()
            if self.logger:
                if self.logger.compile:

                    vtree = Vtree(var_count=nb_vars, vtree_type='balanced')
                    self.sdd_manager = SddManager.from_vtree(vtree)
                    self.sdd_manager.auto_gc_and_minimize_on()
                    cnf_file_name = self.instance_name.replace("_wcnf", ".cnf")
                    self.write_cnf(cnf_file_name)
                    print("compiling...")
                    self.root_node = self.sdd_manager.read_cnf_file(cnf_file_name.encode('utf-8'))
                    print("compiled to sdd")
                    self.root_node.model_count()
                    self.sdd_manager.auto_gc_and_minimize_off()
                    wmc = self.weighted_model_count(self.root_node)
                    print("WMC: ", wmc)
                    # self.count_individual_model_weights(self.root_node)
                    logWMC = math.log10(wmc)
                    load_time = time.perf_counter() - start_time
                    self.logger.log( [0, "-1", "-1", self.n, len(self.cls), self.sdd_manager.global_model_count(self.root_node), self.root_node.size(), self.root_node.count(), load_time, wmc, logWMC])


                # print(self.root_node.model_count())
                # all_solutions, all_costs = self.count_individual_model_weights(self.root_node)
                print("calculated init model weights")
                # self.global_all_solutions = sorted(all_solutions)
                # self.global_all_solution_costs = sorted(all_costs)
                # half_cost = self.global_all_solution_costs[len(self.global_all_solution_costs) // 2]
                # sol_count = 0
                # for c in all_costs:
                #     if c >= half_cost:
                #         sol_count += 1
                # wmc_all_costs = sum(all_costs)
                # if wmc_all_costs != wmc:
                #     print("something wrong", wmc_all_costs, wmc)
                #     exit(9)
                # print("calculated init wmc")
        self.n = len(self.literals)
        return True

    def write_cnf(self,cnf_file_name):
        if not os.path.exists(cnf_file_name):
            f = open(cnf_file_name, "w")
            f.write("p cnf " + str(self.n) + " " + str(len(self.cls)) + "\n")
            for c in self.cls:
                f.write(" ".join([str(x) for x in c]) + " 0 \n")
            f.close()
            return True
        return False

    def write_weights(self, file_name=None):
        if file_name == None:
            file_name = self.weight_file
        if not os.path.exists(file_name):
            f = open(file_name, "w")
            i = 1
            for negl, l  in zip(self.zero_weights, self.one_weights):
                log_neg = math.log(negl)
                log_l = math.log(l)
                f.write(str(i)+" "+str(log_l)+"\n")
                f.write("-"+str(i)+" "+str(log_neg)+"\n")
                i += 1
            f.close()
            return True
        return False

    def create_example(self, literal_weights ):

        example_solutions = [
            [1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 1]
        ]


        cnf = CNF()
        var_count = 9
        self.n = var_count
        self.literals = [1,2,3,4,5,6,7,8,9]
        var_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        vtree_type = "right"

        vtree = Vtree(var_count, var_order, vtree_type)
        self.sdd_manager = SddManager.from_vtree(vtree)

        a,b,c,d,e,f,g,h,i = [ self.sdd_manager.literal(i) for i in range(1, 10)]
        self.root_node = (( a & b & c & ~d & e & ~f & g & ~h & i ) | ( a & b & ~c & ~d & ~e & ~f & g & h & i ) | (~a & b & c & ~d & e & ~f & g & ~h & ~i)
                          | (a & ~b & ~c & ~d & e & ~f & g & ~h & ~i)  | (~a & ~b & c & d & ~e & ~f & g & h & ~i) | (~a & ~b & ~c & ~d & e & ~f & g & ~h & ~i)
                          | (a & ~b & c & ~d & e & f & g & ~h & i)  | (~a & ~b & ~c & d & ~e & f & g & ~h & i) | (a & ~b & c & d & ~e & f & g & ~h & i)
                          | (a & ~b & c & d & ~e & f & g & h & i) )
        print(self.root_node.model_count())

        self.variables = {1: [0,1], 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 9: [0,1]}

        # for model in self.root_node.models():
        #     m = [model[i] for i in range(1, 10)]
        #     print(m)
        #     print(m, example_solutions[sol_index])
        #     if m not in example_solutions:
        #         print("error")
        self.set_weights(literal_weights)

        all_solutions, all_costs = self.count_individual_model_weights(self.root_node)
        self.global_all_solutions = sorted(all_solutions)
        self.global_all_solution_costs = sorted(all_costs)
        half_cost = self.global_all_solution_costs[len(self.global_all_solution_costs) // 2]
        sol_count = 0
        for c in all_costs:
            if c >= half_cost:
                sol_count += 1
        cost_sum = sum(all_costs)



        wmc = self.weighted_model_count(self.root_node)
        if cost_sum != wmc:
            print("something wrong")
            exit(9)

        if self.logger != None:
            self.logger.log([0, "-1", "-1", self.n, len(self.cls), self.root_node.model_count(), self.root_node.size(), self.root_node.count(), -1, wmc, sol_count])


    def print_weighted_solutions(self, node, literal_weights):

        """
        1:  51 52 53 54 55 56 57 58 59
        0:  49 48 47 46 45 44 43 42 41
        """
        node.ref()
        self.sdd_manager.minimize()

        node.manager.auto_gc_and_minimize_off()
        self.wmc = node.wmc(log_mode=False)

        self.one_weights = literal_weights[1]
        self.zero_weights = literal_weights[0]

        # one_weights = [51, 52, 53, 54, 55, 56, 57, 58, 59]
        # zero_weights = [49, 48, 47, 46, 45, 44, 43, 42, 41]

        for i in range(1,10):
            l = node.manager.literal(i)
            self.wmc.set_literal_weight(l,  self.one_weights[i-1])  # Set the required value for literal a to 1
            self.wmc.set_literal_weight(-l,  self.zero_weights[i-1]) # Set the complement to 0
        print([self.wmc.literal_weight(node.manager.literal(j)) for j in range(1, 10)])
        print([self.wmc.literal_weight(~node.manager.literal(j)) for j in range(1, 10)])
            # print(int(math.exp(wmc.propagate())))
            # print(self.depth_first_normal(self.root_node))
        print(node.model_count(), node.elements())
        # print(int(self.wmc .propagate()))
        # print(int(self.depth_first_normal(self.root_node, 0)))
        all_solutions = {}

        print("saving sdd and vtree ... ", end="")
        with open("./sdd_condition_node.dot", "w") as out:
            print(node.dot(), file=out)
        # with open("./vtree.dot", "w") as out:
        #     print(self.sdd_manager.vtree.dot(), file=out)
        print("done")
        to_visit = [node]
        self.weighted_model_count(node, 0, [],to_visit, all_solutions)
        print(all_solutions)
        all_solutions_weights = sorted(all_solutions)
        for c in all_solutions_weights:
            # print(c, all_solutions[c])
            for sol in all_solutions[c]:
                s = [1 if i in sol else 0 for i in range(1, self.n+1)]
                print(c,[key for key, temp in SOLUTIONS.items() if temp == s] , s)
        node.deref()
        return all_solutions

    def set_weights(self, f2_literal_weights):
        self.zero_weights = f2_literal_weights[0]
        self.one_weights = f2_literal_weights[1]


    # def weighted_model_count(self, node):
    #     models = node.models()
    #     all_costs =[]
    #     all_solutions = {}
    #     for m in models:
    #         solution= [m[i] for i in range(1, len(m)+1)]
    #         cost = math.prod([self.one_weights[i-1] if m[i] == 1  else self.zero_weights[i-1] for i in range(1,len(m)+1) ])
    #         print(solution, cost)
    #         all_costs.append(cost)
    #         if cost in all_solutions:
    #             all_solutions[cost].append(solution)
    #         else:
    #             all_solutions[cost] = [solution]
    #     return all_solutions, all_costs

    def count_individual_model_weights(self, node):
        models = node.models()
        all_costs =[]
        all_solutions = {}
        total_cost = 0
        for m in models:
            solution= [m[i] for i in range(1, len(m)+1)]
            cost = math.prod([self.literal_weights[1][i-1] if m[i] == 1  else self.literal_weights[0][i-1] for i in range(1,len(m)+1) ])
            all_costs.append(cost)
            total_cost += cost
            if cost in all_solutions:
                all_solutions[cost].append(solution)
            else:
                all_solutions[cost] = [solution]
        print("total WMC calculated: ", total_cost, len(all_costs), self.sdd_manager.global_model_count(self.root_node))
        return all_solutions, all_costs

    def weighted_model_count(self, node):
        # print("saving sdd and vtree ... ")
        # with open("sdd.dot", "w") as out:
        #     print(node.dot(), file=out)
        # with open("vtree.dot", "w") as out:
        #     print(node.manager.vtree().dot(), file=out)
        # print("done")

        self.wmc = node.wmc(log_mode=False)
        for i in range(1, self.n+1):
            l = self.root_node.manager.literal(i)
            self.wmc.set_literal_weight(l, self.literal_weights[1][i - 1])  # Set the required value for literal a to 1
            self.wmc.set_literal_weight(-l, self.literal_weights[0][i - 1])  # Set the complement to 0
        # print(self.one_weights, self.zero_weights)
        scaled_wmc = self.wmc.propagate()
        # w_cost = scaled_wmc / math.pow(2, self.n)

        return scaled_wmc

    def extend_assignment(self, var, value, score, temp_root):
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[var] = value
        self.partial_assignment.score = score
        self.root_node = temp_root #if we update to the conditioning node

        # if self.root_node.model_count() != 0:
        #     all_solutions, all_costs = self.count_individual_model_weights(self.root_node)
        #     self.global_all_solutions = sorted(all_solutions)
        #     self.global_all_solution_costs = sorted(all_costs)

    def check_wmc_of(self, var, value):
        #LOOK OUT : sdd removes node if its values are interchangeable, looking at global to include both solutions, conditioning doesn't work
        # this way of conjoining  adds another node
        if value == 0:
            var = -var
        self.sdd_manager.auto_gc_and_minimize_off()
        condition_node = self.root_node & self.sdd_manager.get_vars(var) #same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))
        # print("condition ----- ", condition_node.model_count(), condition_node.manager.global_model_count(condition_node))
        # print(condition_node.size(), condition_node.count(), self.root_node.count(), self.root_node.size())
        if condition_node.model_count() == 0:
            # print("0 cost")
            return 0, condition_node.size(), condition_node.count(), condition_node
        # print([m for m in condition_node.models()])
        # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
        # all_solutions = self.print_weighted_solutions(condition_node, f2_literal_weights)
        # self.set_weights(f2_literal_weights)
        # all_solutions, all_costs = self.weighted_model_count(condition_node)
        wmc = self.weighted_model_count(condition_node)
        # print("MC: ", condition_node.manager.global_model_count(condition_node), condition_node.size(), condition_node.count(), condition_node)
        # cost_sum = sum([ cost*len(vals) for cost,vals in all_solutions.items()])
        # print("cost: ", wmc)
        return wmc, condition_node.size(), condition_node.count(), condition_node #return wmc

    def check_wmc_ratio_of(self, var, value):
        #LOOK OUT : sdd removes node if its values are interchangeable, looking at global to include both solutions, conditioning doesn't work
        # this way of conjoining  adds another node
        if value == 0:
            var = -var
        self.sdd_manager.auto_gc_and_minimize_off()
        condition_node = self.root_node & self.sdd_manager.get_vars(var) #same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))
        if condition_node.model_count() == 0:
            return 0, condition_node.size(), condition_node.count(), condition_node
        # all_solutions, all_costs = self.weighted_model_count(condition_node)
        # cost_sum = sum([ cost*len(vals) for cost,vals in all_solutions.items()])
        wmc = self.weighted_model_count(condition_node)

        ratio = wmc / condition_node.size()
        return ratio, condition_node.size(), condition_node.count(), condition_node

    # def calculate_g2(self, variable, value):
    #     if value == 0:
    #         variable = -variable
    #     self.sdd_manager.auto_gc_and_minimize_off()
    #     condition_node = self.root_node & self.sdd_manager.get_vars(variable)  # same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))
    #     if condition_node.manager.model_count(condition_node) != condition_node.model_count():
    #         print("DIFFERENT")
    #         exit(888)
    #     if condition_node.model_count() == 0:
    #         return 0, condition_node.size(), condition_node.count(), condition_node
    #     # wmc = self.weighted_model_count(condition_node)
    #
    #     all_solutions, all_costs = self.count_individual_model_weights(condition_node)
    #     half_cost = self.global_all_solution_costs[len(self.global_all_solution_costs) // 2]
    #     print("half cost: ", half_cost)
    #     sol_count = 0
    #     for c in all_costs:
    #         if c >= half_cost:
    #             sol_count += 1
    #     return sol_count, condition_node.size(), condition_node.count(), condition_node

    def depth_first_normal(self, node):
        #TODO this does not extend T nodes and only takes that solution into account once - so wmc will not be correct
        """Depth first search to compute the WMC.

        This does not yet perform smoothing!
        """
        print(node)
        if node.is_decision():
            rvalue = 0
            for prime, sub in node.elements():
                # Conjunction
                result = self.depth_first_normal(prime) * self.depth_first_normal(sub)
                # Disjunction
                rvalue += result
        elif node.is_true():
            rvalue = 1
        elif node.is_false():
            rvalue = 0
        elif node.is_literal():
            rvalue = self.wmc.literal_weight(node)
        else:
            raise Exception(f"Unknown node type: {node}")
        print("---", node, rvalue)
        return rvalue

class PartialAssigment:
    def __init__(self):
        self.score = 0
        self.assigned = {} # dict (variable, value)

if __name__ ==  "__main__":
    wcnf = WCNF()
    # wcnf.load_file("./input/test.cnf")
    wcnf.load_file("./input/wmc2021_track2_public/track2_001.mcc2021_wcnf")
    # mc = wcnf.sdd_manager.global_model_count(wcnf.root_node)
    # print("mc", mc)
    # wcnf.root_node.ref()
    # wcost = wcnf.weighted_model_count(wcnf.root_node)
    # print("wmc : %.9f" % float(wcost))

    filename = wcnf.instance_name.replace("wcnf", "log.w")
    wcnf.write_weights(filename)

    # with open("./sdd_condition_node.dot", "w") as out:
    #     print(wcnf.root_node.dot(), file=out)
    # with open("./vtree.dot", "w") as out:
    #     print(wcnf.sdd_manager.vtree().dot(), file=out)
    # c = 0
    # for m in wcnf.root_node.models():
    #     c+=1
    #     print([m[i] for i in range(1, 6)])
    # print(c)
    # dfwcost = wcnf.depth_first_normal(wcnf.root_node)

    # print(dfwcost, wcost, mc, wcnf.sdd_manager.global_model_count(wcnf.root_node))
    exit(88)
    # wcnf = WCNF()
    # wcnf.create_example()
    # f1_literal_weights = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59] }
    # wcnf.print_weighted_solutions(f1_literal_weights)

    #
    # wcnf = WCNF()
    # wcnf.create_example()
    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2,4,6,8,10,12, 14, 16, 18]}
    # wcnf.print_weighted_solutions(wcnf.root_node, f2_literal_weights)
    # wcnf.check_g1_of(4,1) #-- this kinda works now, check that works when checking multiple conditioning
    # wcnf.print_weighted_solutions(f2_literal_weights)
    # wcnf.check_g1_of(2,0)

    wcnf = WCNF()
    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    f2_literal_weights = {0: [1, 1, 1, 1, 1, 1, 1, 1, 1], 1: [2, 2, 2, 2, 2, 2, 2, 2, 2]}
    # f2_literal_weights = {0: [1, 1, 1, 1, 1, 1, 1, 1, 1], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}

    wcnf.create_example(f2_literal_weights)
    wcnf.root_node.ref()
    wcost = wcnf.weighted_model_count(wcnf.root_node)
    # print("wcost ",wcost)

    wcnf = WCNF()
    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    f2_literal_weights = {0: [1, 1, 1, 1, 1, 1, 1, 1, 1], 1: [2, 2, 2, 2, 2, 2, 2, 2, 2]}
    # f2_literal_weights = {0: [1, 1, 1, 1, 1, 1, 1, 1, 1], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}

    wcnf.create_example(f2_literal_weights)
    wcnf.root_node.ref()
    wcost = wcnf.depth_first_normal(wcnf.root_node)
    # print(wcost)
    # print(f"Model count: {int(math.exp(wcost))}")
    exit(1)

    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2,4,6,8,10,12, 14, 16, 18]}
    # wcnf.print_weighted_solutions(WCNF.root_node, f2_literal_weights)
    # wcnf.check_g1_of(1,0)
    # wcnf.sdd_manager.auto_gc_and_minimize_off()
    # condition_node = wcnf.root_node & wcnf.sdd_manager.get_vars(1)
    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # all_solutions = wcnf.print_weighted_solutions(condition_node, f2_literal_weights)


    wcnf.sdd_manager.auto_gc_and_minimize_off()
    # condition_node = wcnf.root_node & wcnf.sdd_manager.get_vars(-1)
    # wcnf.set_weights(f2_literal_weights)
    # sol_count, size, count, condition_node = wcnf.calculate_g2(7,1)
    # print("-----------------------------------")
    # print( sol_count, size, count, condition_node )
    # sol_count, size, count, condition_node = wcnf.calculate_g2(2, 0)
    # print("-----------------------------------")
    # print(sol_count, size, count, condition_node)
    # all_solutions, all_costs = wcnf.weighted_model_count(wcnf.root_node)
    # all_solutions = wcnf.print_weighted_solutions(condition_node, f2_literal_weights)

    # all_solutions_weights = sorted(all_solutions)
    # for c in all_solutions_weights:
    #     # print(c, all_solutions[c])
    #     for s in all_solutions[c]:
    #         # s = [1 if i in sol else 0 for i in range(1, wcnf.n+1)]
    #         print(c,[key for key, temp in SOLUTIONS.items() if temp == s] , s)

    vtree = Vtree.from_file(bytes(here / "input" / "simple.vtree"))
    sdd = SddManager.from_vtree(vtree)
    print(f"Created an SDD with {sdd.var_count()} variables")
    root = sdd.read_cnf_file(bytes(here / "input" / "simple.cnf"))
    # For DNF functions use `read_dnf_file`
    # If the vtree is not given, you can also use 'from_cnf_file`

    # Model Counting
    wmc = root.wmc(log_mode=True)
    w = wmc.propagate()
    print(f"Model count: {int(math.exp(w))}")

    # Weighted Model Counting
    lits = [None] + [sdd.literal(i) for i in range(1, sdd.var_count() + 1)]
    # Positive literal weight
    wmc.set_literal_weight(lits[1], math.log(0.5))
    # Negative literal weight
    wmc.set_literal_weight(-lits[1], math.log(0.5))
    w = wmc.propagate()
    print(f"Weighted model count: {math.exp(w)}")
