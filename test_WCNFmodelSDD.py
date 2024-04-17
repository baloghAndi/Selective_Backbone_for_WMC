from pysdd.sdd import SddManager, Vtree

from CNFmodelSDD import CNF
from CSP import PartialAssigment
import numpy as np
import itertools
import math
from array import array


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

    def __init__(self, logger=None):
        self.variables = {}  # name:domain
        self.literals = []
        self.cls = []
        self.partial_assignment = PartialAssigment()
        self.logger = logger
        self.instance_name = ""
        self.p = 0
        self.n= 0

    def create_example(self):

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
        var_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        vtree_type = "balanced"

        vtree = Vtree(var_count, var_order, vtree_type)
        self.sdd_manager = SddManager.from_vtree(vtree)

        a,b,c,d,e,f,g,h,i = [ self.sdd_manager.literal(i) for i in range(1, 10)]
        self.root_node = (( a & b & c & ~d & e & ~f & g & ~h & i ) | ( a & b & ~c & ~d & ~e & ~f & g & h & i ) | (~a & b & c & ~d & e & ~f & g & ~h & ~i)
                         | (a & ~b & ~c & ~d & e & ~f & g & ~h & ~i)  | (~a & ~b & c & d & ~e & ~f & g & h & ~i) | (~a & ~b & ~c & ~d & e & ~f & g & ~h & ~i)
                         | (a & ~b & c & ~d & e & f & g & ~h & i)  | (~a & ~b & ~c & d & ~e & f & g & ~h & i) | (a & ~b & c & d & ~e & f & g & ~h & i)
                         | (a & ~b & c & d & ~e & f & g & h & i) )
        print(self.root_node.model_count())

        for model in self.root_node.models():
            m = [model[i] for i in range(1, 10)]
            print(m)
            # print(m, example_solutions[sol_index])
            if m not in example_solutions:
                print("error")

    def print_weighted_solutions(self, literal_weights):

        """
        1:  51 52 53 54 55 56 57 58 59
        0:  49 48 47 46 45 44 43 42 41
        """
        self.root_node.ref()
        self.sdd_manager.minimize()

        self.sdd_manager.auto_gc_and_minimize_off()
        self.wmc = self.root_node.wmc(log_mode=False)

        one_weights = literal_weights[1]
        zero_weights = literal_weights[0]

        # one_weights = [51, 52, 53, 54, 55, 56, 57, 58, 59]
        # zero_weights = [49, 48, 47, 46, 45, 44, 43, 42, 41]

        for i in range(1,10):
            l = self.sdd_manager.literal(i)
            self.wmc .set_literal_weight(l, one_weights[i-1])  # Set the required value for literal a to 1
            self.wmc .set_literal_weight(-l, zero_weights[i-1]) # Set the complement to 0
        print([self.wmc.literal_weight(self.sdd_manager.literal(j)) for j in range(1, 10)])
        print([self.wmc.literal_weight(~self.sdd_manager.literal(j)) for j in range(1, 10)])
            # print(int(math.exp(wmc.propagate())))
            # print(self.depth_first_normal(self.root_node))
        print(int(self.wmc .propagate()))
        # print(int(self.depth_first_normal(self.root_node, 0)))
        all_solutions = {}
        self.weighted_model_count(self.root_node, 0, [], all_solutions)
        print(all_solutions)
        all_solutions_weights = sorted(all_solutions)
        for c in all_solutions_weights:
            # print(c, all_solutions[c])
            for sol in all_solutions[c]:
                s = [1 if i in sol else 0 for i in range(1, self.n+1)]
                print(c,[key for key, temp in SOLUTIONS.items() if temp == s] , s)

        #
        #     print(i, l,  one_weights[i-1],  zero_weights[i-1],  int(wmc.propagate()))
        # print(int(self.wmc.propagate()))



    def weighted_model_count(self, node, cost, path, all_solutions):
        """Depth first search to compute the WMC.

        This does not yet perform smoothing!
        """
        # print("NODE: ", node, [p.literal for p in path])
        # !!! in last solution node A is not in sdd - both A and -A are solutions so its skipped - > one less solution and wrong solution count
        if node.is_decision():
            for prime, sub in node.elements():
                p =  self.weighted_model_count(prime, cost, path, all_solutions)
                cost+=p
                s =  self.weighted_model_count(sub, cost, path, all_solutions)
                if s != None:
                    cost += s
                    #save solution, first check if all variables have been visited, if not add missing variable, creating two solutions, 1with 0 and 1 with 1 weight
                    current_solution = [p.literal for p in path]
                    if len(current_solution) < self.n:
                        #find missing literal
                        for lit in range(1,self.n+1):
                            if lit not in current_solution and -lit not in current_solution:
                                tcost = cost + self.wmc.literal_weight(lit)
                                tcurrent_solution = current_solution.copy() + [lit]
                                if int(tcost) in all_solutions:
                                    (all_solutions[int(tcost)].append(tcurrent_solution))
                                else:
                                    all_solutions[int(tcost)] = [tcurrent_solution]

                                tcost = cost + self.wmc.literal_weight(-lit)
                                tcurrent_solution = current_solution.copy() + [-lit]
                                if int(tcost) in all_solutions:
                                    (all_solutions[int(tcost)].append(tcurrent_solution))
                                else:
                                    all_solutions[int(tcost)] = [tcurrent_solution]
                    else:
                        if int(cost) in all_solutions:
                            (all_solutions[int(cost)].append(current_solution))
                        else:
                            all_solutions[int(cost)] = [current_solution]
                    path.remove(prime)
                    path.remove(sub)
                else:
                    cost= cost -p
                    path.remove(prime)
                # if p != None and s!= None:
                #     print("+++++++++++++++++++++++++++++++++++++solution", cost,[p.literal for p in path])
                    # if int(cost) in all_solutions:
                    #     (all_solutions[int(cost)].append([p.literal for p in path]))
                    # else:
                    #     all_solutions[int(cost)] = [[p.literal for p in path]]
                    # path.remove(prime)
                    # path.remove(sub)

            # else: #disjunction
            #     self.explore(node.elements()[0])
            #     self.explore(node.elements()[1])
        elif node.is_true():
            # print("Found solution" , self.wmc.one_weight)
            # any further node assignment is acceptalbe but I assume this won't happen often ( actually in case of backbones thsat will show up such way?)
            # print("true")
            return 0
        elif node.is_false():
            #return null, this path does not lead to solutions
            # print("none")
            return None
        elif node.is_literal():
            # print(node, self.wmc.literal_weight(node))
            rvalue =  self.wmc.literal_weight(node)
            path.append(node)
            # print("rvalue")
            return rvalue
        else:
            raise Exception(f"Unknown node type: {node}")


if __name__ ==  "__main__":
    wcnf = WCNF()
    wcnf.create_example()
    f1_literal_weights = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59] }
    wcnf.print_weighted_solutions(f1_literal_weights)


    wcnf = WCNF()
    wcnf.create_example()
    f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2,4,6,8,10,12, 14, 16, 18]}
    wcnf.print_weighted_solutions(f2_literal_weights)




