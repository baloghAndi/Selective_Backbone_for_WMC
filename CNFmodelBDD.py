import CSP
# from dd import bdd as _bdd
from dd import cudd as _bdd
# from dd import cudd_zdd as _zdd
from omega.logic.syntax import conj, disj
# from pysdd.sdd import SddManager, Vtree
# import graphviz
import os

class CNF:
    def __init__(self,logger=None):
        self.variables = {}  # name:domain
        self.literals = []
        self.cls = []
        self.bdd = _bdd.BDD()
        self.bdd.configure(reordering=True)
        self.partial_assignment = PartialAssigment()
        self.logger = logger
        self.instance_name = ""

    # def load_file(self, filename):
    #     self.instance_name = filename
    #     with open(filename, "r") as f:
    #         content = f.readlines()
    #         nb_vars = int(content[0].strip().split(" ")[2])
    #
    #         if nb_vars > 300:
    #             return False
    #         nb_clauses = content[0].strip().split(" ")[3]
    #         self.literals = [self.str_lit(i) for i in range(1,nb_vars+1)]
    #         self.bdd.declare(*self.literals)
    #         for str_clause in content[1:]:
    #             if 'c' in str_clause:
    #                 continue
    #             # str_clause.replace("-","~")
    #             lits = [self.str_lit(i) for i in str_clause.strip().split(" ")[:-1] if i != '' ]
    #             if len(lits) == 0:
    #                 continue
    #             # print(lits)
    #             s = disj(lits)
    #             print(s)
    #             self.cls.append(s)
    #         print("finished reading")
    #         s = conj(self.cls)
    #         print(s)
    #         self.root_node = self.bdd.add_expr(s)
    #         print("created root")
    #         # cn = self.bdd.count(self.root_node )
    #         # print("Model count ", cn)
    #         self.levels_nb = len(self.bdd.var_levels)
    #         self.variables = {i :[0,1] for i in self.literals}
    #         print("before mc")
    #         c = self.bdd.count(self.root_node)
    #         print(" MC: ",  c)
    #         if self.logger:
    #             stats = self.bdd.statistics()
    #             # print(len(self.literals), len(self.cls), print(self.cls[-1]))
    #             cnf_load_time = self.logger.get_time_elapsed()
    #             self.logger.log([0, "-1", "-1", int(c),len(self.bdd), stats['n_vars'], stats['n_nodes'], stats['n_reorderings'], self.root_node.dag_size, cnf_load_time ])
    #             bdd_file = self.instance_name.replace(".cnf", ".dddmp")
    #             self.bdd.dump(self.root_node, bdd_file)
    #     self.n = len(self.literals)
    #     return True

    def load_file_with_apply(self, filename):
        self.instance_name = filename
        with open(filename, "r") as f:
            content = f.readlines()
            nb_vars = int(content[0].strip().split(" ")[2])
            print("NB VARS", nb_vars)
            if nb_vars > 600:
                return False
            nb_clauses = content[0].strip().split(" ")[3]
            self.literals = [self.str_lit(i) for i in range(1, nb_vars + 1)]
            self.bdd.declare(*self.literals)
            for str_clause in content[1:]:
                if 'c' in str_clause:
                    continue
                # str_clause.replace("-","~")
                lits = [self.str_lit(i) for i in str_clause.strip().split(" ")[:-1] if i != '']
                if len(lits) == 0:
                    continue
                s = r' \/ '.join(var for var in lits)
                cls_node = self.bdd.add_expr(s)
                # print(self.root_node.dag_size)
                # _bdd.reorder(self.bdd)
                # print(self.root_node.dag_size)
                k = len(self.cls)
                print(k)
                if len(self.cls) == 0:
                    self.root_node = cls_node
                else:
                    self.root_node = self.bdd.apply("and",self.root_node, cls_node)
                # print(cls_node, len(self.cls), self.root_node.dag_size)
                self.cls.append(s)

                # print("B", self.root_node.dag_size)
                _bdd.reorder(self.bdd)
                # print(self.bdd.var_levels)
                # print("A",self.root_node.dag_size)
            print("finished reading")
            # s = conj(self.cls)
            # print(s)
            # self.root_node = self.bdd.add_expr(s)
            print("created root")
            # cn = self.bdd.count(self.root_node )
            # print("Model count ", cn)
            self.levels_nb = len(self.bdd.var_levels)
            self.variables = {i: [0, 1] for i in self.literals}
            # print("before mc")
            # c = self.bdd.count(self.root_node)
            self.n = len(self.literals)
            c = self.root_node.count(self.n)
            # print(" MC: ", c)
            if self.logger:
                stats = self.bdd.statistics()
                # print(len(self.literals), len(self.cls), print(self.cls[-1]))
                self.load_time = self.logger.get_time_elapsed()
                self.logger.log( [0, "-1", "-1",stats['n_vars'], int(c), len(self.bdd), stats['n_nodes'], stats['n_reorderings'], self.root_node.dag_size, self.load_time])
                bdd_file = self.instance_name.replace(".cnf", ".dddmp")
                print(bdd_file)
                self.bdd.dump(bdd_file, [self.root_node])
        return True

    def str_lit(self,i):
        i = int(i)
        if i < 0 :
            return "~x"+str(abs(i))
        return "x"+str(i)

    def check_score_of_recompile(self, var, value):
        "Deprecated: this recompiled the bdd from scratch"
        if value == 0:
            var = "~"+var
        temp_cls = self.cls + [var]
        s = conj(temp_cls)
        bdd = _bdd.BDD()
        bdd.configure(reordering=True)
        bdd.declare(*self.literals)
        root_node = bdd.add_expr(s)
        c = bdd.count(root_node, len(self.literals)) #TODO: without len of literals count may return smaller numbers, where vars that can have both values are ommited => smaller BDD
        m = bdd.pick_iter(root_node, care_vars=self.literals)
        count = 0
        for x in m:
            # print(x)
            count += 1
        # print("New MC: ", var, value, c, count)
        if c != count:
            print("why?")
            exit(-19)
        return int(c), bdd, root_node

    def check_mc_of(self, var, value):
        bdd_var = self.bdd.var(var)
        if value == 0:
            var = "~" + var
            bdd_var = self.bdd.apply('not', bdd_var)
        # print(self.bdd.var_levels)
        temp_root = self.bdd.apply('and', self.root_node, bdd_var)
        # c = self.bdd.count(temp_root, len(self.literals))  # TODO: without len of literals count may return smaller numbers, where vars that can have both values are ommited => smaller BDD
        # m = self.bdd.pick_iter(temp_root, care_vars=self.literals)
        # count = 0
        # for x in m:
        #     # print(x)
        #     count += 1
        # # print("New MC: ", var, value, c, count,temp_root.count(len(self.literals)) )
        # if c != count or c!= int(temp_root.count(len(self.literals))):
        #     print("why?")
        #     exit(-19)
        return int(temp_root.count(self.n)), self.bdd, temp_root

    def check_mc_bdd_ratio_of(self, var, value):
        bdd_var = self.bdd.var(var)
        if value == 0:
            # var = "~" + var
            bdd_var = self.bdd.apply('not', bdd_var)
        temp_root = self.bdd.apply('and', self.root_node, bdd_var)
        r = temp_root.count(self.n)/temp_root.dag_size
        # print(var, value, r, temp_root.count(self.n), temp_root.dag_size)
        return r, self.bdd, temp_root

    def check_score_of_assignments(self, assign):
        temp_root = self.root_node
        for var, value in assign.items():
            bdd_var = self.bdd.var(var)
            if value == 0:
                var = "~" + var
                bdd_var = self.bdd.apply('not', bdd_var)
            temp_root = self.bdd.apply('and', temp_root, bdd_var)

        return int(temp_root.count(self.n)), self.bdd, temp_root

    def get_best_wrt_assignment(self, assign):
        assignment_root = self.root_node
        for var, value in assign.items():
            bdd_var = self.bdd.var(var)
            if value == 0:
                var = "~" + var
                bdd_var = self.bdd.apply('not', bdd_var)
            assignment_root = self.bdd.apply('and', assignment_root, bdd_var)

        # print("init", assign, "MC:", assignment_root.count(self.n), "BDD:", self.root_node.dag_size, assignment_root.dag_size)

        best_variable = 0
        best_value = 0
        best_mc = 0
        current_root = assignment_root
        current_mc = assignment_root.count(self.n)
        for v in self.variables.keys():
            if v not in assign:
                for value in self.variables[v]:

                    bdd_var = self.bdd.var(v)
                    if value == 0:
                        # var = "~" + v
                        bdd_var = self.bdd.apply('not', bdd_var)
                    current_root = self.bdd.apply('and', assignment_root, bdd_var)
                    current_mc = current_root.count(self.n)
                    # print(v, value, current_mc, current_root.dag_size)
                    if current_mc >= best_mc:
                        best_variable = v
                        best_value = value
                        best_mc = current_mc
                        best_root = current_root
                        # print("best", best_variable , best_value, best_mc, current_root.dag_size )
        # print(best_cost,best_bdd.statistics())
        stats = self.bdd.statistics()
        stats['dag_size'] = best_root.dag_size

        return best_variable,best_value, int(best_root.count(self.n)), self.bdd, self.bdd.statistics(), best_root

    def extend_assignment(self, var, value, score, temp_root):
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[var] = value
        if value == 0:
            var = "~" + var
        self.cls.append(var)
        self.partial_assignment.score = score
        self.root_node = temp_root
        _bdd.reorder(self.bdd)

        # if len( self.partial_assignment.assigned) >= 9:
            # zdd = _zdd.ZDD()
            # zdd.declare(*self.literals)
            # v = self.bdd.copy(self.root_node, zdd)
            # zdd.dump("./DatasetA/bdd"+str(len( self.partial_assignment.assigned))+".png" ,[v])
            # self.bdd.dump("./DatasetA/bdd"+str(len( self.partial_assignment.assigned))+".png" ,[self.root_node])




class PartialAssigment:
    def __init__(self):
        self.score = 0
        self.assigned = {} # dict (variable, value)

if __name__ == "__main__":
    f = "./input/wmc2020_track2_all/track2_005.mcc2020.cnf"
    cnf = CNF()
    cnf.load_file_with_apply(f)
    exit(100)

    cnf =CNF()
    cnf.load_file("../nqueens_4.cnf")
    # cnf.root_node.

    print("----------------------------SDD-----------------------------")
    stats = cnf.get_sdd_stats("../nqueens_4.cnf")
    order = stats["order"]
    print(stats)
    print("----------------------------SDD-----------------------------")

    print(cnf.bdd)
    print(cnf.bdd.statistics())
    cn = cnf.bdd.count(cnf.root_node)
    print("Model count ", cn)
    n = len(cnf.bdd.vars)
    levels = [cnf.bdd.var_at_level(level) for level in range(n)]
    print(levels)
    # cnf.bdd.dump("./8queens.png", [cnf.root_node])


    # new_order = {"x"+str(x):i-1 for i,x in enumerate(order)}
    new_order = {'x'+str(x):i for i,x in enumerate(order)}
    # new_order2 = {'x1': 15, 'x2': 1, 'x3': 2, 'x4': 3, 'x5': 4, 'x6': 5, 'x7': 6, 'x8': 7, 'x9': 8, 'x10': 9, 'x11': 10,'x12': 11, 'x13': 12, 'x14': 13, 'x15': 14, 'x16': 0}
    print(new_order)
    cnf.bdd.reorder(new_order)
    print(cnf.bdd)
    print(cnf.bdd.statistics())
    levels = [cnf.bdd.var_at_level(level) for level in range(n)]
    print(levels)
    print(cnf.root_node.count()) #model count from this node
    print(cnf.root_node.dag_size)
    cnf.bdd.dump("./4queens.png", [cnf.root_node])



    exit(11)
    cnf.bdd.dump("./4queens1.png", [cnf.root_node])
    # c = cnf.check_score_of(1,0)

    print("LEN:", len(cnf.bdd))
    # c, bdd, root = cnf.check_score_of('x4',1)
    c, bdd, root = cnf.check_mc_of('x2', 1)
    levels = [bdd.var_at_level(level) for level in range(n)]
    print(levels)
    print(bdd)
    print(bdd.statistics())
    bdd.dump("./4queens2.png", [root])
    print("LEN:", len(bdd))


    # c = cnf.check_score_of('x2', 1)
    # print(c)

