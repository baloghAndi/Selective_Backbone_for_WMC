import CSP
from pysdd.sdd import SddManager, Vtree
import time
import graphviz
import os

class CNF:
    def __init__(self,logger=None):
        self.variables = {}  # name:domain
        self.literals = []
        self.cls = []
        self.partial_assignment = PartialAssigment()
        self.logger = logger
        self.instance_name = ""
        self.p = 0

    def load_file(self, filename):
        self.instance_name = filename
        with open(filename, "r") as f:
            content = f.readlines()
            nb_vars = int(content[0].strip().split(" ")[2])
            print(nb_vars)
            if nb_vars > 600:
                return False
            nb_vars = int(content[0].strip().split(" ")[2])
            print("NB VARS", nb_vars)
            if nb_vars > 600:
                return False
            nb_clauses = content[0].strip().split(" ")[3]
            self.literals = [i for i in range(1, nb_vars + 1)]
            for str_clause in content[1:]:
                if 'c' in str_clause:
                    continue
                # str_clause.replace("-","~")
                lits = [int(i) for i in str_clause.strip().split(" ")[:-1] if i != '']
                if len(lits) == 0:
                    continue
                self.cls.append(lits)
            print("finished reading")
            self.variables = {i: [0, 1] for i in self.literals}
            self.n = len(self.literals)

            if self.logger:
                if self.logger.compile:
                    start_time = time.perf_counter()
                    vtree = Vtree(var_count=nb_vars, vtree_type='right')
                    self.sdd_manager = SddManager.from_vtree(vtree)
                    self.sdd_manager.auto_gc_and_minimize_on()
                    nb_clauses = content[0].strip().split(" ")[3]
                    self.root_node = self.sdd_manager.read_cnf_file(filename.encode('utf-8'))
                    c = self.root_node.model_count()
                    load_time = time.perf_counter() - start_time
                    # columns = ["p", "var", "value", n, nb cls, "MC", "SDD_size", 'node_count', 'time']
                    self.logger.log(
                        [0, "-1", "-1", self.n, len(self.cls),  c, self.root_node.size(), self.root_node.count(), load_time])
                else:
                    self.logger.log([0, "-1", "-1", self.n, len(self.cls), "-1", "-1", "-1", "-1"])
        self.n = len(self.literals)
        return True


    def check_mc_of(self, var, value):
        #LOOK OUT : sdd removes node if its values are interchangeable, looking at global to include both solutions, conditioning doesn't work
        # this way of conjoining  adds another node
        if value == 0:
            var = -var
        self.sdd_manager.auto_gc_and_minimize_off()
        condition_node = self.root_node & self.sdd_manager.get_vars(var) #same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))

        #!!!! use ref if minimizing vtree
        # print(self.sdd_manager.get_vars(var))
        # condition_node = self.root_node & self.sdd_manager.get_vars(var) #same as managet.conjoin( self.root_node ,self.sdd_manager.get_vars(var))
        # print("condition_node ", var, value, condition_node.manager.global_model_count(condition_node), condition_node.model_count(), condition_node.size(), condition_node.count(), condition_node)
        # condition_node.ref()
        # condition_node.manager.minimize()
        # condition_node = self.root_node.condition(var)

        # condition_node = self.root_node.conjoin(self.sdd_manager.literal(var))
        # print("score_of_assignment, size, node_count, root count ", condition_node.model_count(), condition_node.size(), condition_node.count(), condition_node.node_size(), condition_node.manager.global_model_count(condition_node))
        # self.sdd_manager.auto_gc_and_minimize_on()
        # condition_node.manager.minimize()

        # print(sdd_copy.root_node.size(), sdd_copy.sdd_manager.global_model_count(sdd_copy.root_node))
        # condition_node_copy = sdd_copy.root_node & sdd_copy.sdd_manager.get_vars(var)
        # print("condition_node_copy ", condition_node_copy.manager.global_model_count(condition_node_copy), condition_node_copy.model_count(), condition_node_copy.size(), condition_node_copy.count(), condition_node_copy)

        return condition_node.manager.global_model_count(condition_node), condition_node.size(), condition_node.count(), condition_node


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


    def extend_assignment(self, var, value, score, temp_root):
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[var] = value
        self.partial_assignment.score = score
        self.root_node = temp_root #if we update to the conditioning node
        # if len( self.partial_assignment.assigned) >= 9:
            # zdd = _zdd.ZDD()
            # zdd.declare(*self.literals)
            # v = self.bdd.copy(self.root_node, zdd)
            # zdd.dump("./DatasetA/bdd"+str(len( self.partial_assignment.assigned))+".png" ,[v])
            # self.bdd.dump("./DatasetA/bdd"+str(len( self.partial_assignment.assigned))+".png" ,[self.root_node])

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


class PartialAssigment:
    def __init__(self):
        self.score = 0
        self.assigned = {} # dict (variable, value)

if __name__ == "__main__":
    vtree = Vtree(var_count=120, vtree_type='balanced')
    sdd_manager = SddManager.from_vtree(vtree)
    sdd_manager.auto_gc_and_minimize_off()
    print("reading...")
    root_node = sdd_manager.read_cnf_file("./input/wmc2020_track2_all/track2_005.mcc2020.cnf".encode('utf-8'))
    print("mc ", root_node.model_count())
    exit(100)
    # cnf =CNF()
    # cnf.load_file("../nqueens_4_modified.cnf")
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