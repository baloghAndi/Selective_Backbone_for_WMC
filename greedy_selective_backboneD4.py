#We need a function to score the current partial assignment plus the addition of an assignment.
# score of assignment
import numpy as np
import itertools
import CNFmodelD4 as _wcnfd4
import os
import evaluate
import re
import queue as _queue
import random
import time
import sys
import math
import utils

def get_best_assignment(csp, obj_type, NO_COMPILE, logger):
    "Assumption is that here we already have the bdd extended with the partial assignment"
    best_variable = -1
    best_value = -1
    best_cost = -1
    best_wmc = -1
    best_mc = -1
    if obj_type == "count":
        best_cost = sys.maxsize
    best_size = -1
    best_node_count = -1
    wmc = -1
    mc = -1
    size = -1
    node_count = -1
    best_weight = 0
    best_tb = 0
    backbone_assigned = False
    assign_queue = _queue.PriorityQueue()
    if len(csp.trivial_backbone) > 0:
        for tb in csp.trivial_backbone:
            if abs(tb) in csp.partial_assignment.assigned:
                print("Trivial backbone already assigned")
                exit(6)
            if tb < 0:
                value = 0
            else:
                value = 1
            variable = abs(tb)
            weight = csp.literal_weights[value][variable - 1]
            if  weight > best_weight:
                best_weight = weight
                best_variable = variable
                best_value = value
                best_cost = weight * 100000
                best_wmc = best_cost
                best_tb = tb
        backbone_assigned = True
        csp.trivial_backbone.remove(best_tb)
        print("backbone assigned")
    else:
        print("search best assigment...")
        # temp_root.ref() # use this if minimizinf vtree
        # csp.sdd_manager.auto_gc_and_minimize_on()
        for v in csp.variables.keys():
            if v not in csp.partial_assignment.assigned:
                for value in csp.variables[v]:
                    if obj_type == "WMC" or obj_type == "MC" or obj_type == "SUB":
                        if obj_type == "MC":
                            nb_nodes, nb_edges, mc, comp_time = csp.check_mc_of(v, value)
                        else:
                            nb_nodes, nb_edges,  wmc, comp_time = csp.check_wmc_of(v, value)
                        size = nb_edges
                        node_count = nb_nodes
                        if obj_type == "WMC":
                            score_of_assignment = wmc
                            if wmc == csp.init_WMC and mc == csp.init_MC: # in case assignment is equal to inital compilation wmc and mc assign it, it cannot be larger then these values
                                return v, value, score_of_assignment, size, node_count, mc, wmc
                        elif obj_type == "MC":
                            score_of_assignment = mc
                        elif obj_type == "SUB":
                            score_of_assignment = wmc - mc
                    elif "score" in obj_type:
                        if obj_type == "wscore_half":
                            score_of_assignment = csp.calculate_score(v, value, "half", weighted=True)
                        elif obj_type == "wscore_occratio":
                            score_of_assignment = csp.calculate_score(v, value, "occratio", weighted=True)
                        elif obj_type == "wscore_adjoccratio":
                            score_of_assignment = csp.calculate_score(v, value, "adjoccratio", weighted=True)
                        elif obj_type == "wscore_estimate":
                            score_of_assignment = csp.calculate_score(v, value, "estimate", weighted=True)
                        elif obj_type == "wscore_otherwg":
                            score_of_assignment = csp.calculate_score(v, value, "otherwg", weighted=True)
                        elif obj_type == "score_estimate":
                            score_of_assignment = csp.calculate_score(v, value, "estimate", weighted=False)
                        # node_count, size, wmc, mc, comp_time = csp.check_wmc_of(v, value)
                    elif obj_type == "hybrid_wmc":
                        score_of_assignment = csp.calculate_score(v, value, "estimate", weighted=True)

                    # elif obj_type == "count":
                    #     opp_score = csp.opposite_occurance(v, value)
                    #     score_of_assignment = opp_score  #because we want the minimal
                    #     #print("score_of_assignment, ",v, value, score_of_assignment)
                    #     size = -1
                    #     node_count = -1
                    # elif obj_type == "g2":
                    #     score_of_assignment, size, node_count, temp_root  = csp.calculate_g2(v, value)
                    # elif "ratio" in obj_type :
                    #     score_of_assignment, size, node_count, temp_root = csp.check_wmc_ratio_of(v, value)
                    else:
                        print("ERROR")
                        exit(666)
                    weight = csp.literal_weights[value][v - 1]
                    assign_queue.put(tuple([-1 * score_of_assignment, -1 * weight, v, value]))

                    if obj_type!="count" :
                        if (score_of_assignment > best_cost) or ( obj_type=="WMC" and score_of_assignment == best_cost and csp.literal_weights[value][v-1] >  csp.literal_weights[best_value][best_variable-1] ):
                            best_variable=v
                            best_value=value
                            best_cost=score_of_assignment
                            best_size = size
                            best_node_count = node_count
                            best_wmc = wmc
                            best_mc = mc
                    else:
                        if score_of_assignment <= best_cost:
                            best_variable = v
                            best_value = value
                            best_cost = score_of_assignment
                            best_size = size
                            best_node_count = node_count
                        #print("best: ", v, value)
    if obj_type == "hybrid_wmc" and not backbone_assigned:
        print("E:"+ str([best_variable, best_value, best_cost])+"\n")
        logger.progress_log.write("E:"+ str([best_variable, best_value, best_cost])+"\n")
        logger.progress_log.flush()
        actual_wmc_queue = _queue.PriorityQueue()
        nb_considered = 0
        while not assign_queue.empty():
            top_tuple = assign_queue.get()
            # print(top_tuple)
            score = abs(top_tuple[0])
            if score/abs(best_cost) >= 0.9:
                nb_considered +=1
                var = top_tuple[2]
                val = top_tuple[3]
                nb_nodes, nb_edges,  wmc, comp_time = csp.check_wmc_of(var,val, compile=False)
                weight = csp.literal_weights[val][var - 1]
                actual_wmc_queue.put(tuple([-1 * wmc, -1 * weight, var, val]))
                print(nb_considered, var, val, wmc)
                # logger.progress_log.write(str(top_tuple)+"\n")
                # logger.progress_log.flush()
            else:
                break
        best = actual_wmc_queue.queue[0]
        best_variable = best[2]
        best_value =  best[3]
        best_cost =  best[0]
        best_wmc = best_cost
        best_size = -1
        best_node_count = -1
        # logger.progress_log.write("order after actual wmc calculation: \n")
        # logger.progress_log.flush()
        print("A:"+ str([best_variable, best_value, best_cost, nb_considered]) + "\n")
        logger.progress_log.write("A:"+ str([best_variable, best_value, best_cost, nb_considered]) + "\n")
        logger.progress_log.write("----- \n")
        logger.progress_log.flush()
        # while not actual_wmc_queue.empty():
        #     top_tuple = actual_wmc_queue.get()
        #     logger.progress_log.write(str(top_tuple)+"\n")
        # logger.progress_log.flush()
    elif backbone_assigned:
        logger.progress_log.write("backbone" + "\n")
        logger.progress_log.write(str([best_variable, best_value, best_cost]) + "\n")
        logger.progress_log.write("----- \n")

    if not NO_COMPILE: #if compilation needed but heur does not calculate it
        if (obj_type == "count" or "score" in obj_type) or backbone_assigned or obj_type == "hybrid_wmc":
            nb_nodes, nb_edges,  best_wmc, comp_time = csp.check_wmc_of(best_variable, best_value)
            best_size = nb_edges
            best_node_count = nb_nodes
            _, _, best_mc, _ = csp.check_mc_of(best_variable, best_value)
        if obj_type == "WMC":
            _, _, best_mc, _ = csp.check_mc_of(best_variable, best_value)

    return best_variable,best_value, best_cost, best_size, best_node_count, best_mc, best_wmc


def dynamic_greedy_pWSB(csp, max_p, obj_type,logger, NO_COMPILE=False, sample_size=-1) :
    #obj_type = MC or WMC
    print("DYNAMIC")
    pa = csp.partial_assignment
    p = len(pa.assigned)
    print(p, max_p)
    extra_iterations = 0
    if sample_size != -1:
        sample_index = [ int((i*max_p/sample_size))+1 for i in range(sample_size)]
    iteration_index = 0
    while p < max_p:
        #select the assignment that maximizes the score
        p += 1
        if sample_size != -1:
            index_count = sample_index.count(p)
            if index_count == 0: #no need to compile this run
                NO_COMPILE = True
                extra_iterations = 0
            else:
                NO_COMPILE = False #need to compile
                extra_iterations = index_count

        else:
            iteration_index = p

        print("---------------no compile",  NO_COMPILE , p, extra_iterations)
        best_variable, best_value, best_cost, best_size, best_node_count, mc, wmc = get_best_assignment(csp,obj_type, NO_COMPILE,logger)
        print("assign ",iteration_index , best_variable, best_value, best_cost, wmc, mc)

        elapsed = logger.get_time_elapsed()

        if wmc == 0.0:
            log_line = [iteration_index, best_variable, best_value, csp.n, len(csp.cls), mc, best_size, best_node_count, elapsed, 0, 0, best_cost]
            logger.log(log_line)
            return
        else:
            if wmc == -1:
                logWMC = -1
            else:
                logWMC = math.log10(wmc)
            log_line = [iteration_index, best_variable, best_value, csp.n, len(csp.cls), mc, best_size, best_node_count, elapsed, wmc, logWMC, best_cost]

        while extra_iterations > 0 :
            iteration_index += 1
            log_line[0] = iteration_index
            logger.log(log_line)
            extra_iterations -= 1

        csp.extend_assignment( best_variable,best_value, abs(best_cost), propagate=True )
        # print( p, best_variable, best_value, best_cost)

    # print("p=", p)
    # print("variable,value,score")
    # for i in result:
    #     print(','.join(map(str, i)))


def dynamic_greedy_pWSB_at_variable_percent(csp, max_p, obj_type,logger, NO_COMPILE=False, sample_size=-1, var_percentage=-1) :
    #obj_type = MC or WMC
    pa = csp.partial_assignment
    p = len(pa.assigned)
    print(p, max_p)
    variable_assignment = round((var_percentage * max_p) / 100)
    print("DYNAMIC ---- var ", variable_assignment, "out of ", max_p)

    while p < variable_assignment:
        #select the assignment that maximizes the score
        p += 1

        best_variable, best_value, best_cost, best_size, best_node_count, mc, wmc = get_best_assignment(csp,obj_type, NO_COMPILE,logger)
        print("assign ",p , best_variable, best_value, best_cost, wmc, mc)
        elapsed = logger.get_time_elapsed()
        log_line = [p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size, best_node_count, elapsed, wmc, -1, best_cost]
        logger.log(log_line)
        if wmc == 0  or wmc == -1:
            print("expr failed")
            exit(22)

        csp.extend_assignment( best_variable,best_value, abs(best_cost), propagate=True )

    best_variable, best_value, best_cost, best_size, best_node_count, mc, wmc = get_best_assignment(csp,obj_type, NO_COMPILE,logger)
    print("assign ",p , best_variable, best_value, best_cost, wmc, mc)

    cnf_file_name = csp.instance_name.replace(".cnf", "_temp" + csp.obj_type + csp.heur_type + "_22percent_medium3.cnf")
    csp.print_clauses(cnf_file_name, csp.cls, csp.n)

    # csp.extend_assignment(best_variable, best_value, abs(best_cost), propagate=True)



    elapsed = logger.get_time_elapsed()

    if wmc == 0.0:
        log_line = [p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size, best_node_count, elapsed, 0, 0, best_cost]
        logger.log(log_line)
        return
    else:
        if wmc == -1:
            logWMC = -1
        else:
            logWMC = math.log10(wmc)
        log_line = [p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size, best_node_count, elapsed, wmc, logWMC, best_cost]
    logger.log(log_line)


    # print( p, best_variable, best_value, best_cost)

    # print("p=", p)
    # print("variable,value,score")
    # for i in result:
    #     print(','.join(map(str, i)))
def dynamic_random(csp, max_p, obj_type, logger, NO_COMPILE=False):
    print("DYNAMIC RANDOM")
    pa = csp.partial_assignment
    p = len(pa.assigned)
    print(p, max_p)
    best_mc = -1
    best_wmc = -1
    best_size =-1
    best_node_count =-1
    while p < max_p:
        best_weight = 0
        not_yet_assigned = []
        backbones = []
        # select the assignment that maximizes the score
        p += 1
        #check if there's a backbone
        if len(csp.trivial_backbone) > 0:
            for tb in csp.trivial_backbone:
                if abs(tb) in csp.partial_assignment.assigned:
                    print("Trivial backbone already assigned")
                    exit(6)
                if tb < 0:
                    value = 0
                else:
                    value = 1
                variable = abs(tb)
                weight = csp.literal_weights[value][variable - 1]
                if weight > best_weight:
                    best_weight = weight
                    best_variable = variable
                    best_value = value
                    best_cost = weight * 100000
                    best_tb = tb

            csp.trivial_backbone.remove(best_tb)
            print("backbone assigned")
        else:
            print("search assignment...")
            for variable in csp.variables.keys():
                if variable not in csp.partial_assignment.assigned:
                    for value in csp.variables[variable]:
                        lit = variable
                        if value == 0:
                            lit = -variable
                        if [lit] in csp.cls:
                            print("should not be backbone")
                            exit(6)
                        else:
                            not_yet_assigned.append(lit)
            best_lit = random.choice(not_yet_assigned)
            if best_lit < 0:
                best_value = 0
            else:
                best_value = 1
            best_variable = abs(best_lit)
            best_cost = 0
        if not NO_COMPILE:
            nb_nodes, nb_edges, best_wmc, comp_time = csp.check_wmc_of(best_variable, best_value)
            _, _, best_mc, _ = csp.check_mc_of(best_variable, best_value)
            best_size = nb_edges
            best_node_count = nb_nodes

        elapsed = logger.get_time_elapsed()

        if best_wmc == 0.0:
            logger.log(
                [p, best_variable, best_value, csp.n, len(csp.cls), best_mc, best_size, best_node_count, elapsed, 0, 0,
                 best_cost])
            return
        else:
            if best_wmc == -1:
                logWMC = -1
            else:
                logWMC = math.log10(best_wmc)
            logger.log([p, best_variable, best_value, csp.n, len(csp.cls), best_mc, best_size, best_node_count, elapsed, best_wmc, logWMC, best_cost])

        csp.extend_assignment(best_variable, best_value, abs(best_cost), propagate=True)


def order_var_assignments(csp, obj_type):
    """
    This is used for the static and random ordering
    """
    assign_queue = _queue.PriorityQueue()
    wmc = -1
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                # print("var , val ", v, value)
                if obj_type == "WMC" or obj_type == "MC" or obj_type == "SUB":
                    if obj_type == "MC":
                        node_count, nb_edges, mc, comp_time = csp.check_mc_of(v, value)
                    else:
                        node_count, nb_edges, wmc, comp_time = csp.check_wmc_of(v, value)
                    score_of_assignment = wmc
                    size = nb_edges
                    if obj_type == "MC":
                        score_of_assignment = mc
                    elif obj_type == "SUB":
                        score_of_assignment = wmc - mc
                    # print("score_of_assignment, size, node_count, temp_root ", score_of_assignment, size, node_count )
                elif obj_type =="count":
                    opp_score = csp.opposite_occurance(v, value)
                    score_of_assignment = -1* opp_score
                    size = -1
                    node_count = -1
                elif "score" in obj_type:
                    if obj_type == "wscore_half":
                        score_of_assignment = csp.calculate_score(v, value, "half", weighted=True)
                    elif obj_type == "wscore_occratio":
                        score_of_assignment = csp.calculate_score(v, value, "occratio", weighted=True)
                    elif   obj_type == "wscore_adjoccratio":
                        score_of_assignment = csp.calculate_score(v, value, "adjoccratio", weighted=True)
                    elif obj_type == "wscore_estimate":
                        score_of_assignment = csp.calculate_score(v, value, "estimate", weighted=True)
                    elif obj_type == "wscore_otherwg":
                        score_of_assignment = csp.calculate_score(v, value, "otherwg", weighted=True)
                    elif obj_type == "score_estimate":
                        score_of_assignment = csp.calculate_score(v, value, "estimate", weighted=False)

                    size = -1
                    node_count = -1
                elif obj_type == "g2":
                    score_of_assignment, size, node_count, temp_root  = csp.calculate_g2(v, value)
                elif "ratio" in obj_type:
                    score_of_assignment, size, node_count, temp_root = csp.check_wmc_ratio_of(v, value)

                else:
                    print("Something went wrong")
                    exit(6)

                #best_variable, best_value, best_cost, best_bdd, stats
                #minimize
                # temp_root.ref()
                # csp.sdd_manager.minimize()
                # size = temp_root.size()
                # node_count = temp_root.count()
                if obj_type == "WMC":
                    weight = csp.literal_weights[value][v-1]
                else:
                    weight = 1
                assign_queue.put(tuple([-1*score_of_assignment,-1*weight, v, value, size, node_count]))


    return assign_queue

def random_var_assignments(csp,seed): #TODO
    """
    this contains both x=0 and x=1 so at selection the paris have to be removed - random assignments are chosen for variables.
    :param csp:
    :return:
    """
    assign_queue = []
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                assign_queue.append(tuple([v, value]))
    random.Random(seed).shuffle(assign_queue)
    return assign_queue

def random_pWSB(csp, seed, logger):#TODO
    """
    Here both variable and value are selcted randomly
    :param csp:
    :param seed:
    :return:
    """
    assign_queue = random_var_assignments(csp,seed)
    p = 0
    seen_vars = set()
    for item in assign_queue:
        variable = item[0]
        value = item[1]
        if variable not in seen_vars:

            p += 1
            seen_vars.add(variable)
            node_count, nb_edges, wmc, mc, comp_time = csp.check_wmc_of(variable, value)
            elapsed = logger.get_time_elapsed()
            if wmc == 0.0:
                logger.log([p, variable, value, csp.n, len(csp.cls), mc, nb_edges, node_count, elapsed, 0, 0])
                return
            else:
                logWMC = math.log10(wmc)
                logger.log([p, variable, value, csp.n, len(csp.cls), mc, nb_edges, node_count, elapsed, wmc, logWMC])

            csp.extend_assignment(variable, value, wmc, propagate=False)

def random_selection_pWSB(csp, seed,logger, obj_type):#TODO
    # TODO obj mc/ratio is not taken into account
    """
    Select variables randomly and assign value that maximizes model count
    :param csp:
    :param seed:
    :return:
    """
    p = 0
    assign_queue = list(csp.variables.keys())
    random.Random(seed).shuffle(assign_queue)

    for variable in assign_queue:

            # need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])

        value=0
        #use wmc as obj
        if obj_type == "WMC":
            node_count0, nb_edges0, wmc0, mc0, comp_time0 = csp.check_wmc_of(variable, 0)
            node_count1, nb_edges1, wmc1, mc1, comp_time1 = csp.check_wmc_of(variable, 1)
            score0 = wmc0
            score1 = wmc1
        elif obj_type == "MC":
            node_count0, nb_edges0, mc0, comp_time0 = csp.check_mc_of(variable, 0)
            node_count1, nb_edges1, mc1, comp_time1 = csp.check_mc_of(variable, 1)
            score0 = mc0
            score1 = mc1
        # elif obj_type == "SUB":
        #     score0 = wmc0 - mc0
        #     score1 = wmc1 - mc1
        elif obj_type == "count":
            score0 = csp.opposite_occurance(variable, 0)
            score1 = csp.opposite_occurance(variable, 1)
        elif "score" in obj_type:
            if obj_type == "wscore_half":
                score0 = csp.calculate_score(variable, 0, "half", weighted=True)
                score1 = csp.calculate_score(variable, 1, "half", weighted=True)
            elif obj_type == "wscore_occratio":
                score0 = csp.calculate_score(variable, 0, "occratio", weighted=True)
                score1 = csp.calculate_score(variable, 1, "occratio", weighted=True)
            elif obj_type == "wscore_adjoccratio":
                score0 = csp.calculate_score(variable, 0, "adjoccratio", weighted=True)
                score1 = csp.calculate_score(variable, 1, "adjoccratio", weighted=True)
            elif obj_type == "wscore_estimate":
                score0 = csp.calculate_score(variable, 0, "estimate", weighted=True)
                score1 = csp.calculate_score(variable, 1, "estimate", weighted=True)
            elif obj_type == "wscore_otherwg":
                score0 = csp.calculate_score(variable, 0, "otherwg", weighted=True)
                score1 = csp.calculate_score(variable, 1, "otherwg", weighted=True)
            elif obj_type == "score_estimate":
                score0 = csp.calculate_score(variable, 0, "estimate", weighted=False)
                score1 = csp.calculate_score(variable, 1, "estimate", weighted=False)

        else:
            print("ERROR in obj")
            exit(9)
        if score0 > score1:
            best_cost = score0
            value = 0
            node_count, size, wmc, mc, comp_time = csp.check_wmc_of(variable, 0)
        else:
            value=1
            best_cost = score1
            node_count, size, wmc, mc, comp_time = csp.check_wmc_of(variable, 1)
        p += 1
        elapsed = logger.get_time_elapsed()
        if wmc == 0.0:
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, 0, 0, best_cost])
            return
        else:
            logWMC = math.log10(wmc)
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, logWMC, best_cost])

        print("extend", variable, value)
        csp.extend_assignment(variable, value, wmc, propagate=False)

def static_greedy_pWSB(csp, obj_type,logger,  NO_COMPILE=False):
    print("STATIC")
    assign_queue = order_var_assignments(csp, obj_type)
    p = 0
    seen_vars = set()
    while not assign_queue.empty():
        mc = -1
        size=-1
        node_count=-1
        wmc = -1
        item = assign_queue.get() # tuple of ( score_of_assignment,var, value, size, node_count )
        score_of_assignment = abs(item[0])
        literal_weight = abs(item[1])
        variable = item[2]
        value = item[3]
        print("assign", p, variable, value, len(seen_vars), assign_queue.qsize())
        if variable not in seen_vars:
            p += 1
            seen_vars.add(variable)
            #need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])

            if not NO_COMPILE:
                if obj_type == "WMC" or obj_type == "MC" or obj_type == "SUB":
                    if obj_type == "MC":
                        nb_nodes, nb_edges, mc, comp_time = csp.check_mc_of(variable, value)
                    else:
                        node_count, nb_edges,wmc, comp_time = csp.check_wmc_of(variable, value)
                        if not NO_COMPILE:
                            _, _, mc, _ = csp.check_mc_of(variable, value)
                    size = nb_edges
                    score_of_assignment = wmc
                    if obj_type == "MC":
                        score_of_assignment = mc
                    elif obj_type  ==  "SUB":
                        score_of_assignment = wmc -mc
                elif obj_type == "count":
                    opp_count = csp.opposite_occurance(variable, value)
                    score_of_assignment = opp_count
                    node_count, nb_edges, wmc, comp_time = csp.check_wmc_of(variable, value)
                    size = nb_edges
                elif "score" in obj_type:
                    if obj_type == "wscore_half":
                        score_of_assignment = csp.calculate_score(variable, value, "half", weighted=True)
                    elif obj_type == "wscore_occratio":
                        score_of_assignment = csp.calculate_score(variable, value, "occratio", weighted=True)
                    elif obj_type == "wscore_adjoccratio":
                        score_of_assignment = csp.calculate_score(variable, value, "adjoccratio", weighted=True)
                    elif obj_type == "wscore_estimate":
                        score_of_assignment = csp.calculate_score(variable, value, "estimate", weighted=True)
                    elif obj_type == "wscore_otherwg":
                        score_of_assignment = csp.calculate_score(variable, value, "otherwg", weighted=True)
                    elif obj_type == "score_estimate":
                        score_of_assignment = csp.calculate_score(variable, value, "estimate", weighted=False)
                    if not NO_COMPILE:
                        node_count, size,wmc, comp_time = csp.check_wmc_of(variable, value)
                        _, _,mc, _ = csp.check_mc_of(variable, value)

                elif obj_type == "static_ratio":
                    score_of_assignment, size, node_count, temp_root = csp.check_wmc_ratio_of(variable, value)
                elif obj_type == "g2":
                    score_of_assignment, size, node_count, temp_root = csp.calculate_g2(variable, value)

                else:
                    print("ERROR")
                    exit(666)

            elapsed = logger.get_time_elapsed()

            ## calculate both wmc and g2
            # print([p,  variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc])
            if wmc ==  0.0:
                logger.log([p,  variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, 0, 0, score_of_assignment])
                return
            else:
                if wmc == -1:
                    logWMC = -1
                else:
                    logWMC = math.log10(wmc)
                logger.log([p,  variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, logWMC, score_of_assignment])

            csp.extend_assignment(variable, value, score_of_assignment, propagate=False)

def print_mc_per_vars(csp, logger):#TODO
    assign_queue = order_var_assignments(csp, "mc")
    p= 0
    while not assign_queue.empty():
        print("=======================", p)
        item = assign_queue.get()  # tuple of ( score_of_assignment,var, value, size, node_count )
        score_of_assignment = abs(item[0])
        variable = item[1]
        value = item[2]
        sdd_size = item[3]
        node_count = item[4]
        p += 1

        elapsed = logger.get_time_elapsed()
        logger.log([p, variable, value, csp.n, len(csp.cls), score_of_assignment, sdd_size, node_count, elapsed])
def inti_compilation(alg_type, d, filename, out_folder, obj_type):
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]
    expr_data = evaluate.ExprData(columns)
    if "random" in alg_type or "ls" in alg_type:
        # stats_file = d + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
    else:
        # stats_file = d + "dataset_stats_" + alg_type + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + ".csv"
    logger = evaluate.Logger(stats_file, columns, expr_data, out_folder, compile=True)
    print(filename)
    all_start = time.perf_counter()
    logger.log_expr(filename)
    start = time.perf_counter()
    logger.set_start_time(start)
    cnf = _wcnfd4.WCNF(logger, scalar=0)
    b = cnf.load_file(filename, obj_type, alg_type)
    print(logger.get_time_elapsed())


def run_sdd(alg_type, filename, seed, out_folder, obj_type, scalar=3, NO_COMPILE=False, part="", sample_size=-1):
    # obj_type: mc or g2
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC", "obj"]
    if "random" in alg_type or "ls" in alg_type:
        # stats_file = d + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
        stats_file = out_folder + "dataset_stats_init" + alg_type + "_" + str(seed) + ".csv"
    else:
        # stats_file = d + "dataset_stats_" + alg_type + ".csv"
        stats_file = out_folder + "dataset_stats_init" + alg_type + ".csv"
    if part != "":
        stats_file = stats_file.replace(".csv", "_part"+str(part)+".csv")
    print("stats file: --------", stats_file)
    expr_data = evaluate.ExprData(columns)
    logger = evaluate.Logger(stats_file, columns, expr_data, out_folder, compile=True)
    print(filename)
    all_start = time.perf_counter()
    logger.log_expr(filename)
    start = time.perf_counter()
    logger.set_start_time(start)
    cnf = _wcnfd4.WCNF(logger, scalar=scalar,NO_COMPILE=NO_COMPILE)
    b = cnf.load_file(filename, obj_type, alg_type)
    print(logger.get_time_elapsed())
    # if not b:
    #     return

    #TODO: iterate here between the alf types to avoid spending time for loding initial compilation -
    # needt to save first line in log to add it to all consequtive stat files
    maxp = len(cnf.literals)
    if alg_type == "dynamic":
        logger.progress_log.write(filename + "\n")
        logger.progress_log.flush()
        if "init" in stats_file:
            print("ONLY INIT !!!!!!!!!!!!!!!")
            maxp = 1
        dynamic_greedy_pWSB(cnf, maxp, obj_type, logger,NO_COMPILE, sample_size)
    elif alg_type == "dynamic_ratio":
        dynamic_greedy_pWSB(cnf, maxp, "dynamic_ratio",logger,NO_COMPILE, sample_size)
    elif alg_type == "rand_dynamic":
        random.seed(seed)
        dynamic_random(cnf, maxp, obj_type, logger, NO_COMPILE)
    elif alg_type == "static":
        static_greedy_pWSB(cnf, obj_type, logger, NO_COMPILE, sample_size)
    elif alg_type == "static_ratio":
        static_greedy_pWSB(cnf, "static_ratio",logger, NO_COMPILE, sample_size)
    elif "random" == alg_type:
        random_pWSB(cnf, seed,logger)
    elif "random_selection" in alg_type:
        random_selection_pWSB(cnf, seed,logger, obj_type)
    # elif alg_type == "ls":
    #     local_search_pWSB(cnf, maxp, seed + f_count,logger)
    elif alg_type == "init":
        print_mc_per_vars(cnf,logger)
    else:
        print("Something wrong")

    all_end = time.perf_counter()
    logger.close()
    print("ELAPSED TIME: ", all_end - all_start)

def run_at_p_percent_variable(alg_type, filename, seed, out_folder, obj_type, scalar=3, NO_COMPILE=True,part="",  var_percentage=0):
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "edge_count", 'node_count', 'time', 'WMC', "logWMC",
               "obj"]
    if "random" in alg_type or "ls" in alg_type:
        # stats_file = d + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
        stats_file = out_folder + "dataset_stats_p_" + alg_type + "_" + str(seed) + ".csv"
    else:
        # stats_file = d + "dataset_stats_" + alg_type + ".csv"
        stats_file = out_folder + "dataset_stats_medium4_p_" + alg_type + ".csv"
    if part != "":
        stats_file = stats_file.replace(".csv", "_part"+str(part)+".csv")
    if var_percentage != 0:
        stats_file = stats_file.replace("_part", "_p" + str(var_percentage)+ "_part")
    print("stats file: --------", stats_file)
    expr_data = evaluate.ExprData(columns)
    logger = evaluate.Logger(stats_file, columns, expr_data, out_folder, compile=True)
    print(filename)
    all_start = time.perf_counter()
    logger.log_expr(filename)
    start = time.perf_counter()
    logger.set_start_time(start)
    cnf = _wcnfd4.WCNF(logger, scalar=scalar, NO_COMPILE=NO_COMPILE)
    b = cnf.load_file(filename, obj_type, alg_type)
    print(logger.get_time_elapsed())
    # if not b:
    #     return

    # TODO: iterate here between the alf types to avoid spending time for loding initial compilation -
    # needt to save first line in log to add it to all consequtive stat files
    maxp = len(cnf.literals)
    if alg_type == "dynamic":
        logger.progress_log.write(filename + "\n")
        logger.progress_log.flush()
        dynamic_greedy_pWSB_at_variable_percent(cnf, maxp, obj_type, logger, NO_COMPILE=True, sample_size=0, var_percentage=var_percentage)
    all_end = time.perf_counter()
    logger.close()
    print("ELAPSED TIME: ", all_end - all_start)

if __name__ == "__main__":
    # type = "dynamic"
    # type = "dynamic_ratio"
    type = "static"
    # type = "static_ratio"
    # type = "random"
    # type = "random_selection"
    # type = "init"
    # type = "ls"
    # d = "./iscas/iscas99/"
    # d = "./DatasetA/"
    # d = "./DatasetB/"
    seed = 1234

    # # --------------------------------------- run.sh

    # inobj = sys.argv[3]
    # d = sys.argv[1] #"./input/wmc2022_track2_private/"
    # folder = d.split("/")[-2]
    # filename = sys.argv[2]
    # alg_type = sys.argv[3]
    # filename = sys.argv[1] #'./input/Benchmark_preproc2/C169_FV.cnf'
    # alg_type = sys.argv[2]


    d = sys.argv[1] #"./input/wmc2022_track2_private/"
    folder = d.split("/")[-2]
    filename = sys.argv[2]
    inobj = sys.argv[3] #hybrid_wmc
    alg_type = sys.argv[4]
    part = str(sys.argv[5])
    # NO_COMPILE = False


    # d = "./input/Dataset_preproc/"
    # folder = d.split("/")[-2]
    # filename = d+"01_istance_K3_N15_M45_01.cnf"
    # inobj = "hybrid_wmc"
    # alg_type = "dynamic"
    # NO_COMPILE = False
    # part=1
    sample_size = -1


    no_init_compilation = ['16_uts_k3_p_t3.cnf', '16_uts_k4_p_t2.cnf', '15_sort_num_s_5_p_t3.cnf', '16_uts_k3_p_t4.cnf', '16_uts_k2_p_t9.cnf',
                           '15_sort_num_s_6_p_t2.cnf', '15_sort_num_s_5_p_t4.cnf', '16_uts_k3_p_t5.cnf', '16_uts_k4_p_t3.cnf', '16_uts_k5_p_t2.cnf',
                           '15_sort_num_s_5_p_t5.cnf', '16_uts_k3_p_t6.cnf', '03_iscas85_c2670_isc.cnf', '16_uts_k4_p_t4.cnf', '15_sort_num_s_6_p_t3.cnf',
                           '16_uts_k3_p_t7.cnf', '15_sort_num_s_5_p_t6.cnf', '16_uts_k5_p_t3.cnf', '15_sort_num_s_7_p_t2.cnf', '16_uts_k3_p_t8.cnf',
                           '09_coins_p01_p_t10.cnf', '09_coins_p02_p_t10.cnf', '09_coins_p03_p_t10.cnf', '09_coins_p04_p_t10.cnf', '09_coins_p05_p_t10.cnf',
                           '15_sort_num_s_5_p_t7.cnf', '16_uts_k4_p_t5.cnf', '07_blocks_right_4_p_t6.cnf', '07_blocks_right_5_p_t4.cnf', '09_coins_p10_p_t5.cnf',
                           '16_uts_k3_p_t9.cnf', '15_sort_num_s_6_p_t4.cnf', '07_blocks_right_6_p_t3.cnf', '15_sort_num_s_5_p_t8.cnf', '16_uts_k10_p_t1.cnf',
                           '16_uts_k3_p_t10.cnf', '16_uts_k4_p_t6.cnf', '07_blocks_right_4_p_t7.cnf', '16_uts_k5_p_t4.cnf', '09_coins_p10_p_t6.cnf',
                           '15_sort_num_s_5_p_t9.cnf', '07_blocks_right_5_p_t5.cnf', '11_emptyroom_d20_g10_corners_p_t10.cnf', '15_sort_num_s_6_p_t5.cnf',
                           '07_blocks_right_4_p_t8.cnf', '15_sort_num_s_7_p_t3.cnf', '16_uts_k4_p_t7.cnf', '15_sort_num_s_5_p_t10.cnf', '07_blocks_right_6_p_t4.cnf',
                           '09_coins_p10_p_t7.cnf', '16_uts_k5_p_t5.cnf', '07_blocks_right_4_p_t9.cnf', '07_blocks_right_5_p_t6.cnf', '16_uts_k4_p_t8.cnf',
                           '15_sort_num_s_6_p_t6.cnf', '09_coins_p10_p_t8.cnf', '07_blocks_right_4_p_t10.cnf', '16_uts_k4_p_t9.cnf', '07_blocks_right_5_p_t7.cnf',
                           '07_blocks_right_6_p_t5.cnf', '16_uts_k5_p_t6.cnf', '11_emptyroom_d28_g14_corners_p_t10.cnf', '15_sort_num_s_7_p_t4.cnf', '09_coins_p10_p_t9.cnf', '15_sort_num_s_6_p_t7.cnf', '16_uts_k4_p_t10.cnf', '07_blocks_right_5_p_t8.cnf', '03_iscas85_c7552_isc.cnf', '09_coins_p10_p_t10.cnf', '16_uts_k5_p_t7.cnf', '15_sort_num_s_6_p_t8.cnf', '07_blocks_right_6_p_t6.cnf', '05_iscas93_s6669_bench.cnf', '16_uts_k10_p_t2.cnf', '15_sort_num_s_7_p_t5.cnf', '07_blocks_right_5_p_t9.cnf', '16_uts_k5_p_t8.cnf', '15_sort_num_s_6_p_t9.cnf', '07_blocks_right_6_p_t7.cnf', '07_blocks_right_5_p_t10.cnf', '15_sort_num_s_6_p_t10.cnf', '16_uts_k5_p_t9.cnf', '15_sort_num_s_7_p_t6.cnf', '07_blocks_right_6_p_t8.cnf', '16_uts_k5_p_t10.cnf', '15_sort_num_s_7_p_t7.cnf', '07_blocks_right_6_p_t9.cnf', '16_uts_k10_p_t3.cnf', '10_comm_p10_p_t3.cnf', '07_blocks_right_6_p_t10.cnf', '15_sort_num_s_7_p_t8.cnf', '10_comm_p05_p_t10.cnf', '15_sort_num_s_7_p_t9.cnf', '16_uts_k10_p_t4.cnf', '10_comm_p10_p_t4.cnf', '15_sort_num_s_7_p_t10.cnf', '16_uts_k10_p_t5.cnf', '10_comm_p10_p_t5.cnf', '10_comm_p10_p_t6.cnf', '16_uts_k10_p_t6.cnf', '10_comm_p10_p_t7.cnf', '16_uts_k10_p_t7.cnf', '10_comm_p10_p_t8.cnf', '16_uts_k10_p_t8.cnf', '10_comm_p10_p_t9.cnf', '16_uts_k10_p_t9.cnf', '10_comm_p10_p_t10.cnf', '08_bomb_b10_t10_p_t19.cnf', '16_uts_k10_p_t10.cnf', '08_bomb_b10_t10_p_t20.cnf']

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

    medium_instances = ['03_iscas85_c1355.isc.cnf', '03_iscas85_c1908.isc.cnf', '03_iscas85_c880.isc.cnf',
                        '04_iscas89_s1196.bench.cnf', '04_iscas89_s1238.bench.cnf',
                        '04_iscas89_s1423.bench.cnf', '04_iscas89_s1488.bench.cnf', '04_iscas89_s1494.bench.cnf',
                        '04_iscas89_s641.bench.cnf', '04_iscas89_s713.bench.cnf',
                        '04_iscas89_s820.bench.cnf', '04_iscas89_s832.bench.cnf', '04_iscas89_s838.1.bench.cnf',
                        '04_iscas89_s953.bench.cnf', '05_iscas93_s1196.bench.cnf',
                        '05_iscas93_s1269.bench.cnf', '05_iscas93_s1512.bench.cnf', '05_iscas93_s635.bench.cnf',
                        '05_iscas93_s938.bench.cnf', '05_iscas93_s967.bench.cnf',
                        '05_iscas93_s991.bench.cnf', '06_iscas99_b04.cnf', '06_iscas99_b07.cnf', '06_iscas99_b11.cnf',
                        '06_iscas99_b13.cnf', '07_blocks_right_2_p_t10.cnf',
                        '07_blocks_right_2_p_t4.cnf', '07_blocks_right_2_p_t5.cnf', '07_blocks_right_2_p_t6.cnf',
                        '07_blocks_right_2_p_t7.cnf', '07_blocks_right_2_p_t8.cnf',
                        '07_blocks_right_2_p_t9.cnf', '07_blocks_right_3_p_t2.cnf', '07_blocks_right_3_p_t3.cnf',
                        '07_blocks_right_3_p_t4.cnf', '07_blocks_right_3_p_t5.cnf',
                        '07_blocks_right_4_p_t2.cnf', '07_blocks_right_4_p_t3.cnf', '07_blocks_right_5_p_t1.cnf',
                        '07_blocks_right_5_p_t2.cnf', '07_blocks_right_6_p_t1.cnf',
                        '08_bomb_b10_t5_p_t1.cnf', '08_bomb_b5_t1_p_t3.cnf', '08_bomb_b5_t1_p_t4.cnf',
                        '08_bomb_b5_t1_p_t5.cnf', '08_bomb_b5_t1_p_t6.cnf', '08_bomb_b5_t1_p_t7.cnf',
                        '08_bomb_b5_t1_p_t8.cnf', '08_bomb_b5_t5_p_t2.cnf', '08_bomb_b5_t5_p_t3.cnf',
                        '09_coins_p01_p_t2.cnf', '09_coins_p01_p_t3.cnf', '09_coins_p01_p_t4.cnf',
                        '09_coins_p01_p_t5.cnf', '09_coins_p02_p_t2.cnf', '09_coins_p02_p_t3.cnf',
                        '09_coins_p02_p_t4.cnf', '09_coins_p02_p_t5.cnf', '09_coins_p03_p_t2.cnf',
                        '09_coins_p03_p_t3.cnf', '09_coins_p03_p_t4.cnf', '09_coins_p03_p_t5.cnf',
                        '09_coins_p04_p_t2.cnf', '09_coins_p04_p_t3.cnf', '09_coins_p04_p_t4.cnf',
                        '09_coins_p04_p_t5.cnf', '09_coins_p05_p_t2.cnf', '09_coins_p05_p_t3.cnf',
                        '09_coins_p05_p_t4.cnf', '09_coins_p05_p_t5.cnf', '09_coins_p10_p_t1.cnf',
                        '09_coins_p10_p_t2.cnf', '10_comm_p01_p_t3.cnf', '10_comm_p01_p_t4.cnf', '10_comm_p01_p_t5.cnf',
                        '10_comm_p01_p_t6.cnf', '10_comm_p02_p_t2.cnf',
                        '10_comm_p02_p_t3.cnf', '10_comm_p03_p_t1.cnf', '10_comm_p03_p_t2.cnf', '10_comm_p04_p_t1.cnf',
                        '10_comm_p05_p_t1.cnf', '11_emptyroom_d12_g6_p_t3.cnf',
                        '11_emptyroom_d12_g6_p_t4.cnf', '11_emptyroom_d12_g6_p_t5.cnf', '11_emptyroom_d12_g6_p_t6.cnf',
                        '11_emptyroom_d12_g6_p_t7.cnf', '11_emptyroom_d16_g8_p_t2.cnf',
                        '11_emptyroom_d16_g8_p_t3.cnf', '11_emptyroom_d16_g8_p_t4.cnf', '11_emptyroom_d16_g8_p_t5.cnf',
                        '11_emptyroom_d20_g10_corners_p_t2.cnf', '11_emptyroom_d20_g10_corners_p_t3.cnf',
                        '11_emptyroom_d20_g10_corners_p_t4.cnf', '11_emptyroom_d24_g12_p_t2.cnf',
                        '11_emptyroom_d24_g12_p_t3.cnf', '11_emptyroom_d28_g14_corners_p_t1.cnf',
                        '11_emptyroom_d28_g14_corners_p_t2.cnf', '11_emptyroom_d28_g14_corners_p_t3.cnf',
                        '11_emptyroom_d4_g2_p_t10.cnf', '11_emptyroom_d4_g2_p_t9.cnf', '11_emptyroom_d8_g4_p_t10.cnf',
                        '11_emptyroom_d8_g4_p_t4.cnf', '11_emptyroom_d8_g4_p_t5.cnf', '11_emptyroom_d8_g4_p_t6.cnf',
                        '11_emptyroom_d8_g4_p_t7.cnf', '11_emptyroom_d8_g4_p_t8.cnf',
                        '11_emptyroom_d8_g4_p_t9.cnf', '13_ring2_r6_p_t10.cnf', '13_ring2_r6_p_t5.cnf',
                        '13_ring2_r6_p_t6.cnf', '13_ring2_r6_p_t7.cnf', '13_ring2_r6_p_t8.cnf', '13_ring2_r6_p_t9.cnf',
                        '13_ring2_r8_p_t10.cnf', '13_ring2_r8_p_t4.cnf', '13_ring2_r8_p_t5.cnf', '13_ring2_r8_p_t6.cnf',
                        '13_ring2_r8_p_t7.cnf', '13_ring2_r8_p_t8.cnf', '13_ring2_r8_p_t9.cnf',
                        '13_ring_3_p_t10.cnf', '13_ring_3_p_t7.cnf', '13_ring_3_p_t8.cnf', '13_ring_3_p_t9.cnf',
                        '13_ring_4_p_t10.cnf', '13_ring_4_p_t5.cnf', '13_ring_4_p_t6.cnf', '13_ring_4_p_t7.cnf',
                        '13_ring_4_p_t8.cnf', '13_ring_4_p_t9.cnf', '13_ring_5_p_t10.cnf', '13_ring_5_p_t4.cnf',
                        '13_ring_5_p_t5.cnf', '13_ring_5_p_t6.cnf', '13_ring_5_p_t7.cnf', '13_ring_5_p_t8.cnf',
                        '13_ring_5_p_t9.cnf', '14_safe_safe_10_p_t10.cnf', '14_safe_safe_30_p_t3.cnf',
                        '14_safe_safe_30_p_t4.cnf', '14_safe_safe_30_p_t5.cnf', '14_safe_safe_30_p_t6.cnf',
                        '14_safe_safe_30_p_t7.cnf', '14_safe_safe_30_p_t8.cnf', '14_safe_safe_30_p_t9.cnf',
                        '15_sort_num_s_3_p_t10.cnf', '15_sort_num_s_4_p_t4.cnf', '15_sort_num_s_4_p_t5.cnf',
                        '15_sort_num_s_4_p_t6.cnf', '15_sort_num_s_4_p_t7.cnf', '15_sort_num_s_4_p_t8.cnf',
                        '15_sort_num_s_4_p_t9.cnf', '15_sort_num_s_5_p_t2.cnf', '15_sort_num_s_6_p_t1.cnf',
                        '15_sort_num_s_7_p_t1.cnf', '16_uts_k2_p_t4.cnf', '16_uts_k2_p_t5.cnf', '16_uts_k2_p_t6.cnf',
                        '16_uts_k2_p_t7.cnf', '16_uts_k2_p_t8.cnf', '16_uts_k3_p_t2.cnf',
                        '16_uts_k4_p_t1.cnf', '16_uts_k5_p_t1.cnf']

    medium3 = ['05_iscas93_s1269_bench.cnf','16_uts_k2_p_t8.cnf']
        # '03_iscas85_c1355_isc.cnf', '03_iscas85_c1908_isc.cnf', '05_iscas93_s1269.bench.cnf', '06_iscas99_b04.cnf', '16_uts_k2_p_t7.cnf', '16_uts_k2_p_t8.cnf'
                                                                                                                                                 
               # '07_blocks_right_2_p_t10.cnf', '07_blocks_right_2_p_t5.cnf',
               # '07_blocks_right_2_p_t8.cnf', '07_blocks_right_3_p_t5.cnf', '07_blocks_right_5_p_t2.cnf', '07_blocks_right_6_p_t1.cnf',
               #
               # '13_ring2_r6_p_t10.cnf', '13_ring2_r6_p_t9.cnf',
               # '13_ring2_r8_p_t10.cnf', '13_ring2_r8_p_t8.cnf', '13_ring2_r8_p_t9.cnf', '13_ring_5_p_t10.cnf', '13_ring_5_p_t6.cnf',
               #
               # '15_sort_num_s_4_p_t7.cnf', '15_sort_num_s_4_p_t8.cnf',
               # '15_sort_num_s_4_p_t9.cnf', '15_sort_num_s_5_p_t2.cnf', '15_sort_num_s_6_p_t1.cnf', '15_sort_num_s_7_p_t1.cnf', ]


    large_instances = ['05_iscas93_s3271_bench.cnf', '05_iscas93_s3330_bench.cnf', '05_iscas93_s3384_bench.cnf', '05_iscas93_s4863_bench.cnf', '06_iscas99_b05.cnf',
                       '06_iscas99_b12.cnf', '07_blocks_right_3_p_t10.cnf', '07_blocks_right_3_p_t6.cnf', '07_blocks_right_3_p_t7.cnf', '07_blocks_right_3_p_t8.cnf',
                       '07_blocks_right_3_p_t9.cnf', '07_blocks_right_4_p_t4.cnf', '07_blocks_right_4_p_t5.cnf', '07_blocks_right_5_p_t3.cnf', '07_blocks_right_6_p_t2.cnf',
                       '08_bomb_b10_t10_p_t10.cnf', '08_bomb_b10_t10_p_t11.cnf', '08_bomb_b10_t10_p_t12.cnf', '08_bomb_b10_t10_p_t13.cnf', '08_bomb_b10_t10_p_t14.cnf',
                       '08_bomb_b10_t10_p_t15.cnf', '08_bomb_b10_t10_p_t16.cnf', '08_bomb_b10_t10_p_t17.cnf', '08_bomb_b10_t10_p_t18.cnf', '08_bomb_b10_t10_p_t1.cnf',
                       '08_bomb_b10_t10_p_t2.cnf', '08_bomb_b10_t10_p_t3.cnf', '08_bomb_b10_t10_p_t4.cnf', '08_bomb_b10_t10_p_t5.cnf', '08_bomb_b10_t10_p_t6.cnf', '08_bomb_b10_t10_p_t7.cnf',
                       '08_bomb_b10_t10_p_t8.cnf', '08_bomb_b10_t10_p_t9.cnf', '08_bomb_b10_t5_p_t10.cnf', '08_bomb_b10_t5_p_t2.cnf', '08_bomb_b10_t5_p_t3.cnf', '08_bomb_b10_t5_p_t4.cnf',
                       '08_bomb_b10_t5_p_t5.cnf', '08_bomb_b10_t5_p_t6.cnf', '08_bomb_b10_t5_p_t7.cnf', '08_bomb_b10_t5_p_t8.cnf', '08_bomb_b10_t5_p_t9.cnf', '08_bomb_b20_t5_p_t10.cnf',
                       '08_bomb_b20_t5_p_t1.cnf', '08_bomb_b20_t5_p_t2.cnf', '08_bomb_b20_t5_p_t3.cnf', '08_bomb_b20_t5_p_t4.cnf', '08_bomb_b20_t5_p_t5.cnf', '08_bomb_b20_t5_p_t6.cnf',
                       '08_bomb_b20_t5_p_t7.cnf', '08_bomb_b20_t5_p_t8.cnf', '08_bomb_b20_t5_p_t9.cnf', '08_bomb_b5_t1_p_t10.cnf', '08_bomb_b5_t1_p_t9.cnf', '08_bomb_b5_t5_p_t10.cnf',
                       '08_bomb_b5_t5_p_t4.cnf', '08_bomb_b5_t5_p_t5.cnf', '08_bomb_b5_t5_p_t6.cnf', '08_bomb_b5_t5_p_t7.cnf', '08_bomb_b5_t5_p_t8.cnf', '08_bomb_b5_t5_p_t9.cnf',
                       '09_coins_p01_p_t6.cnf', '09_coins_p01_p_t7.cnf', '09_coins_p01_p_t8.cnf', '09_coins_p01_p_t9.cnf', '09_coins_p02_p_t6.cnf', '09_coins_p02_p_t7.cnf',
                       '09_coins_p02_p_t8.cnf', '09_coins_p02_p_t9.cnf', '09_coins_p03_p_t6.cnf', '09_coins_p03_p_t7.cnf', '09_coins_p03_p_t8.cnf', '09_coins_p03_p_t9.cnf',
                       '09_coins_p04_p_t6.cnf', '09_coins_p04_p_t7.cnf', '09_coins_p04_p_t8.cnf', '09_coins_p04_p_t9.cnf', '09_coins_p05_p_t6.cnf', '09_coins_p05_p_t7.cnf',
                       '09_coins_p05_p_t8.cnf', '09_coins_p05_p_t9.cnf', '09_coins_p10_p_t3.cnf', '09_coins_p10_p_t4.cnf', '10_comm_p01_p_t10.cnf', '10_comm_p01_p_t7.cnf',
                       '10_comm_p01_p_t8.cnf', '10_comm_p01_p_t9.cnf', '10_comm_p02_p_t10.cnf', '10_comm_p02_p_t4.cnf', '10_comm_p02_p_t5.cnf', '10_comm_p02_p_t6.cnf',
                       '10_comm_p02_p_t7.cnf', '10_comm_p02_p_t8.cnf', '10_comm_p02_p_t9.cnf', '10_comm_p03_p_t10.cnf', '10_comm_p03_p_t3.cnf', '10_comm_p03_p_t4.cnf',
                       '10_comm_p03_p_t5.cnf', '10_comm_p03_p_t6.cnf', '10_comm_p03_p_t7.cnf', '10_comm_p03_p_t8.cnf', '10_comm_p03_p_t9.cnf', '10_comm_p04_p_t10.cnf',
                       '10_comm_p04_p_t2.cnf', '10_comm_p04_p_t3.cnf', '10_comm_p04_p_t4.cnf', '10_comm_p04_p_t5.cnf', '10_comm_p04_p_t6.cnf', '10_comm_p04_p_t7.cnf',
                       '10_comm_p04_p_t8.cnf', '10_comm_p04_p_t9.cnf', '10_comm_p05_p_t2.cnf', '10_comm_p05_p_t3.cnf', '10_comm_p05_p_t4.cnf', '10_comm_p05_p_t5.cnf',
                       '10_comm_p05_p_t6.cnf', '10_comm_p05_p_t7.cnf', '10_comm_p05_p_t8.cnf', '10_comm_p05_p_t9.cnf', '10_comm_p10_p_t1.cnf', '10_comm_p10_p_t2.cnf',
                       '11_emptyroom_d12_g6_p_t10.cnf', '11_emptyroom_d12_g6_p_t8.cnf', '11_emptyroom_d12_g6_p_t9.cnf', '11_emptyroom_d16_g8_p_t10.cnf', '11_emptyroom_d16_g8_p_t6.cnf',
                       '11_emptyroom_d16_g8_p_t7.cnf', '11_emptyroom_d16_g8_p_t8.cnf', '11_emptyroom_d16_g8_p_t9.cnf', '11_emptyroom_d20_g10_corners_p_t5.cnf',
                       '11_emptyroom_d20_g10_corners_p_t6.cnf', '11_emptyroom_d20_g10_corners_p_t7.cnf', '11_emptyroom_d20_g10_corners_p_t8.cnf', '11_emptyroom_d20_g10_corners_p_t9.cnf',
                       '11_emptyroom_d24_g12_p_t10.cnf', '11_emptyroom_d24_g12_p_t4.cnf', '11_emptyroom_d24_g12_p_t5.cnf', '11_emptyroom_d24_g12_p_t6.cnf',
                       '11_emptyroom_d24_g12_p_t7.cnf', '11_emptyroom_d24_g12_p_t8.cnf', '11_emptyroom_d24_g12_p_t9.cnf', '11_emptyroom_d28_g14_corners_p_t4.cnf',
                       '11_emptyroom_d28_g14_corners_p_t5.cnf', '11_emptyroom_d28_g14_corners_p_t6.cnf', '11_emptyroom_d28_g14_corners_p_t7.cnf', '11_emptyroom_d28_g14_corners_p_t8.cnf',
                       '11_emptyroom_d28_g14_corners_p_t9.cnf', '14_safe_safe_30_p_t10.cnf', '15_sort_num_s_4_p_t10.cnf', '16_uts_k2_p_t10.cnf']

    medium4 = [ '15_sort_num_s_7_p_t',
        '04_iscas89_s1494_bench.cnf', '04_iscas89_s820_bench.cnf', '04_iscas89_s832_bench.cnf', '04_iscas89_s953_bench.cnf',
               '05_iscas93_s967_bench.cnf',
               '07_blocks_right_2_p_t5.cnf', '07_blocks_right_2_p_t10.cnf', '07_blocks_right_2_p_t8.cnf', '07_blocks_right_3_p_t5.cnf',
               '07_blocks_right_5_p_t2.cnf',
               '13_ring2_r6_p_t9.cnf', '13_ring_5_p_t6.cnf']

    medium_part2 = ['03_iscas85_c1355_isc.cnf', '03_iscas85_c1908_isc.cnf', '03_iscas85_c880_isc.cnf', '05_iscas93_s1269_bench.cnf',
                    '06_iscas99_b04.cnf', '06_iscas99_b11.cnf', '07_blocks_right_3_p_t5.cnf', '07_blocks_right_4_p_t3.cnf',
                    '07_blocks_right_5_p_t2.cnf', '07_blocks_right_6_p_t1.cnf',
                    '09_coins_p01_p_t5.cnf', '09_coins_p02_p_t5.cnf',
                    '09_coins_p03_p_t5.cnf', '09_coins_p04_p_t5.cnf', '09_coins_p05_p_t5.cnf', '09_coins_p10_p_t2.cnf',
                    '11_emptyroom_d8_g4_p_t10.cnf', '13_ring2_r6_p_t10.cnf', '13_ring2_r6_p_t7.cnf', '13_ring2_r6_p_t8.cnf',
                    '13_ring2_r6_p_t9.cnf', '13_ring2_r8_p_t10.cnf', '13_ring2_r8_p_t6.cnf', '13_ring2_r8_p_t7.cnf', '13_ring2_r8_p_t8.cnf',
                    '13_ring2_r8_p_t9.cnf', '13_ring_4_p_t10.cnf', '13_ring_5_p_t10.cnf', '13_ring_5_p_t8.cnf', '13_ring_5_p_t9.cnf',
                    '15_sort_num_s_4_p_t4.cnf', '15_sort_num_s_4_p_t5.cnf', '15_sort_num_s_4_p_t6.cnf', '15_sort_num_s_4_p_t7.cnf',
                    '15_sort_num_s_4_p_t8.cnf', '15_sort_num_s_4_p_t9.cnf', '15_sort_num_s_5_p_t2.cnf', '15_sort_num_s_5_p_t3.cnf',
                    '15_sort_num_s_5_p_t4.cnf', '15_sort_num_s_6_p_t1.cnf', '15_sort_num_s_6_p_t2.cnf', '15_sort_num_s_7_p_t1.cnf',
                    '16_uts_k2_p_t5.cnf', '16_uts_k2_p_t6.cnf', '16_uts_k2_p_t7.cnf', '16_uts_k2_p_t8.cnf', '16_uts_k2_p_t9.cnf',
                    '16_uts_k3_p_t3.cnf', '16_uts_k3_p_t4.cnf', '16_uts_k4_p_t2.cnf']

    filename_only  = filename.split("/")[-1]
    if filename_only.count(".") > 1:
        filename_only = filename_only.replace(".", "_", filename_only.count(".") - 1)
    if filename_only not in medium4:
        print('skip ', filename_only)
        exit(2)
    print("processing ", filename_only)

    # run(alg_type, d, filename,  seed)
    out_folder = "./results_aaai3/" + folder + "_" + inobj + "/"
    # out_folder = "./results2/" + folder + "_" + inobj + "/"
    # out_folder = "./results/" + folder + "_NO_COMPILE_2_" + inobj + "/"


    print(alg_type, inobj, filename, d, out_folder)
    # run(alg_type, d, filename,  seed)

    # out_folder = "./results/" + folder + "_" + inobj + "/"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    NO_COMPILE = False
    # run_sdd(alg_type, filename, seed, out_folder, inobj, NO_COMPILE=NO_COMPILE, part=part, sample_size=sample_size)
    run_at_p_percent_variable(alg_type, filename, seed, out_folder, inobj, NO_COMPILE=True, part=part ,var_percentage=22)

    # inti_compilation("init300", d, filename, out_folder, inobj)
    exit(0)
    # --------------------------------------- end run.sh


    seed = 1234

    # f2 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # f1 = {0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.10, 0.11, 0.12, 0.13, 0.14, 0.15], 1: [0.2, 0.4, 0.6, 0.8, 0.10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.2, 0.21, 0.22, 23, 24]}
    # f1 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24]}
    # f1 = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59]}
    input_weights = [None]
    inobj = ["wscore_otherwg" ]#[ "wscore_half", "wscore_occratio", "wscore_adjoccratio"] #WMC" # = "comp" -- for whcn the compilation has to be performed  -or "count" for literal and such countign
    # inobj = ["score_estimate"]#[ "wscore_half", "wscore_occratio", "wscore_adjoccratio"] #WMC" # = "comp" -- for whcn the compilation has to be performed  -or "count" for literal and such countign
    # inobj = "count"

    # input = "../selective_backbone/"
    # folder = "DatasetA/"
    # out_folder = "./results/data_wsdd_g1/" + folder

    # input = "./input/preproc/"
    input = "./input/"
    # folder = "wmc2022_track2/"
    # out_folder = "./results/wmc2022/" + folder

    # folder = "wmc2020_track2_all/"
    # out_folder = "./results/wmc2020/" + folder

    # folder = "wmc2021_track2_public/"
    # out_folder = "./results/" + folder

    # folder = "wmc2022_track2_public/"
    # folder = "wmc2022_track2_private/"
    folder = "Benchmark_preproc/"
    # out_folder = "./results/" + folder.replace("/", "_"+inobj+"/")

    # folder = "test/"
    # out_folder = "./results/" + folder.replace("/", "_"+inobj+"/")

    # folder = ""
    # out_folder = "./results/" + folder

    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    d = input + folder

    # filename = "instance_K3_N15_M45_01.cnf"
    # f = os.path.join(d, filename)
    # run_sdd(type, d, f, seed, out_folder, inobj)
    # exit(100)

    files = ["./input/Benchmark_preproc/2bitcomp_5.cnf",
              "./input/Benchmark_preproc/2bitmax_6.cnf",
              "./input/Benchmark_preproc/4step.cnf",
              "./input/Benchmark_preproc/5step.cnf",
              "./input/Benchmark_preproc/ais8.cnf",
              "./input/Benchmark_preproc/binsearch.16.pp.cnf",
              "./input/Benchmark_preproc/blasted_case100.cnf",
              "./input/Benchmark_preproc/blasted_case102.cnf",
              "./input/Benchmark_preproc/blasted_case105.cnf",
              "./input/Benchmark_preproc/blasted_case108.cnf",
              "./input/Benchmark_preproc/blasted_case120.cnf",
              "./input/Benchmark_preproc/blasted_case123.cnf",
              './input/Benchmark_preproc/blasted_case200.cnf',
              './input/Benchmark_preproc/blasted_case202.cnf',
              './input/Benchmark_preproc/blasted_case36.cnf',
              './input/Benchmark_preproc/blasted_case54.cnf',
              './input/Benchmark_preproc/blasted_case_3_b14_1.cnf',
              './input/Benchmark_preproc/fs-01.net.cnf',
              './input/Benchmark_preproc/nocountdump14.cnf',
              './input/Benchmark_preproc/or-100-10-10-UC-20.cnf',
              './input/Benchmark_preproc/or-100-10-1-UC-50.cnf',
              './input/Benchmark_preproc/or-100-10-2-UC-60.cnf',
              './input/Benchmark_preproc/or-50-5-1-UC-40.cnf',
              './input/Benchmark_preproc/or-60-20-3-UC-20.cnf',
              './input/Benchmark_preproc/or-60-5-7-UC-20.cnf']
    # files = [f for f in os.listdir(d) ]
    # files = [f for f in os.listdir(d) if re.match('.*\.cnf', f) and "temp" not in f ]
    # files = [f for f in os.listdir(d) if re.match('.*001\.mcc2020\.wcnf', f) and "temp" not in f ]
    print(files)
    # files.sort()
    f_count = 0
    for type in ["static", "dynamic", "random_selection"]:
        for filename in files:
            # if f_count > 0:
            #     break
            f_count += 1
            f =  filename
            # f = os.path.join(d, filename)
            print(filename)

        # for type in ["dynamic", "static", "random", "random_selection"]:
        # for type in [  "random_selection", "dynamic"]: #, "dynamic", "random_selection"] :
            # for type in [ "static", "dynamic"] :
                # for type in ["dynamic", "dynamic_ratio", "static", "static_ratio"]:#, "random", "random_selection"]:
                #     print(type, d, f, seed, out_folder, inobj, input_weights[findex]) #-- looks like its not better with the copy - try to copy vtree as well and minimize?
            for obj in inobj:
                out_folder = "./results/" + folder.replace("/", "_" + obj + "/")
                run_sdd(type, d, f, seed, out_folder, obj)
            # exit(666)
                # inti_compilation("init", d, f, out_folder, inobj)
                # exit(666)
                # utils.util_create_cnf_and_weight_file(f)
            # utils.write_minic2d_file(f, folder)
                # utils.eliminate_trivial_backbone(f)
                # utils.create_scale2_weights_file(f)
            # break

    #-----example-----------
    # seed = 1234
    #
    # f2 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # f1 = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59]}
    # input_weights = [f1, f2]
    #
    # for findex in range(len(input_weights)):
    #     for inobj in input_obj:
    #         out_folder = "./results/test/" + "f" + str(findex + 1) + inobj + "/"
    #         if not os.path.exists(out_folder):
    #             os.makedirs(out_folder)
    #         d = ""
    #         f = "f" + str(findex + 1) + "_" + inobj + ".cnf"
    #         for type in ["dynamic", "static"]:  # , "random", "random_selection"]:
    #             # for type in ["dynamic", "dynamic_ratio", "static", "static_ratio"]:#, "random", "random_selection"]:
    #             #     print(type, d, f, seed, out_folder, inobj, input_weights[findex]) #-- looks like its not better with the copy - try to copy vtree as well and minimize?
    #             run_sdd(type, d, f, seed, out_folder, inobj, input_weights[
    #                 findex])  # -- looks like its not better with the copy - try to copy vtree as well and minimize?

    #-----example-----------


    # d = "./DatasetA/"
    # d = "./iscas/iscas85/"
    # d = "./iscas/iscas99/"
    # alg_type = "init"

    # run_sdd(alg_type, d, seed)
    # run(alg_type, d, seed)

    # if "random" in type or "ls" in type:
    #     stats_file = d + "dataset_stats_" + type + "_"+str(seed)+".csv"
    # else:
    #     stats_file = d+"dataset_stats_"+type+".csv"
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']

    # run(type, d, seed, stats_file, columns)
    # exit(10)

    # for d in ["./DatasetA/"]:
    #     # ,"./DatasetB/", "./iscas/", "./BayesianNetwork/", "./Planning/"]:
    #     all_type = ["init",  "random","random_selection", "static","static_ratio", "dynamic", "dynamic_ratio"]
    #     for alg_type in all_type:
    #         seed = 1234
    #         print("============================",type,"============================")
    #         run_sdd(alg_type, d, seed)

    # exit(10)

    ################################# Iterate through files ########################################
    # folder = "DatasetA/"
    # out_folder = "./aaai_data/output/sdd/" +folder
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # d = "./aaai_data/input/"+folder
    # files = [f for f in os.listdir(d) if re.match('.*cnf', f)]
    # files.sort()
    # f_count = 0
    #     # ,"./DatasetB/", "./iscas/", "./BayesianNetwork/", "./Planning/"]:
    # all_type = ["init"]#,"static","static_ratio"]
    #     # ,  "random","random_selection", "static","static_ratio", "dynamic", "dynamic_ratio"]
    # for alg_type in all_type:
    #     for filename in files:
    #         f_count += 1
    #         f = os.path.join(d, filename)
    #         print("Input file:", f)
    #         run_sdd(alg_type, f, seed, out_folder)
    #
    # exit(10)
    ################################# Iterate through files ########################################

    #random
    # type = "random_1234"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_rand = evaluate.ExprData(columns)
    # expr_data_rand.read_stats_file(stats_file)
    #
    # # static
    # type = "static"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_static= evaluate.ExprData(columns)
    # expr_data_static.read_stats_file(stats_file)
    #
    #
    # # dynamic
    # type = "dynamic"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_dynamic = evaluate.ExprData(columns)
    # expr_data_dynamic.read_stats_file(stats_file)
    #
    # # dynamic
    # type = "dynamic_ratio"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_dynamic2 = evaluate.ExprData(columns)
    # expr_data_dynamic2.read_stats_file(stats_file)
    #
    # # static_ratio
    # # type = "static_ratio"
    # # stats_file = d + "dataset_stats_" + type + ".csv"
    # # expr_data_static2 = evaluate.ExprData(columns)
    # # expr_data_static2.read_stats_file(stats_file)
    #
    # # random selection
    # type = "random_selection_1234"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_random2 = evaluate.ExprData(columns)
    # expr_data_random2.read_stats_file(stats_file)
    #
    # # Loca search
    # type = "ls"
    # stats_file = d + "dataset_stats_" + type +"_"+str(seed)+ ".csv"
    # expr_data_ls = evaluate.ExprData(columns)
    # expr_data_ls.read_stats_file(stats_file)
    #
    # plot_type = "raw"
    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic,expr_data_dynamic2, expr_data_random2,expr_data_ls], "efficiency", ["random", "static", "dynamic", "dynamic_ratio", "random_selection_1234", "ls"], "init")
    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic,expr_data_dynamic2, expr_data_random2,expr_data_ls], "ratio", ["random", "static", "dynamic", "dynamic_ratio", "random_selection_1234", "ls"], "init")
    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic,expr_data_dynamic2, expr_data_random2, expr_data_ls], "MC", ["random", "static", "dynamic", "dynamic_ratio", "random_selection_1234", "ls"], plot_type)
    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic,expr_data_dynamic2, expr_data_random2, expr_data_ls], "dag_size", ["random", "static", "dynamic", "dynamic_ratio", "random_selection_1234", "ls"], plot_type)

    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic], "dag_size", ["random", "static", "dynamic"], plot_type)

    # expr_data.plot_all_exprs("./DatasetA/all_plots_"+column_name+"_"+type+".png", column_name)

    # expr_data.plot_all_efficiencies_percentage()
    # #
    # expr_data2 = evaluate.ExprData(columns)
    # expr_data2.read_stats_file("./DatasetA/dataset_stats_static.csv")
    #
    # expr_data3 = evaluate.ExprData(columns)
    # expr_data3.read_stats_file("./DatasetA/dataset_stats_random.csv")
    # #
    # evaluate.plot_multiple([expr_data, expr_data2, expr_data3], "efficiency")

    # cnf = _cnf.CNF()
    # cnf.load_file("./nqueens_4.cnf")
    # selctive_backbone_vars, best_cost = greedy_pWSB(cnf, 10)
    # print(selctive_backbone_vars, best_cost)


    # for i in range(1,10):
    #     print("---------------------",i,"---------------------")
        # print("----","greedyI","----")
        # csp = CSP()
        # csp.init_with_solutions(solutions)
        # selctive_backbone_vars, best_cost = greedy_pWSB(csp, i)
        # # print(sorted(selctive_backbone_vars), best_cost)
        # calculate_nb_covered_solutions(solutions, selctive_backbone_vars)
        # print("----", "greedyO", "----")
        # csp = CSP()
        # csp.init_with_solutions(solutions)
        # selctive_backbone_vars, best_cost = greedy_pWSB_noremove(csp, i)
        # calculate_nb_covered_solutions(solutions, selctive_backbone_vars)
        #print(sorted(selctive_backbone_vars), best_cost)
        # print("----", "global best", "----")
        # max_selected_vars, max_nb_solutions = find_global_best_p_selective_backbone(solutions, i)
        # # print(max_selected_vars, max_nb_solutions)
        # calculate_nb_covered_solutions(solutions, max_selected_vars)


