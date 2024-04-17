#We need a function to score the current partial assignment plus the addition of an assignment.
# score of assignment
import numpy as np
import itertools
import WCNFmodelSDD as _wcnfSDD
import os
import evaluate
import re
import queue as _queue
import random
import time
import sys
import math

def get_best_assignment(csp, obj_type):
    "Assumption is that here we already have the bdd extended with the partial assignment"
    best_variable = 0
    best_value = 0
    best_cost = 0
    temp_root = csp.root_node
    # temp_root.ref() # use this if minimizinf vtree
    # csp.sdd_manager.auto_gc_and_minimize_on()
    csp.root_node.ref()
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                if obj_type == "WMC":
                    score_of_assignment, size, node_count, temp_root = csp.check_wmc_of(v, value)
                elif obj_type == "g2":
                    score_of_assignment, size, node_count, temp_root  = csp.calculate_g2(v, value)
                elif "ratio" in obj_type :
                    score_of_assignment, size, node_count, temp_root = csp.check_wmc_ratio_of(v, value)
                else:
                    print("ERROR")
                    exit(666)
                if score_of_assignment >= best_cost:
                    best_variable=v
                    best_value=value
                    best_cost=score_of_assignment
                    best_size = size
                    best_node_count = node_count
                    best_root = temp_root.copy()
                    #look out this might take a lot of memory might need to clear at some stage
                    # temp_root.deref()

                    # print("best", best_variable , best_value)
                    # best_root.ref() # use this if minimizinf vtree
    # print("------------------------", best_variable,best_value)
    # print("best:",best_variable,best_value, best_cost, best_size, best_node_count, best_root)
    # print(csp.root_node.size(), csp.root_node.count(), best_root.size(), best_root.count())
    # t = best_root
    # best_root.vtree().minimize(best_root.manager)
    # csp.sdd_manager.minimize()
    # print(csp.root_node.size(),csp.root_node.count()  , best_root.size(), best_root.count() )
    # best_root.ref()
    # csp.sdd_manager.minimize()
    # best_size = best_root.size()
    return best_variable,best_value, best_cost, best_size, best_node_count, best_root

def dynamic_greedy_pWSB(csp, max_p, obj_type,logger):
    pa = csp.partial_assignment
    p = len(pa.assigned)
    print(p, max_p)
    while p < max_p:
        #select the assignment that maximizes the score
        p += 1
        # print("-------", p)
        best_variable,best_value, best_cost, best_size, best_node_count, best_root = get_best_assignment(csp,obj_type)

        elapsed = logger.get_time_elapsed()
        mc = best_root.manager.global_model_count(best_root)
        wmc= best_cost
        ## calculate both wmc and g2
        # wmc, a, b, c = csp.check_wmc_of(best_variable, best_value)
        # topcount, size, count, condition_node = csp.calculate_g2(best_variable, best_value)

        if wmc == 0.0:
            logger.log([p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size,best_node_count, elapsed, 0, 0])
            return
        else:
            logWMC = math.log10(wmc)
            logger.log([p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size,best_node_count, elapsed, wmc, logWMC])


        csp.extend_assignment(best_variable,best_value,best_cost, best_root)
        # print( p, best_variable, best_value, best_cost)

    # print("p=", p)
    # print("variable,value,score")
    # for i in result:
    #     print(','.join(map(str, i)))

def order_var_assignments(csp, obj_type):
    assign_queue = _queue.PriorityQueue()
    csp.root_node.ref()
    csp.sdd_manager.auto_gc_and_minimize_on()
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                # print("var , val ", v, value)
                # if v == 15:
                #     print("stop") #looks like sdd eliminated a var when x15=1
                if obj_type == "WMC":
                    score_of_assignment, size, node_count, temp_root  = csp.check_wmc_of(v, value)
                    # print("score_of_assignment, size, node_count, temp_root ", score_of_assignment, size, node_count )
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
                weight = csp.literal_weights[value][v - 1]
                assign_queue.put(tuple([-1*score_of_assignment,-1*weight, v, value, size, node_count]))
    # csp.root_node.deref() #not sure if this is needed
    return assign_queue

def random_var_assignments(csp,seed):
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

def random_pWSB(csp, seed, logger):
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

            # need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])
            score_of_assignment, size, node_count, temp_root = csp.check_wmc_of(variable, value)
            # if score_of_assignment == 0:
            #     continue
            p += 1
            seen_vars.add(variable)
            elapsed = logger.get_time_elapsed()
            mc = temp_root.manager.global_model_count(temp_root)
            wmc = score_of_assignment
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, -1])
            csp.extend_assignment(variable, value, score_of_assignment, temp_root)

def random_selection_pWSB(csp, seed,logger):
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

    seen_vars = set()
    for variable in assign_queue:

            # need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])

        value=0
        wmc = 0
        score_of_assignment_0, size0, node_count0, temp_root0 = csp.check_wmc_of(variable, 0)
        score_of_assignment_1, size1, node_count1, temp_root1 = csp.check_wmc_of(variable, 1)

        if score_of_assignment_0 > score_of_assignment_1:
            score_of_assignment = score_of_assignment_0
            size = size0
            temp_root = temp_root0
            node_count = node_count0
        else:
            score_of_assignment = score_of_assignment_1
            size = size1
            temp_root = temp_root1
            node_count = node_count1
            value=1
        p += 1
        elapsed = logger.get_time_elapsed()
        mc = temp_root.manager.global_model_count(temp_root)
        wmc = score_of_assignment
        if score_of_assignment == 0.0:
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, 0])
            return
        else:
            logWMC = math.log10(wmc)
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, logWMC])
        csp.extend_assignment(variable, value, score_of_assignment, temp_root)

def static_greedy_pWSB(csp, obj_type,logger):
    print("STATIC")
    assign_queue = order_var_assignments(csp, obj_type)
    p = 0
    seen_vars = set()
    while not assign_queue.empty():

        item = assign_queue.get() # tuple of ( score_of_assignment,var, value, size, node_count )
        score_of_assignment = abs(item[0])
        weight = abs(item[1])
        variable = item[2]
        value = item[3]
        sdd_size = item[4]
        node_count = item[5]
        print("assign", p, variable, value, len(seen_vars), assign_queue.qsize())
        if variable not in seen_vars:
            p += 1
            seen_vars.add(variable)
            #need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])
            if obj_type == "WMC":
                score_of_assignment, size, node_count, temp_root= csp.check_wmc_of(variable, value)
            elif obj_type == "static_ratio":
                score_of_assignment, size, node_count, temp_root = csp.check_wmc_ratio_of(variable, value)
            elif obj_type == "g2":
                score_of_assignment, size, node_count, temp_root = csp.calculate_g2(variable, value)
            else:
                print("ERROR")
                exit(666)

            elapsed = logger.get_time_elapsed()
            #is size sdd size or root size?
            mc = temp_root.manager.global_model_count(temp_root)
            # logger.log( [p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, score_of_assignment])

            ## calculate both wmc and g2
            # wmc, a, b, c = csp.check_wmc_of(variable, value)
            # topcount, size, count, condition_node = csp.calculate_g2(variable, value)
            wmc = score_of_assignment
            if wmc == 0.0:
                logger.log([p,  variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, 0])
                return
            else:
                logWMC = math.log10(wmc)
                logger.log([p,  variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed, wmc, logWMC])

            csp.extend_assignment(variable, value, score_of_assignment, temp_root)

def print_mc_per_vars(csp, logger):
    assign_queue = order_var_assignments(csp, "mc")
    p= 0
    while not assign_queue.empty():
        print("=======================", p)
        item = assign_queue.get()  # tuple of ( score_of_assignment,var, value, size, node_count )
        score_of_assignment = abs(item[0])
        weight = abs(item[1])
        variable = item[2]
        value = item[3]
        sdd_size = item[4]
        node_count = item[5]
        p += 1

        elapsed = logger.get_time_elapsed()
        logger.log([p, variable, value, csp.n, len(csp.cls), score_of_assignment, sdd_size, node_count, elapsed])


def run_sdd(alg_type, d, filename, seed, out_folder, obj_type, literal_weights):
    # obj_type: mc or g2
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC",  "SDD size", 'node_count', 'time', 'WMC', "logWMC"]
    if "random" in alg_type or "ls" in alg_type:
        # stats_file = d + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
    else:
        # stats_file = d + "dataset_stats_" + alg_type + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + ".csv"


    expr_data = evaluate.ExprData(columns)
    logger = evaluate.Logger(stats_file, columns, expr_data, out_folder, compile=True)

    print(filename)
    all_start = time.perf_counter()
    logger.log_expr(filename)
    start = time.perf_counter()
    logger.set_start_time(start)
    cnf = _wcnfSDD.WCNF(logger, scalar=3)
    # wcnf = WCNF()
    # f2_literal_weights = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # f1_literal_weights = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59] }
    # cnf.create_example(literal_weights)
    print(literal_weights)
    b = cnf.load_file(filename, literal_weights)
    # cnf.set_weights(literal_weights)
    # b = cnf.load_file_with_apply(f)
    print(logger.get_time_elapsed())
    # if not b:
    #     return

    #TODO: iterate here between the alf types to avoid spending time for loding initial compilation -
    # needt to save first line in log to add it to all consequtive stat files
    maxp = len(cnf.literals)
    if alg_type == "dynamic":
        dynamic_greedy_pWSB(cnf, maxp, obj_type, logger)
    elif alg_type == "dynamic_ratio":
        dynamic_greedy_pWSB(cnf, maxp, "dynamic_ratio",logger)
    elif alg_type == "static":
        static_greedy_pWSB(cnf, obj_type, logger)
    elif alg_type == "static_ratio":
        static_greedy_pWSB(cnf, "static_ratio",logger)
    elif "random" == alg_type:
        random_pWSB(cnf, seed,logger)
    elif "random_selection" in alg_type:
        random_selection_pWSB(cnf, seed,logger)
    # elif alg_type == "ls":
    #     local_search_pWSB(cnf, maxp, seed + f_count,logger)
    elif alg_type == "init":
        print_mc_per_vars(cnf,logger)
    else:
        print("Something wrong")

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

    # --------------------------------------- run.sh
    print(sys.argv)
    inobj = sys.argv[4]
    d = sys.argv[1]  # "./input/wmc2022_track2_private/"
    folder = d.split("/")[-2]
    filename = sys.argv[2]
    alg_type = sys.argv[3]
    # run(alg_type, d, filename,  seed)
    out_folder = "./results/sdd/" + folder + "_" + inobj + "/"
    print(alg_type, d, filename, out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    not_compiling = ['./input/Benchmark/20_rd_r45.cnf', './input/Benchmark/23_rd_r45.cnf', './input/Benchmark/2bitadd_11.cnf', './input/Benchmark/2bitadd_12.cnf', './input/Benchmark/3bitadd_31.cnf', './input/Benchmark/3bitadd_32.cnf', './input/Benchmark/5_100_sd_schur.cnf', './input/Benchmark/5_140_sd_schur.cnf', './input/Benchmark/54.sk_12_97.cnf', './input/Benchmark/ais12.cnf', './input/Benchmark/binsearch.16.cnf', './input/Benchmark/binsearch.32.cnf', './input/Benchmark/blasted_case_0_b12_even1.cnf', './input/Benchmark/blasted_case_0_b12_even2.cnf', './input/Benchmark/blasted_case_0_b12_even3.cnf', './input/Benchmark/blasted_case_0_ptb_1.cnf', './input/Benchmark/blasted_case_0_ptb_2.cnf', './input/Benchmark/blasted_case104.cnf', './input/Benchmark/blasted_case12.cnf', './input/Benchmark/blasted_case141.cnf', './input/Benchmark/blasted_case142.cnf', './input/Benchmark/blasted_case_1_4_b14_even.cnf', './input/Benchmark/blasted_case_1_b12_even1.cnf', './input/Benchmark/blasted_case_1_b12_even2.cnf', './input/Benchmark/blasted_case_1_b12_even3.cnf', './input/Benchmark/blasted_case1_b14_even3.cnf', './input/Benchmark/blasted_case_1_b14_even.cnf', './input/Benchmark/blasted_case_1_ptb_2.cnf', './input/Benchmark/blasted_case_2_b12_even1.cnf', './input/Benchmark/blasted_case_2_b12_even2.cnf', './input/Benchmark/blasted_case_2_b12_even3.cnf', './input/Benchmark/blasted_case_2_b14_even.cnf', './input/Benchmark/blasted_case_2_ptb_2.cnf', './input/Benchmark/blasted_case_3_4_b14_even.cnf', './input/Benchmark/blasted_case37.cnf', './input/Benchmark/blasted_case3_b14_even3.cnf', './input/Benchmark/blasted_case42.cnf', './input/Benchmark/blasted_squaring10.cnf', './input/Benchmark/blasted_squaring11.cnf', './input/Benchmark/blasted_squaring12.cnf', './input/Benchmark/blasted_squaring14.cnf', './input/Benchmark/blasted_squaring16.cnf', './input/Benchmark/blasted_squaring1.cnf', './input/Benchmark/blasted_squaring2.cnf', './input/Benchmark/blasted_squaring30.cnf', './input/Benchmark/blasted_squaring3.cnf', './input/Benchmark/blasted_squaring4.cnf', './input/Benchmark/blasted_squaring5.cnf', './input/Benchmark/blasted_squaring60.cnf', './input/Benchmark/blasted_squaring6.cnf', './input/Benchmark/blasted_squaring70.cnf', './input/Benchmark/blasted_squaring7.cnf', './input/Benchmark/blasted_squaring8.cnf', './input/Benchmark/blasted_squaring9.cnf', './input/Benchmark/blasted_TR_b12_1_linear.cnf', './input/Benchmark/blasted_TR_b12_2_linear.cnf', './input/Benchmark/blasted_TR_b12_even2_linear.cnf', './input/Benchmark/blasted_TR_b12_even3_linear.cnf', './input/Benchmark/blasted_TR_b12_even7_linear.cnf', './input/Benchmark/blasted_TR_b14_2_linear.cnf', './input/Benchmark/blasted_TR_b14_3_linear.cnf', './input/Benchmark/blasted_TR_b14_even2_linear.cnf', './input/Benchmark/blasted_TR_b14_even3_linear.cnf', './input/Benchmark/blasted_TR_b14_even_linear.cnf', './input/Benchmark/blasted_TR_device_1_even_linear.cnf', './input/Benchmark/blasted_TR_ptb_1_linear.cnf', './input/Benchmark/blasted_TR_ptb_2_linear.cnf', './input/Benchmark/blockmap_20_03.net.cnf', './input/Benchmark/blockmap_22_02.net.cnf', './input/Benchmark/bmc-galileo-8.cnf', './input/Benchmark/bmc-galileo-9.cnf', './input/Benchmark/bmc-ibm-10.cnf', './input/Benchmark/bmc-ibm-11.cnf', './input/Benchmark/bmc-ibm-12.cnf', './input/Benchmark/bmc-ibm-6.cnf', './input/Benchmark/bmc-ibm-7.cnf', './input/Benchmark/C129_FR.cnf', './input/Benchmark/c1355.isc.cnf', './input/Benchmark/C140_FV.cnf', './input/Benchmark/C140_FW.cnf', './input/Benchmark/C168_FW.cnf', './input/Benchmark/c1908.isc.cnf', './input/Benchmark/C202_FS.cnf', './input/Benchmark/C202_FW.cnf', './input/Benchmark/C203_FCL.cnf', './input/Benchmark/C203_FS.cnf', './input/Benchmark/C203_FW.cnf', './input/Benchmark/C208_FA.cnf', './input/Benchmark/C208_FC.cnf', './input/Benchmark/C210_FS.cnf', './input/Benchmark/C210_FW.cnf', './input/Benchmark/C220_FV.cnf', './input/Benchmark/C220_FW.cnf', './input/Benchmark/c2670.isc.cnf', './input/Benchmark/c7552.isc.cnf', './input/Benchmark/c880.isc.cnf', './input/Benchmark/compress.sk_17_291.cnf', './input/Benchmark/ConcreteRoleAffectationService.sk_119_273.cnf', './input/Benchmark/diagStencilClean.sk_41_36.cnf', './input/Benchmark/diagStencil.sk_35_36.cnf', './input/Benchmark/enqueueSeqSK.sk_10_42.cnf', './input/Benchmark/fphp-010-020.cnf', './input/Benchmark/fphp-015-020.cnf', './input/Benchmark/fs-07.net.cnf', './input/Benchmark/fs-10.net.cnf', './input/Benchmark/fs-13.net.cnf', './input/Benchmark/fs-16.net.cnf', './input/Benchmark/fs-19.net.cnf', './input/Benchmark/fs-22.net.cnf', './input/Benchmark/fs-25.net.cnf', './input/Benchmark/fs-28.net.cnf', './input/Benchmark/fs-29.net.cnf', './input/Benchmark/isolateRightmost.sk_7_481.cnf', './input/Benchmark/jburnim_morton.sk_13_530.cnf', './input/Benchmark/karatsuba.sk_7_41.cnf', './input/Benchmark/lang12.cnf', './input/Benchmark/lang15.cnf', './input/Benchmark/lang16.cnf', './input/Benchmark/lang19.cnf', './input/Benchmark/lang20.cnf', './input/Benchmark/lang23.cnf', './input/Benchmark/lang24.cnf', './input/Benchmark/lang27.cnf', './input/Benchmark/lang28.cnf', './input/Benchmark/listReverse.sk_11_43.cnf', './input/Benchmark/log2.sk_72_391.cnf', './input/Benchmark/log-5.cnf', './input/Benchmark/logcount.sk_16_86.cnf', './input/Benchmark/LoginService2.sk_23_36.cnf', './input/Benchmark/logistics.c.cnf', './input/Benchmark/logistics.d.cnf', './input/Benchmark/ls10-normalized.cnf', './input/Benchmark/ls11-normalized.cnf', './input/Benchmark/ls12-normalized.cnf', './input/Benchmark/ls13-normalized.cnf', './input/Benchmark/ls14-normalized.cnf', './input/Benchmark/ls15-normalized.cnf', './input/Benchmark/ls16-normalized.cnf', './input/Benchmark/ls8-normalized.cnf', './input/Benchmark/ls9-normalized.cnf', './input/Benchmark/lss.sk_6_7.cnf', './input/Benchmark/mastermind_03_08_05.net.cnf', './input/Benchmark/mastermind_10_08_03.net.cnf', './input/Benchmark/nocountdump4.cnf', './input/Benchmark/or-100-10-10.cnf', './input/Benchmark/or-100-10-10-UC-10.cnf', './input/Benchmark/or-100-10-1.cnf', './input/Benchmark/or-100-10-1-UC-10.cnf', './input/Benchmark/or-100-10-2.cnf', './input/Benchmark/or-100-10-2-UC-10.cnf', './input/Benchmark/or-100-10-3.cnf', './input/Benchmark/or-100-10-3-UC-10.cnf', './input/Benchmark/or-100-10-4.cnf', './input/Benchmark/or-100-10-4-UC-10.cnf', './input/Benchmark/or-100-10-5.cnf', './input/Benchmark/or-100-10-5-UC-10.cnf', './input/Benchmark/or-100-10-6.cnf', './input/Benchmark/or-100-10-6-UC-10.cnf', './input/Benchmark/or-100-10-7.cnf', './input/Benchmark/or-100-10-7-UC-10.cnf', './input/Benchmark/or-100-10-8.cnf', './input/Benchmark/or-100-10-8-UC-10.cnf', './input/Benchmark/or-100-10-9.cnf', './input/Benchmark/or-100-10-9-UC-10.cnf', './input/Benchmark/or-100-20-10.cnf', './input/Benchmark/or-100-20-10-UC-10.cnf', './input/Benchmark/or-100-20-10-UC-20.cnf', './input/Benchmark/or-100-20-1.cnf', './input/Benchmark/or-100-20-1-UC-10.cnf', './input/Benchmark/or-100-20-1-UC-20.cnf', './input/Benchmark/or-100-20-1-UC-30.cnf', './input/Benchmark/or-100-20-2.cnf', './input/Benchmark/or-100-20-2-UC-10.cnf', './input/Benchmark/or-100-20-3.cnf', './input/Benchmark/or-100-20-3-UC-10.cnf', './input/Benchmark/or-100-20-3-UC-20.cnf', './input/Benchmark/or-100-20-4.cnf', './input/Benchmark/or-100-20-4-UC-10.cnf', './input/Benchmark/or-100-20-4-UC-30.cnf', './input/Benchmark/or-100-20-5.cnf', './input/Benchmark/or-100-20-5-UC-10.cnf', './input/Benchmark/or-100-20-5-UC-20.cnf', './input/Benchmark/or-100-20-6.cnf', './input/Benchmark/or-100-20-6-UC-10.cnf', './input/Benchmark/or-100-20-6-UC-20.cnf', './input/Benchmark/or-100-20-7.cnf', './input/Benchmark/or-100-20-7-UC-10.cnf', './input/Benchmark/or-100-20-7-UC-20.cnf', './input/Benchmark/or-100-20-8.cnf', './input/Benchmark/or-100-20-8-UC-10.cnf', './input/Benchmark/or-100-20-8-UC-20.cnf', './input/Benchmark/or-100-20-9.cnf', './input/Benchmark/or-100-20-9-UC-10.cnf', './input/Benchmark/or-100-5-10.cnf', './input/Benchmark/or-100-5-10-UC-10.cnf', './input/Benchmark/or-100-5-1.cnf', './input/Benchmark/or-100-5-2.cnf', './input/Benchmark/or-100-5-2-UC-10.cnf', './input/Benchmark/or-100-5-3.cnf', './input/Benchmark/or-100-5-4.cnf', './input/Benchmark/or-100-5-4-UC-10.cnf', './input/Benchmark/or-100-5-4-UC-20.cnf', './input/Benchmark/or-100-5-5.cnf', './input/Benchmark/or-100-5-5-UC-10.cnf', './input/Benchmark/or-100-5-6.cnf', './input/Benchmark/or-100-5-7.cnf', './input/Benchmark/or-100-5-7-UC-10.cnf', './input/Benchmark/or-100-5-8.cnf', './input/Benchmark/or-100-5-8-UC-10.cnf', './input/Benchmark/or-100-5-9.cnf', './input/Benchmark/or-60-10-1.cnf', './input/Benchmark/or-60-10-3.cnf', './input/Benchmark/or-60-10-4.cnf', './input/Benchmark/or-60-10-6.cnf', './input/Benchmark/or-60-10-7.cnf', './input/Benchmark/or-60-10-8.cnf', './input/Benchmark/or-60-10-9.cnf', './input/Benchmark/or-60-20-10.cnf', './input/Benchmark/or-60-20-2.cnf', './input/Benchmark/or-60-20-3.cnf', './input/Benchmark/or-60-20-4.cnf', './input/Benchmark/or-60-20-6.cnf', './input/Benchmark/or-60-20-7.cnf', './input/Benchmark/or-60-20-9.cnf', './input/Benchmark/or-60-5-1.cnf', './input/Benchmark/or-60-5-2.cnf', './input/Benchmark/or-60-5-3.cnf', './input/Benchmark/or-60-5-6.cnf', './input/Benchmark/or-60-5-8.cnf', './input/Benchmark/or-60-5-9.cnf', './input/Benchmark/or-70-10-10.cnf', './input/Benchmark/or-70-10-1.cnf', './input/Benchmark/or-70-10-2.cnf', './input/Benchmark/or-70-10-3.cnf', './input/Benchmark/or-70-10-4.cnf', './input/Benchmark/or-70-10-5.cnf', './input/Benchmark/or-70-10-6.cnf', './input/Benchmark/or-70-10-7.cnf', './input/Benchmark/or-70-10-8.cnf', './input/Benchmark/or-70-10-9.cnf', './input/Benchmark/or-70-20-10.cnf', './input/Benchmark/or-70-20-10-UC-20.cnf', './input/Benchmark/or-70-20-1.cnf', './input/Benchmark/or-70-20-2.cnf', './input/Benchmark/or-70-20-3.cnf', './input/Benchmark/or-70-20-3-UC-10.cnf', './input/Benchmark/or-70-20-4.cnf', './input/Benchmark/or-70-20-4-UC-10.cnf', './input/Benchmark/or-70-20-5.cnf', './input/Benchmark/or-70-20-6.cnf', './input/Benchmark/or-70-20-7.cnf', './input/Benchmark/or-70-20-8.cnf', './input/Benchmark/or-70-20-8-UC-10.cnf', './input/Benchmark/or-70-20-9.cnf', './input/Benchmark/or-70-5-10.cnf', './input/Benchmark/or-70-5-1.cnf', './input/Benchmark/or-70-5-2.cnf', './input/Benchmark/or-70-5-3.cnf', './input/Benchmark/or-70-5-4.cnf', './input/Benchmark/or-70-5-5.cnf', './input/Benchmark/or-70-5-6.cnf', './input/Benchmark/or-70-5-6-UC-10.cnf', './input/Benchmark/or-70-5-7.cnf', './input/Benchmark/or-70-5-8.cnf', './input/Benchmark/or-70-5-9.cnf', './input/Benchmark/par32-1-c.cnf', './input/Benchmark/par32-1.cnf', './input/Benchmark/par32-2-c.cnf', './input/Benchmark/par32-2.cnf', './input/Benchmark/par32-3-c.cnf', './input/Benchmark/par32-3.cnf', './input/Benchmark/par32-4-c.cnf', './input/Benchmark/par32-4.cnf', './input/Benchmark/par32-5-c.cnf', './input/Benchmark/par32-5.cnf', './input/Benchmark/parity.sk_11_11.cnf', './input/Benchmark/partition.sk_22_155.cnf', './input/Benchmark/Pollard.sk_1_10.cnf', './input/Benchmark/prob005.pddl.cnf', './input/Benchmark/prob012.pddl.cnf', './input/Benchmark/reverse.sk_11_258.cnf', './input/Benchmark/s13207a_15_7.cnf', './input/Benchmark/s13207a_3_2.cnf', './input/Benchmark/s13207a_7_4.cnf', './input/Benchmark/s15850a_15_7.cnf', './input/Benchmark/s15850a_3_2.cnf', './input/Benchmark/s15850a_7_4.cnf', './input/Benchmark/s38417_15_7.cnf', './input/Benchmark/s38417_3_2.cnf', './input/Benchmark/s38417_7_4.cnf', './input/Benchmark/s38584_15_7.cnf', './input/Benchmark/s38584_3_2.cnf', './input/Benchmark/s38584_7_4.cnf', './input/Benchmark/s5378a_15_7.cnf', './input/Benchmark/s5378a_3_2.cnf', './input/Benchmark/s5378a_7_4.cnf', './input/Benchmark/s9234a_15_7.cnf', './input/Benchmark/s9234a_3_2.cnf', './input/Benchmark/s9234a_7_4.cnf', './input/Benchmark/sat-grid-pbl-0030.cnf', './input/Benchmark/scenarios_aig_traverse.sb.pl.sk_5_102.cnf', './input/Benchmark/scenarios_lldelete1.sb.pl.sk_6_409.cnf', './input/Benchmark/scenarios_llinsert2.sb.pl.sk_6_407.cnf', './input/Benchmark/scenarios_llreverse.sb.pl.sk_8_25.cnf', './input/Benchmark/scenarios_lltraversal.sb.pl.sk_5_23.cnf', './input/Benchmark/scenarios_tree_delete4.sb.pl.sk_4_114.cnf']

    if filename in not_compiling:
        print("not compiling")
        exit(555)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if filename > "./input/Benchmark/nocountdump6.cnf":
        run_sdd(alg_type, d, filename, seed, out_folder, inobj, None)
    exit(0)
    # --------------------------------------- end run.sh



    seed = 1234

    # f2 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # f1 = {0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.10, 0.11, 0.12, 0.13, 0.14, 0.15], 1: [0.2, 0.4, 0.6, 0.8, 0.10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.2, 0.21, 0.22, 23, 24]}
    # f1 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24]}
    # f1 = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59]}
    input_weights = [None]
    inobj = "WMC"


    # input = "../selective_backbone/"
    # folder = "DatasetA/"
    # out_folder = "./results/data_wsdd_g1/" + folder


    # input = "./input/preproc/"
    # folder = "wmc2022_track2_private/"
    input= "./input/"
    folder = "Benchmark/"
    out_folder = "./results/sdd/" + folder.replace("/", "_" + inobj + "/")


    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    d = input + folder
    #./input/Benchmark/nocountdump14.cnf
    files = [f for f in os.listdir(d) if re.match('.*nocountdump14.cnf', f) and "temp" not in f]
    files.sort()
    f_count = 0
    for filename in files:
        if f_count > 0:
            break
        f_count += 1
        f = os.path.join(d, filename)
        print(filename)
        for type in [ "static" ]: #, "random", "random_selection"]:
            # for type in ["dynamic", "dynamic_ratio", "static", "static_ratio"]:#, "random", "random_selection"]:
            #     print(type, d, f, seed, out_folder, inobj, input_weights[findex]) #-- looks like its not better with the copy - try to copy vtree as well and minimize?
            run_sdd(type, d, f, seed, out_folder, inobj, None) #-- looks like its not better with the copy - try to copy vtree as well and minimize?
    exit(11)
    #-----example-----------
    # seed = 1234
    #
    # f2 = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [2, 4, 6, 8, 10, 12, 14, 16, 18]}
    # f1 = {0: [49, 48, 47, 46, 45, 44, 43, 42, 41], 1: [51, 52, 53, 54, 55, 56, 57, 58, 59]}
    # input_weights = [f1, f2]
    #
    # input_obj = ["g1", "g2"]
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
    inobj = "WMC"
    input = "./input/preproc/"
    folder = "wmc2022_track2_private/"
    out_folder = "./results/sdd/" + folder.replace("/", "_" + inobj + "/")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    d = input + folder
    files = [f for f in os.listdir(d) if re.match('.*010\.cnf', f) and "temp" not in f]
    # files = [f for f in os.listdir(d) if re.match('.*001\.mcc2020\.wcnf', f) and "temp" not in f ]
    print(files)
    files.sort()
    f_count = 0
    for filename in files:
        f_count += 1
        f = os.path.join(d, filename)
        print(filename)
        # for type in ["dynamic", "static", "random", "random_selection"]:
        for type in ["dynamic"]:
            # for type in [ "static", "dynamic"] :
            # for type in ["dynamic", "dynamic_ratio", "static", "static_ratio"]:#, "random", "random_selection"]:
            #     print(type, d, f, seed, out_folder, inobj, input_weights[findex]) #-- looks like its not better with the copy - try to copy vtree as well and minimize?
            run_sdd(type, d, f, seed, out_folder,inobj, None)  # -- looks like its not better with the copy - try to copy vtree as well and minimize?
            #run_sdd(alg_type, d, filename, seed, out_folder, obj_type, literal_weights)

    exit(10)
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


