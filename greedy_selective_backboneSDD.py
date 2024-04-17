#We need a function to score the current partial assignment plus the addition of an assignment.
# score of assignment
import numpy as np
import itertools
import CNFmodelSDD as _cnfSDD
import os
import evaluate
import re
import queue as _queue
import random
import time
import sys


def get_best_assignment(csp, obj_type):
    "Assumption is that here we already have the bdd extended with the partial assignment"
    best_variable = 0
    best_value = 0
    best_cost = 0
    # temp_root.ref() # use this if minimizinf vtree
    # csp.sdd_manager.auto_gc_and_minimize_on()
    csp.root_node.ref()
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                if obj_type == "mc":
                    score_of_assignment, size, node_count, temp_root = csp.check_mc_of(v, value)
                elif "ratio" in obj_type :
                    score_of_assignment, size, node_count, temp_root = csp.check_mc_bdd_ratio_of(v, value)
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
    csp.root_node.deref()
    return best_variable,best_value, best_cost, best_size, best_node_count, best_root

def dynamic_greedy_pWSB(csp, max_p, obj_type,logger):
    pa = csp.partial_assignment
    p = len(pa.assigned)
    while p < max_p:
        #select the assignment that maximizes the score
        p += 1
        # print("-------", p)
        best_variable,best_value, best_cost, best_size, best_node_count, best_root = get_best_assignment(csp,obj_type)

        elapsed = logger.get_time_elapsed()
        mc = best_root.manager.global_model_count(best_root)
        logger.log([p, best_variable, best_value, csp.n, len(csp.cls), mc, best_size,best_node_count, elapsed])

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
                # print(v, value)
                # if v == 15:
                #     print("stop") #looks like sdd eliminated a var when x15=1
                if obj_type == "mc":
                    score_of_assignment, size, node_count, temp_root  = csp.check_mc_of(v, value)
                    # print("score_of_assignment, size, node_count, temp_root ", score_of_assignment, size, node_count )
                elif "ratio" in obj_type:
                    score_of_assignment, size, node_count, temp_root = csp.check_mc_bdd_ratio_of(v, value)
                else:
                    print("Something went wrong")
                    exit(6)

                #best_variable, best_value, best_cost, best_bdd, stats
                #minimize
                # temp_root.ref()
                # csp.sdd_manager.minimize()
                # size = temp_root.size()
                # node_count = temp_root.count()
                assign_queue.put(tuple([-1*score_of_assignment,v, value, size, node_count]))
    # csp.root_node.deref() #not sure if this is needed
    csp.root_node.deref()
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

            score_of_assignment, size, node_count, temp_root = csp.check_mc_of(variable, value)
            # if score_of_assignment == 0:
            #     continue
            elapsed = logger.get_time_elapsed()
            p += 1
            seen_vars.add(variable)
            mc = temp_root.manager.global_model_count(temp_root)
            logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed])
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


        value=0
        score_of_assignment_0, size0, node_count0, temp_root0 = csp.check_mc_of(variable, 0)
        score_of_assignment_1, size1, node_count1, temp_root1 = csp.check_mc_of(variable, 1)
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
            value = 1
        p += 1
        elapsed = logger.get_time_elapsed()
        mc = temp_root.manager.global_model_count(temp_root)
        logger.log([p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed])

        csp.extend_assignment(variable, value, score_of_assignment, temp_root)

def static_greedy_pWSB(csp, obj_type,logger):
    assign_queue = order_var_assignments(csp, obj_type)
    p = 0
    seen_vars = set()
    while not assign_queue.empty():

        item = assign_queue.get() # tuple of ( score_of_assignment,var, value, size, node_count )
        score_of_assignment = abs(item[0])
        variable = item[1]
        value = item[2]
        sdd_size = item[3]
        node_count = item[4]
        if variable not in seen_vars:
            p += 1
            seen_vars.add(variable)
            #need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])
            if obj_type == "mc":
                score_of_assignment, size, node_count, temp_root= csp.check_mc_of(variable, value)
            elif obj_type == "static_ratio":
                score_of_assignment, size, node_count, temp_root = csp.check_mc_bdd_ratio_of(variable, value)
            else:
                print("ERROR")
                exit(666)

            elapsed = logger.get_time_elapsed()
            #is size sdd size or root size?
            mc = temp_root.manager.global_model_count(temp_root)
            logger.log( [p, variable, value, csp.n, len(csp.cls), mc, size, node_count, elapsed])
            csp.extend_assignment(variable, value, score_of_assignment, temp_root)

def print_mc_per_vars(csp, logger):
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

def local_search_pWSB(csp, maxp, seed, logger):
    for p_nb in range(1,maxp+1):
        local_search_pWSB_for_p_iterative(csp, p_nb, seed+p_nb,logger)

def local_search_pWSB_for_p(csp,p, seed):
    """
    Select p random variables
    Find best assignment to these – the one that maximizes model count
    Chose a random variable from this /or the worse coverage in the initial assignment problem
    Eliminate this and chose another random/top of static priority queue
    :param cnf:
    :return:
    """
    #get random p variables
    variables = list(csp.variables.keys())
    random.seed(seed)
    assigned_vars = random.sample(variables, p)
    # init_solution_vars = random.choices(variables, k=p)

    #get best assingment or random?
    random.seed(seed)
    init_assignment = [1 if random.random() > 0.5 else 0 for i in range(len(assigned_vars))]

    #chose an assignment and eliminate it and choose another var - get best assignment for that
    current_assignment = {v:val for v, val in zip(assigned_vars,init_assignment)}
    current_mc, bdd, temp_root = csp.check_score_of_assignments(current_assignment)
    current_dag_size = temp_root.dag_size
    current_root = temp_root
    nb_no_update = 0
    # print("init", current_assignment, "MC:", current_mc, "BDD:", cnf.root_node.dag_size, current_root.dag_size, current_dag_size)
    i =0
    chosen_index=0
    assignment_choice = [i for i in range(len(current_assignment))]
    random.seed(seed)
    while len(assignment_choice) > 0:
        i += 1
        # print("===========================",chosen_index,"===========================")
        # random.seed(seed+i)
        # temp_v = list(current_assignment.keys())
        # chosen_var = random.choice(temp_v)
        # chosen_value = current_assignment[chosen_var]


        chosen_index = random.choice(assignment_choice)
        chosen_var = assigned_vars[chosen_index]
        # print(current_assignment, assigned_vars, chosen_index, chosen_var, current_mc, current_dag_size)
        chosen_value = current_assignment[chosen_var]


        #remove chosen var from solution and pick next best accordingly
        current_assignment.pop(chosen_var)
        assigned_vars.pop(chosen_index)
        # print(assignment)
        csp.partial_assignment.assigned = current_assignment.copy()
        csp.partial_assignment.score = -1
        best_variable,best_value, new_mc, best_bdd, stats, best_root = csp.get_best_wrt_assignment(current_assignment)
        stats = bdd.statistics()
        stats['dag_size'] = best_root.dag_size
        # logger.log([chosen_var, best_variable, best_value, best_mc, len(bdd), stats['n_vars'], stats['n_nodes'],
        #             stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed()])

        # print("current", chosen_var, "MC:", current_mc, "BDD:", current_root.dag_size)
        # print("best:", best_variable, "MC:", best_mc, "BDD:", best_root.dag_size)

        #if current assignment can be improved by replacing chosen var with best_variable - change and restart can_be_changed
        if  new_mc > current_mc  :
            assigned_vars.insert(chosen_index,best_variable)
            assignment_choice.pop(chosen_index)
            current_assignment = csp.partial_assignment.assigned.copy()
            current_assignment[best_variable] = best_value
            current_mc = new_mc
            current_dag_size = stats['dag_size']
            current_root = best_root
            nb_no_update = 0
        else:
            assigned_vars.insert(chosen_index, chosen_var)
            current_assignment[chosen_var] = chosen_value
            nb_no_update += 1
        # print(current_assignment, assigned_vars, chosen_var)
        chosen_index += 1

    # print("FINAL",current_assignment,current_mc)
    logger.log([p, list(current_assignment.keys()), list(current_assignment.values()), current_mc, "-1", "-1", "-1", "-1", current_root.dag_size, logger.get_time_elapsed()])
    # logger.log([best_assignment])
def local_search_pWSB_for_p_iterative(csp,p, seed,logger):
    """
    Select p random variables
    Find best assignment to these – the one that maximizes model count
    Chose a random variable from this /or the worse coverage in the initial assignment problem
    Eliminate this and chose another random/top of static priority queue
    :param cnf:
    :return:
    """
    #get random p variables
    variables = list(csp.variables.keys())
    random.seed(seed)
    assigned_vars = random.sample(variables, p)
    # init_solution_vars = random.choices(variables, k=p)

    #get best assingment or random?
    random.seed(seed)
    # init_assignment = [1 if random.random() > 0.5 else 0 for i in range(len(assigned_vars))]
    init_assignment = []
    for v in assigned_vars:
        mc_value0 = csp.check_mc_of(v, 0)
        mc_value1 = csp.check_mc_of(v, 1)
        if mc_value1 > mc_value0:
            init_assignment.append(1)
        else:
            init_assignment.append(0)

    #chose an assignment and eliminate it and choose another var - get best assignment for that
    current_assignment = {v:val for v, val in zip(assigned_vars,init_assignment)}
    current_mc, bdd, temp_root = csp.check_score_of_assignments(current_assignment)
    current_dag_size = temp_root.dag_size
    current_root = temp_root
    nb_no_update = 0
    # print("init", current_assignment, "MC:", current_mc, "BDD:", cnf.root_node.dag_size, current_root.dag_size, current_dag_size)
    i =0
    chosen_index=0
    while nb_no_update <= p:
        i += 1
        # print("===========================",chosen_index,"===========================")
        # random.seed(seed+i)
        # temp_v = list(current_assignment.keys())
        # chosen_var = random.choice(temp_v)
        # chosen_value = current_assignment[chosen_var]

        if chosen_index >= p:
            chosen_index = 0
        chosen_var = assigned_vars[chosen_index]
        # print(current_assignment, assigned_vars, chosen_index, chosen_var, current_mc, current_dag_size)
        chosen_value = current_assignment[chosen_var]


        #remove chosen var from solution and pick next best accordingly
        current_assignment.pop(chosen_var)
        assigned_vars.pop(chosen_index)
        # print(assignment)
        csp.partial_assignment.assigned = current_assignment.copy()
        csp.partial_assignment.score = -1
        best_variable,best_value, new_mc, best_bdd, stats, best_root = csp.get_best_wrt_assignment(current_assignment)
        stats = bdd.statistics()
        stats['dag_size'] = best_root.dag_size
        # logger.log([chosen_var, best_variable, best_value, best_mc, len(bdd), stats['n_vars'], stats['n_nodes'],
        #             stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed()])

        # print("current", chosen_var, "MC:", current_mc, "BDD:", current_root.dag_size)
        # print("best:", best_variable, "MC:", best_mc, "BDD:", best_root.dag_size)

        #if current assignment can be improved by replacing chosen var with best_variable - change and restart can_be_changed
        if new_mc > current_mc :
            assigned_vars.insert(chosen_index,best_variable)
            current_assignment = csp.partial_assignment.assigned.copy()
            current_assignment[best_variable] = best_value
            current_mc = new_mc
            current_dag_size = stats['dag_size']
            current_root = best_root
            nb_no_update = 0
        else:

            nb_no_update += 1
            if current_mc == 0:
                assigned_vars.insert(chosen_index, best_variable)
                current_assignment = csp.partial_assignment.assigned.copy()
                current_assignment[best_variable] = best_value
                current_mc = new_mc
                current_dag_size = stats['dag_size']
                current_root = best_root
            else:
                assigned_vars.insert(chosen_index, chosen_var)
                current_assignment[chosen_var] = chosen_value
        # print(current_assignment, assigned_vars, chosen_var)
        chosen_index += 1

    # print("FINAL",current_assignment,current_mc)
    sorted_assign = sorted(list(current_assignment.keys()))
    sorted_assign_values = [current_assignment[i] for i in sorted_assign]
    logger.log([p, sorted_assign,sorted_assign_values, current_mc, "-1", "-1", "-1", "-1", current_root.dag_size, logger.get_time_elapsed()])
    # logger.log([p, list(current_assignment.keys()), list(current_assignment.values()), current_mc, "-1", "-1", "-1", "-1", current_root.dag_size, logger.get_time_elapsed()])
    # logger.log([best_assignment])

def run_sdd(alg_type, d, filename, seed, out_folder):
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    columns = ["p", "var", "value", "nb_vars", "nb_cls", "MC", "SDD size", 'node_count', 'time']
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
    cnf = _cnfSDD.CNF(logger)
    b = cnf.load_file(filename)
    # b = cnf.load_file_with_apply(f)
    print(logger.get_time_elapsed())
    if not b:
        return
    #TODO: iterate here between the alf types to avoid spending time for loding initial compilation -
    # needt to save first line in log to add it to all consequtive stat files
    maxp = len(cnf.literals)
    if alg_type == "dynamic":
        dynamic_greedy_pWSB(cnf, maxp, "mc", logger)
    elif alg_type == "dynamic_ratio":
        dynamic_greedy_pWSB(cnf, maxp, "dynamic_ratio",logger)
    elif alg_type == "static":
        static_greedy_pWSB(cnf, "mc",logger)
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
    # type = "static"
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
    # d = sys.argv[1]
    # folder = d.split("/")[-2]
    # filename = sys.argv[2]
    # alg_type = sys.argv[3]
    # # run(alg_type, d, filename,  seed)
    # out_folder = "./results/data_sdd/" + folder + "/"
    # print(alg_type, d, filename, out_folder)
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    #
    # run_sdd(alg_type, d, filename, seed, out_folder)
    # --------------------------------------- end run.sh
    # ------ RUN WMC2020
    input = "./input/"
    folder = "wmc2020_track2_all/"
    out_folder = "./results/wmc2020_sdd/" + folder

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    d = input + folder

    files = [f for f in os.listdir(d) if f.endswith(".cnf")]
    files.sort()
    f_count = 0
    for filename in files:
        # if f_count > 0:
        #     break
        f_count += 1
        f = os.path.join(d, filename)
        print(filename)
        for type in ["dynamic", "static", "random", "random_selection"]:
            run_sdd(type, d, f, seed, out_folder)

    # ------ RUN wmc2020


    # seed = 1234

    # input = "../selective_backbone/"
    # folder = "DatasetA/"
    # out_folder = "./results/data_sdd/" + folder

    #
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # d = input + folder
    #
    # files = [f for f in os.listdir(d) if re.match('.*cnf', f)]
    # files.sort()
    # f_count = 0
    # for filename in files:
    #     f_count += 1
    #     f = os.path.join(d, filename)
    #     print(filename)
    #     # if "instance_K3_N15_M45_01.cnf" in filename:
    #     run_sdd(type, d, f, seed, out_folder) #-- looks like its not better with the copy - try to copy vtree as well and minimize?



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


