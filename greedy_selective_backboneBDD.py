import CNFmodelBDD as _cnf
import os
import evaluate
import queue as _queue
import random
import time
import sys
import re

def get_best_assignment(csp, obj_type):
    "Assumption is that here we already have the bdd extended with the partial assignment"
    best_variable = 0
    best_value = 0
    best_cost = 0
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                if obj_type == "mc":
                    score_of_assignment, bdd, root = csp.check_mc_of(v, value)
                    score_of_assignment = int(score_of_assignment)
                elif obj_type == "dynamic_ratio":
                    score_of_assignment, bdd, root = csp.check_mc_bdd_ratio_of(v, value)
                else:
                    print("ERROR")
                    exit(6)
                if score_of_assignment >= best_cost:
                    best_variable=v
                    best_value=value
                    best_cost=score_of_assignment
                    best_bdd = bdd
                    best_root = root
                    # print("best", best_variable , best_value)
    # print(best_cost,best_bdd.statistics())
    # print("chose", best_variable, best_value, best_cost)
    stats = best_bdd.statistics()
    stats['dag_size'] = best_root.dag_size
    # if int(stats['n_nodes']) <= 40:
    #     filename = csp.instance_name.replace(".cnf", "_" + best_variable + ".png")
    #     best_bdd.dump(filename, [best_root])
    # print("------------------------", best_variable, best_value)
    best_variable=int(best_variable.replace("x", ""))
    return best_variable,best_value, best_cost, best_bdd, stats, best_root

def dynamic_greedy_pWSB(csp, max_p, obj_type,logger):
    pa = csp.partial_assignment
    p = len(pa.assigned)
    while p < max_p:
        #select the assignment that maximizes the score
        p += 1

        best_variable, best_value, best_cost, best_bdd, stats, best_root = get_best_assignment(csp,obj_type) #TODO need to reference root otherwise I get weird numbers, maybe bdd gets dereferencesd
        elapsed = logger.get_time_elapsed()
        logger.log([p, best_variable, best_value, stats['n_vars'], int(best_root.count(csp.n)), len(best_bdd),  stats['n_nodes'], stats['n_reorderings'], stats['dag_size'],elapsed])

        csp.extend_assignment(best_variable,best_value,best_cost, best_root)
        # print( p, best_variable, best_value, best_cost)


def order_var_assignments(csp, obj_type):
    assign_queue = _queue.PriorityQueue()
    for v in csp.variables.keys():
        if v not in csp.partial_assignment.assigned:
            for value in csp.variables[v]:
                # print(v, value)
                if obj_type == "mc":
                    score_of_assignment, bdd, root = csp.check_mc_of(v, value)
                    # print("score_of_assignment ", score_of_assignment, )

                elif "ratio" in obj_type:
                    score_of_assignment, bdd, root = csp.check_mc_bdd_ratio_of(v, value)
                else:
                    print("Something went wrong")
                    exit(6)
                stats = bdd.statistics()
                stats['dag_size'] = root.dag_size
                #best_variable, best_value, best_cost, best_bdd, stats
                assign_queue.put(tuple([-1* score_of_assignment,v, value,bdd, stats]))
    return assign_queue

def random_var_assignments(csp,seed,logger):
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
    assign_queue = random_var_assignments(csp,seed, logger)
    p = 0
    seen_vars = set()
    for item in assign_queue:
        variable = item[0]
        value = item[1]
        if variable not in seen_vars:

            # need to recheck score, mc and bdd size with the partial assingment and the current extension - the initial calculations were only useful for ordering
            # logger.log([p, variable, value, score_of_assignment, len(bdd), stats['n_vars'], stats['n_nodes'],
            #             stats['n_reorderings'], stats['dag_size']])

            score_of_assignment, bdd, root = csp.check_mc_of(variable, value)
            # if score_of_assignment == 0:
            #     continue
            p += 1
            seen_vars.add(variable)
            stats = bdd.statistics()
            stats['dag_size'] = root.dag_size
            logger.log([p, variable, value, stats['n_vars'],score_of_assignment, len(bdd), stats['n_nodes'],
                        stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed() ])

            csp.extend_assignment(variable, value, score_of_assignment, root)

def random_selection_pWSB(csp, seed,logger, obj):
    """
    Select variables randomly and assign value that maximizes model count or the mc/bdd ratio
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
        if obj == "mc":
            score_of_assignment_0, bdd0, root0 = csp.check_mc_of(variable, 0)
            score_of_assignment_1, bdd1, root1 = csp.check_mc_of(variable, 1)
        elif "ratio" in obj:
            score_of_assignment_0, bdd0, root0 = csp.check_mc_bdd_ratio_of(variable, 0)
            score_of_assignment_1, bdd1, root1 = csp.check_mc_bdd_ratio_of(variable, 1)
        else:
            print("Something is off")
            exit(6)
        if score_of_assignment_0 > score_of_assignment_1:
            score_of_assignment = score_of_assignment_0
            bdd = bdd0
            root = root0
        else:
            score_of_assignment = score_of_assignment_1
            bdd = bdd1
            root = root1
            value=1
        p += 1
        stats = bdd.statistics()
        stats['dag_size'] = root.dag_size
        logger.log([p, variable, value, stats['n_vars'],int(root.count(csp.n)), len(bdd), stats['n_nodes'],
                        stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed()])

        csp.extend_assignment(variable, value, score_of_assignment, root)

def static_greedy_pWSB(csp, obj_type,logger):
    assign_queue = order_var_assignments(csp, obj_type)
    p = 0
    seen_vars = set()
    while not assign_queue.empty():

        item = assign_queue.get()
        score_of_assignment = abs(item[0])
        variable = item[1]
        value = item[2]
        bdd = item[3]
        stats = item[4]
        if variable not in seen_vars:
            p += 1
            seen_vars.add(variable)
            if obj_type == "mc":
                score_of_assignment, bdd, best_root = csp.check_mc_of(variable, value)
                score_of_assignment = int(score_of_assignment)
            elif obj_type == "static_ratio":
                score_of_assignment, bdd, best_root = csp.check_mc_bdd_ratio_of(variable, value)
            else:
                print("ERROR")
                exit(666)
            stats = bdd.statistics()
            stats['dag_size'] = best_root.dag_size
            logger.log([p, variable, value, stats['n_vars'],int(best_root.count(csp.n)), len(bdd),  stats['n_nodes'], stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed() ])

            csp.extend_assignment(variable, value, score_of_assignment, best_root)

def print_mc_per_vars(csp, logger):
    assign_queue = order_var_assignments(csp)
    p= 0
    while not assign_queue.empty():

        item = assign_queue.get()
        score_of_assignment = abs(item[0])
        variable = item[1]
        value = item[2]
        bdd = item[3]
        stats = item[4]
        p += 1
        logger.log([p, variable, value, stats['n_vars'], score_of_assignment, len(bdd), stats['n_nodes'], stats['n_reorderings'], stats['dag_size'], logger.get_time_elapsed()])

def run(alg_type, d, filename, seed, out_folder):
    #assume init was run already and it saved the dddmp file, if no file do not even run expr
    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    columns = ["p", "var", "value", 'n_vars', "MC", "BDD len",  'n_nodes', 'n_reorderings', 'dag_size', 'time']
    if "random" in alg_type or "ls" in alg_type:
        # stats_file = d + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + "_" + str(seed) + ".csv"
    else:
        # stats_file = d + "dataset_stats_" + alg_type + ".csv"
        stats_file = out_folder + "dataset_stats_" + alg_type + ".csv"

    # if alg_type != "init":
    #     bdd_file = filename.replace(".cnf", ".dddmp")
    #     if not os.path.exists(bdd_file):
    #         print("INITIAL COMPILATION WILL FAIL")
    #         return
    # columns = ["p", "var", "value", "MC", "SDD size", 'node_count', 'node_size',  'time']

    expr_data = evaluate.ExprData(columns)
    logger = evaluate.Logger(stats_file, columns, expr_data, out_folder, compile=True)

    # files = [f for f in os.listdir(d) if re.match('.*cnf', f)]
    # convert = lambda text: float(text) if text.isdigit() else text
    # alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    # print(files)
    # print([ f for f in files])
    # files.sort()
    all_start = time.perf_counter()
    # f_count = 0
    # for filename in files:
    #     f_count += 1
    f =  filename
    # f = os.path.join(d, filename)
    print(f)
    # exit(1)

    logger.log_expr(f)
    start = time.perf_counter()
    logger.set_start_time(start)
    cnf = _cnf.CNF(logger)
    # b = cnf.load_file(f)
    b = cnf.load_file_with_apply(f)
    print(logger.get_time_elapsed())
    if not b: # file too large not looking at it now
        return
    print("running expr", alg_type, filename)
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
        random_selection_pWSB(cnf, seed,logger,"mc")
    elif "random_ratio_selection" in alg_type:
        random_selection_pWSB(cnf, seed,logger,"ratio")
    elif alg_type == "init":
        print_mc_per_vars(cnf,logger)
    else:
        print("Something wrong")

        # break
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
    # d = "./iscas/iscas93/"
    # d = "./DatasetA/"
    # d = "./DatasetB/"


    # exprs = ["./paper_data/BayesianNetwork/", "./paper_data/Planning/uts/",  "./paper_data/iscas/iscas89/" ,"./paper_data/DatasetA/","./paper_data/DatasetB/", "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/", "./paper_data/Planning/bomb/"]
    # exprs = ["./paper_data/iscas/iscas89/"]
    # for e in exprs:
    #     evaluate_folder(e, reorder=True)
    # exit(10)

    # seed = 1234
    # d = sys.argv[1]
    # filename = sys.argv[2]
    # alg_type = sys.argv[3]
    # print(alg_type, d, filename)
    # run(alg_type, d, filename,  seed)


    # d = "./iscas/iscas85/"
    # d = "./iscas/iscas93/"
    # alg_type = "static"

    # input = "../selective_backbone/"
    # folder = "DatasetA/"
    # out_folder = "./results/data/" + folder
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
    #     # if "instance_K3_N15_M45_01" in filename:
    #     run(type, d,f, seed, out_folder)

    seed = 1234
    d = sys.argv[1]
    folder = d.split("/")[-2]
    filename = sys.argv[2]
    alg_type = sys.argv[3]
    # run(alg_type, d, filename,  seed)
    out_folder = "./results/data/" + folder + "/"
    print(alg_type, d, filename, out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    run(alg_type, d, filename, seed, out_folder)


    # all_type = ["init", "random", "random_selection", "static", "static_ratio", "dynamic", "dynamic_ratio"]
    # for alg_type in all_type:
    #     run(alg_type, d, seed)

    # files = [f for f in os.listdir(d) if re.match('.*cnf', f)]
    # files.sort()
    # f_count = 0
    # for filename in files:
    #     f_count += 1
    #     f = os.path.join(d, filename)
    #     print(filename)
    #     run(alg_type, d,f, seed)

    # run(type, d, seed, stats_file, columns)
    # exit(10)

    # for d in ["./DatasetA/","./DatasetB/", "./iscas/", "./BayesianNetwork/", "./Planning/"]:
    #     all_type = ["init",  "random","random_selection", "static","static_ratio", "dynamic", "dynamic_ratio"]
    #     for alg_type in all_type:
    #         seed = 1234
    #         print("============================",type,"============================")
    #         if "random" in type or "ls" in type:
    #             stats_file = d + "dataset_stats_" + type + "_" + str(seed) + ".csv"
    #         else:
    #             stats_file = d + "dataset_stats_" + type + ".csv"
    #         run(alg_type, d, seed, stats_file, columns)
    #
    # exit(10)

    # columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    # #random
    # type = "random_1234"
    # stats_file = d + "1dataset_stats_" + type + ".csv"
    # expr_data_rand = evaluate.ExprData(columns)
    # expr_data_rand.read_stats_file(stats_file)
    #
    # # static
    # type = "static"
    # stats_file = d + "1dataset_stats_" + type + ".csv"
    # expr_data_static= evaluate.ExprData(columns)
    # expr_data_static.read_stats_file(stats_file)
    #
    #
    # # dynamic
    # type = "dynamic"
    # stats_file = d + "1dataset_stats_" + type + ".csv"
    # expr_data_dynamic = evaluate.ExprData(columns)
    # expr_data_dynamic.read_stats_file(stats_file)
    #
    # # dynamic
    # type = "dynamic_ratio"
    # stats_file = d + "1dataset_stats_" + type + ".csv"
    # expr_data_dynamic2 = evaluate.ExprData(columns)
    # expr_data_dynamic2.read_stats_file(stats_file)

    # static_ratio
    # type = "static_ratio"
    # stats_file = d + "dataset_stats_" + type + ".csv"
    # expr_data_static2 = evaluate.ExprData(columns)
    # expr_data_static2.read_stats_file(stats_file)

    # random selection
    # type = "random_selection_1234"
    # stats_file = d + "1dataset_stats_" + type + ".csv"
    # expr_data_random2 = evaluate.ExprData(columns)
    # expr_data_random2.read_stats_file(stats_file)

    # Loca search
    # type = "ls"
    # stats_file = d + "1dataset_stats_" + type +"_"+str(seed)+ ".csv"
    # expr_data_ls = evaluate.ExprData(columns)
    # expr_data_ls.read_stats_file(stats_file)
    #
    # plot_type = "raw"
    # evaluate.plot_multiple([expr_data_rand, expr_data_static, expr_data_dynamic,expr_data_dynamic2, expr_data_random2], "efficiency", ["random", "static", "dynamic", "dynamic_ratio", "random_selection_1234" ], "init")
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


