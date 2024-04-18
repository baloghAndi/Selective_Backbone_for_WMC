import utils
import bisect
from CNFmodelD4 import WCNF
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def util_create_cnf_and_weight_file(wcnf_file):
    wcnf = WCNF()
    wcnf.load_wcnf_file(wcnf_file)
    print(wcnf_file)
    cnff = wcnf_file.replace("wcnf", "cnf")
    weightfile = wcnf_file.replace("wcnf", "w")
    # wcnf.write_cnf(cnff)
    # wcnf.write_weights(weightfile)
    log_weightfile = wcnf_file.replace(".cnf", "_w10.w")
    print(log_weightfile)
    wcnf.write_scaled_weights(log_weightfile)

def create_scale2_weights_file(wcnf_file):
    wcnf = WCNF()
    wcnf.load_wcnf_file(wcnf_file)
    log_weightfile = wcnf_file.replace(".wcnf", "_w2.w")
    print(log_weightfile)
    wcnf.write_scaled_weights(log_weightfile)

def write_minic2d_file(wcnf_file, input_folder):
    wcnf = WCNF(scalar=3)
    b = wcnf.load_file(wcnf_file)
    if not b:
        return
    filename = wcnf_file.replace(".wcnf", "_w10.wcnf")
    filename = filename.replace(input_folder, input_folder+"minic2d/")
    print(filename)
    with open(filename, "w+") as f:
        f.write("p cnf " + str(wcnf.n) + " " + str(len(wcnf.cls)) + "\n")
        for c in wcnf.cls:
            f.write(" ".join([str(x) for x in c]) + " 0 \n")
        weights = []
        for x,y in zip(wcnf.literal_weights[1],wcnf.literal_weights[0]):
            weights.append(x)
            weights.append(y)
        f.write("c weights "+" ".join([str(x) for x in weights]))
        f.close()

def eliminate_trivial_backbone(cnf_file):
    "eliminate backbones that are defined by a  clause with a single variable"
    cnf = WCNF(scalar=3)
    loaded = (cnf.load_file(cnf_file))
    if not loaded:
        return False
    backbones = []
    for c in cnf.cls:
        if len(c) == 1: #backbone
            if c[0] not in backbones:
                backbones.append(c[0])
    backbones.sort()
    literal_mapping = create_literal_mapping(cnf.literals, [abs(x) for x in backbones])
    csp_clauses = []
    for clause in cnf.cls:
        if len(clause) == 1:
            continue
        updated_clause = []
        for lit in clause:
            # print(lit, clause)
            if lit in backbones:
                print("backbone")
                updated_clause = []
                break #no need for this clause
            elif -1*lit in backbones: #if opposite lit in backbone just ignore the literal
                print("opp backbone")
                continue
            if lit < 0:
                updated_clause.append(-literal_mapping[abs(lit)])
            else:
                updated_clause.append(literal_mapping[abs(lit)])
            # print(updated_clause)
        if len(updated_clause) > 0:
            csp_clauses.append(updated_clause)

    cnf.n = cnf.n - len(backbones)
    print("ELIMInATED BACKNONES: ", len(backbones) )
    # cnf_file_name = cnf.instance_name.replace("input", "input/preproc")
    cnf_file_name = cnf.instance_name.replace("Benchmark_original", "Benchmark")
    cnf.print_clauses(cnf_file_name, csp_clauses, cnf.n)
    cnf.cls = csp_clauses

    cnf.literals = [i for i in range(1,cnf.n+1)]

    # weight_file = cnf.weight_file.replace("input","input/preproc")
    weight_file = cnf.weight_file.replace("Benchmark_original", "Benchmark")
    # weight_file = weight_file.replace(".w","_w2.w")
    weights = {0: cnf.n * [1], 1: cnf.n * [1]}
    for lit in cnf.literals:
        old_lit = list(literal_mapping.keys())[list(literal_mapping.values()).index(lit)]
        weights[0][lit-1] = cnf.literal_weights[0][old_lit-1]
        weights[1][lit-1] = cnf.literal_weights[1][old_lit-1]
    cnf.literal_weights = weights
    # cnf.write_scaled_weights(weight_file)
    cnf.write_weights(weight_file)
    return cnf_file_name


def create_literal_mapping(literals, backbones):
    #create mapping of remaining variables and current ones given the backbones we take out
    literal_mapping = {i:i for i in literals}
    prev_valid_lits = []
    for l in literal_mapping.keys():
        if l in backbones:
            literal_mapping[l] = 0
            bisect.insort(prev_valid_lits, l)
            # prev_valid_lits.append(l)
        else:
            if len(prev_valid_lits) != 0:
                pl = prev_valid_lits.pop(0)
                bisect.insort(prev_valid_lits, l)
                literal_mapping[l] = pl
    # print(literal_mapping)
    return literal_mapping


def generate_random_uniform_weights(cnf_filename):
    np.random.seed(seed=123)
    scalar = 3
    csp = WCNF(scalar=scalar)
    csp.load_file(cnf_filename)
    # max_w = 1000000
    # ws = np.random.uniform(0, max_w, csp.n)
    max_w = 1
    ws = np.random.uniform(0, max_w, csp.n)
    positive_weights = [x if x>0 else max_w for x in ws] #in case 0 is generated max should be added to make var indifferent
    print(max_w in positive_weights)
    negative_weights = [max_w - x if x<max_w else max_w for x in positive_weights]
    # all = positive_weights + negative_weights
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,3))
    # min_max_scaler.fit_transform(np.reshape(all, (-1, 1)))
    # positive_weights_scaled =  min_max_scaler.transform(np.reshape(positive_weights, (-1, 1)))
    # negative_weights_scaled =  min_max_scaler.transform(np.reshape(negative_weights, (-1, 1)))
    # print(negative_weights_scaled)

    csp.literal_weights =  {0:[ scalar* x for x in negative_weights], 1:[ scalar* y  for y in positive_weights]}
    # print(csp.literal_weights)
    csp.write_weights(csp.weight_file, force=True)


    # plt.plot([i for i in range(len(positive_weights)) ], positive_weights)
    # plt.show()

def count_irrelevant_literals():
    files = ["./input/Benchmark/2bitcomp_5.cnf",
              "./input/Benchmark/2bitmax_6.cnf",
              "./input/Benchmark/4step.cnf",
              "./input/Benchmark/5step.cnf",
              "./input/Benchmark/ais8.cnf",
              "./input/Benchmark/binsearch.16.pp.cnf",
              "./input/Benchmark/blasted_case100.cnf",
              "./input/Benchmark/blasted_case102.cnf",
              "./input/Benchmark/blasted_case105.cnf",
              "./input/Benchmark/blasted_case108.cnf",
              "./input/Benchmark/blasted_case120.cnf",
              "./input/Benchmark/blasted_case123.cnf",
              './input/Benchmark/blasted_case200.cnf',
              './input/Benchmark/blasted_case202.cnf',
              './input/Benchmark/blasted_case36.cnf',
              './input/Benchmark/blasted_case54.cnf',
              './input/Benchmark/blasted_case_3_b14_1.cnf',
              './input/Benchmark/fs-01.net.cnf',
              './input/Benchmark/nocountdump14.cnf',
              './input/Benchmark/or-100-10-10-UC-20.cnf',
              './input/Benchmark/or-100-10-1-UC-50.cnf',
              './input/Benchmark/or-100-10-2-UC-60.cnf',
              './input/Benchmark/or-50-5-1-UC-40.cnf',
              './input/Benchmark/or-60-20-3-UC-20.cnf',
              './input/Benchmark/or-60-5-7-UC-20.cnf']
    for filename in files:
        cnf = WCNF()
        print("----------------------", filename)
        cnf.load_file(filename)
        cnf.count_irrelevant_lits()

def remove_unconstrained_vars():
    input = "./input/"
    folder = "Benchmark_original/"
    d = input + folder
    files = [f for f in os.listdir(d) if re.match('.*\.cnf', f) and "temp" not in f]
    # files = ["./input/Benchmark_original/2bitcomp_5.cnf",
    #          "./input/Benchmark_original/2bitmax_6.cnf",
    #          "./input/Benchmark_original/4step.cnf",
    #          "./input/Benchmark_original/5step.cnf",
    #          "./input/Benchmark_original/ais8.cnf",
    #          "./input/Benchmark_original/binsearch.16.pp.cnf",
    #          "./input/Benchmark_original/blasted_case100.cnf",
    #          "./input/Benchmark_original/blasted_case102.cnf",
    #          "./input/Benchmark_original/blasted_case105.cnf",
    #          "./input/Benchmark_original/blasted_case108.cnf",
    #          "./input/Benchmark_original/blasted_case120.cnf",
    #          "./input/Benchmark_original/blasted_case123.cnf",
    #          './input/Benchmark_original/blasted_case200.cnf',
    #          './input/Benchmark_original/blasted_case202.cnf',
    #          './input/Benchmark_original/blasted_case36.cnf',
    #          './input/Benchmark_original/blasted_case54.cnf',
    #          './input/Benchmark_original/blasted_case_3_b14_1.cnf',
    #          './input/Benchmark_original/fs-01.net.cnf',
    #          './input/Benchmark_original/nocountdump14.cnf',
    #          './input/Benchmark_original/or-100-10-10-UC-20.cnf',
    #          './input/Benchmark_original/or-100-10-1-UC-50.cnf',
    #          './input/Benchmark_original/or-100-10-2-UC-60.cnf',
    #          './input/Benchmark_original/or-50-5-1-UC-40.cnf',
    #          './input/Benchmark_original/or-60-20-3-UC-20.cnf',
    #          './input/Benchmark_original/or-60-5-7-UC-20.cnf']
    for filename in files:
        f = os.path.join(d, filename)
        print("----------------------", filename)
        if "s27_new_15_7.cnf" == filename:

            f = eliminate_trivial_backbone(f)
            if f != False:
                remove_unconstrained_vars_from_cnf(f)


def remove_unconstrained_vars_from_cnf(cnf_file):
    cnf = WCNF(scalar=3)
    cnf.load_file(cnf_file)
    lit_count = {}
    for l in cnf.literals:
        lit_count[l] = 0
        lit_count[-l] = 0
    for c in cnf.cls:
        for l in c:
            lit_count[l] += 1
    to_remove = []
    for l in cnf.literals:
        if lit_count[l] == 0 and lit_count[-l] == 0:
            if abs(l) not in to_remove:
                to_remove.append(abs(l))
    to_remove.sort()
    print(len(to_remove), to_remove)
    literal_mapping = create_literal_mapping(cnf.literals,to_remove)
    csp_clauses = []
    for clause in cnf.cls:
        updated_clause = []
        for lit in clause:
            if lit < 0:
                updated_clause.append(-literal_mapping[abs(lit)])
            else:
                updated_clause.append(literal_mapping[abs(lit)])
            # print(updated_clause)
        if len(updated_clause) > 0:
            csp_clauses.append(updated_clause)

    cnf.n = cnf.n - len(to_remove)
    # instance_name = cnf.instance_name.replace("Benchmark_original", "Benchmark")
    cnf.print_clauses(cnf.instance_name, csp_clauses, cnf.n)
    cnf.cls = csp_clauses

    cnf.literals = [i for i in range(1,cnf.n+1)]

    # weight_file = cnf.weight_file.replace("Benchmark_original","Benchmark")
    # weight_file = weight_file.replace(".w","_w2.w")
    weights = {0: cnf.n * [1], 1: cnf.n * [1]}
    for lit in cnf.literals:
        old_lit = list(literal_mapping.keys())[list(literal_mapping.values()).index(lit)]
        weights[0][lit-1] = cnf.literal_weights[0][old_lit-1]
        weights[1][lit-1] = cnf.literal_weights[1][old_lit-1]
    cnf.literal_weights = weights
    # cnf.write_scaled_weights(weight_file)
    cnf.write_weights(cnf.weight_file)


def preprocess(cnf_file):
    """
    If a clause contains the same literal more than once, we remove all but one occurrences of this literal in the clause.
    If a clause contains both a literal and its negation, we remove the clause.
    If two clauses are composed of the exact same set of literals, regardless of the order in which literals appear in the clause, we remove one of the clauses.
    If there are trivial backbones in the form of unit clauses, we eliminate them and propagate this.
    Eliminate any variable that is not constrained by any clauses
    propagate changes and redo these checks until no more can eliminated"""
    cnf = WCNF(scalar=3)
    loaded = cnf.load_file(cnf_file)
    if not loaded:
        return False
    c1 = True
    c2 = True
    c3 = True
    c4 = True
    c5 = True
    init_nb_vars = cnf.n
    nb_vars = cnf.n
    nb_cls = len(cnf.cls)
    init_nb_cls = len(cnf.cls)
    var_diff = 0
    cls_diff = 0
    while c1 or c2 or c3 or c4 or c5:
        c1, cnf = eliminate_same_lit_in_cls(cnf)
        var_diff += nb_vars - cnf.n
        cls_diff += nb_cls - len(cnf.cls)
        nb_vars = cnf.n
        nb_cls = len(cnf.cls)
        print("=================================", "eliminate_same_lit_in_cls ", var_diff, cls_diff)

        c2, cnf = eliminate_conflict_cls(cnf)
        var_diff += nb_vars - cnf.n
        cls_diff += nb_cls - len(cnf.cls)
        nb_vars = cnf.n
        nb_cls = len(cnf.cls)
        print("=================================", "eliminate_conflict_cls ", var_diff, cls_diff)

        c3, cnf = eliminate_same_cls(cnf)
        var_diff += nb_vars - cnf.n
        cls_diff += nb_cls - len(cnf.cls)
        nb_vars = cnf.n
        nb_cls = len(cnf.cls)
        print("=================================", "eliminate_same_cls ", var_diff, cls_diff)

        c4, cnf = eliminate_backbone(cnf)
        var_diff += nb_vars - cnf.n
        cls_diff += nb_cls - len(cnf.cls)
        nb_vars = cnf.n
        nb_cls = len(cnf.cls)
        print("=================================", "eliminate_backbone ", var_diff, cls_diff)

        c5, cnf = eliminate_uncstr_var(cnf)
        var_diff += nb_vars - cnf.n
        cls_diff += nb_cls - len(cnf.cls)
        nb_vars = cnf.n
        nb_cls = len(cnf.cls)
        print("=================================", "eliminate_uncstr_var ", var_diff, cls_diff)
    new_file = cnf.instance_name.replace("Dataset","Dataset_preproc")
    # new_file = cnf.instance_name.replace("Benchmark_original","Benchmark_preproc")
    cnf.write_cnf(new_file)
    return init_nb_vars-cnf.n, init_nb_cls-len(cnf.cls)


def eliminate_same_lit_in_cls(cnf):
    """If a clause contains the same literal more than once, we remove all but one occurrences of this literal in the clause.
    """
    change = False
    new_cls = []
    for cls in cnf.cls:
        if len(set(cls)) != len(cls):
            new_cls.append(set(cls))
            print("ELIMINATE LIT", cls)
            change = True
        else:
            new_cls.append(cls)
    cnf.cls = new_cls
    return change, cnf



def eliminate_conflict_cls(cnf):
    """If a clause contains both a literal and its negation, we remove the clause."""
    change = False
    new_cls = []
    for cls in cnf.cls:
        add_cls = True
        for l in cls:
            if 0-l in cls:
                change = True
                add_cls = False
                print("ELIMINATE CONFLICT: ", l)
        if add_cls:
            new_cls.append(cls)
    cnf.cls = new_cls
    return change, cnf

def eliminate_same_cls(cnf):
    """If two clauses are composed of the exact same set of literals, regardless of the order in which literals
    appear in the clause, we remove one of the clauses."""
    change = False
    # unique_cls = set(frozenset(cls) for cls in cnf.cls)
    unique_cls = {}
    unique_cls = {tuple(sorted(p)) for p in cnf.cls}

    # print("unique")
    if len(unique_cls) != len(cnf.cls):
        change = True
        print("ELIMINATE cls ", len(cnf.cls)- len(unique_cls) )
        if len(cnf.cls)- len(unique_cls) == 29:
            print("stop")
        # print(len(unique_cls))
        # print(len(cnf.cls))
        new_cls = list(list(c) for c in unique_cls)
        # for c in  new_cls:
        #     if c not in cnf.cls:
        #         print("not in ", c)
        # print(cnf.cls - unique_cls )
        cnf.cls = new_cls.copy()
    return change, cnf

def eliminate_backbone(cnf):
    backbones = []
    abs_backbones = []
    for c in cnf.cls:
        if len(c) == 1:  # backbone
            if abs(c[0]) not in abs_backbones:
                abs_backbones.append(abs(c[0]))
                backbones.append(c[0])
    # backbones = [abs(x) for x in backbones]
    abs_backbones.sort()
    print(len(abs_backbones))
    literal_mapping = create_literal_mapping(cnf.literals, abs_backbones)
    csp_clauses = []
    for clause in cnf.cls:
        if len(clause) == 1:
            continue
        updated_clause = []
        for lit in clause:
            # print(lit, clause)
            if lit in backbones:
                # print("backbone")
                updated_clause = []
                break  # no need for this clause
            elif -1 * lit in backbones:  # if opposite lit in backbone just ignore the literal
                # print("opp backbone")
                continue
            else:
                if lit < 0:
                    updated_clause.append(-literal_mapping[abs(lit)])
                else:
                    updated_clause.append(literal_mapping[abs(lit)])
            # print(updated_clause)
        if len(updated_clause) > 0:
            csp_clauses.append(updated_clause.copy())
    cnf.n = cnf.n - len(backbones)
    # print("ELIMInATED BACKNONES: ", len(backbones))
    cnf.cls = csp_clauses
    cnf.literals = [i for i in range(1, cnf.n + 1)]
    change = False
    if len(backbones) > 0:
        change = True

    weights = {0: cnf.n * [1], 1: cnf.n * [1]}
    for lit in cnf.literals:
        old_lit = list(literal_mapping.keys())[list(literal_mapping.values()).index(lit)]
        weights[0][lit - 1] = cnf.literal_weights[0][old_lit - 1]
        weights[1][lit - 1] = cnf.literal_weights[1][old_lit - 1]
    cnf.literal_weights = weights
    return change, cnf
def eliminate_uncstr_var(cnf):
    change = False
    lit_count = {}
    for l in cnf.literals:
        lit_count[l] = 0
        lit_count[-l] = 0
    for c in cnf.cls:
        for l in c:
            # print(len(cnf.literals), c, l)
            lit_count[l] += 1
    to_remove = []
    for l in cnf.literals:
        if lit_count[l] == 0 and lit_count[-l] == 0:
            if abs(l) not in to_remove:
                to_remove.append(abs(l))
    to_remove.sort()
    # print(len(to_remove), to_remove)
    if len(to_remove) > 0:
        change = True
    print("ELIM var", to_remove)
    literal_mapping = create_literal_mapping(cnf.literals, to_remove)
    csp_clauses = []
    for clause in cnf.cls:
        updated_clause = []
        for lit in clause:
            if lit < 0:
                updated_clause.append(-literal_mapping[abs(lit)])
            else:
                updated_clause.append(literal_mapping[abs(lit)])
            # print(updated_clause)
        if len(updated_clause) > 0:
            csp_clauses.append(updated_clause)

    cnf.n = cnf.n - len(to_remove)
    cnf.cls = csp_clauses
    cnf.literals = [i for i in range(1, cnf.n + 1)]

    weights = {0: cnf.n * [1], 1: cnf.n * [1]}
    for lit in cnf.literals:
        old_lit = list(literal_mapping.keys())[list(literal_mapping.values()).index(lit)]
        weights[0][lit - 1] = cnf.literal_weights[0][old_lit - 1]
        weights[1][lit - 1] = cnf.literal_weights[1][old_lit - 1]
    cnf.literal_weights = weights

    return change, cnf

def preprocess_folder():
    input = "./input/"
    folder = "Dataset/"
    # folder = "Benchmark_original/"
    # folder = "Benchmark_original/"
    d = input + folder
    files = [f for f in os.listdir(d) if re.match('.*\.cnf', f) and "temp" not in f]
    # temp_d = "./input/Benchmark_preproc/"
    # already_processed_files = [f for f in os.listdir(temp_d) if re.match('.*\.cnf', f) and "temp" not in f]
    # temp_d = "./input/Benchmark_preproc2/"
    # temp = [f for f in os.listdir(temp_d) if re.match('.*\.cnf', f) and "temp" not in f]
    # already_processed_files.extend(temp)
    # files = ["./input/Benchmark_original/2bitcomp_5.cnf"]
    #          "./input/Benchmark_original/2bitmax_6.cnf",
    #          "./input/Benchmark_original/4step.cnf",
    #          "./input/Benchmark_original/5step.cnf",
    #          "./input/Benchmark_original/ais8.cnf",
    #          "./input/Benchmark_original/binsearch.16.pp.cnf",
    #          "./input/Benchmark_original/blasted_case100.cnf",
    #          "./input/Benchmark_original/blasted_case102.cnf",
    #          "./input/Benchmark_original/blasted_case105.cnf",
    #          "./input/Benchmark_original/blasted_case108.cnf",
    #          "./input/Benchmark_original/blasted_case120.cnf",
    #          "./input/Benchmark_original/blasted_case123.cnf",
    #          './input/Benchmark_original/blasted_case200.cnf',
    #          './input/Benchmark_original/blasted_case202.cnf',
    #          './input/Benchmark_original/blasted_case36.cnf',
    #          './input/Benchmark_original/blasted_case54.cnf',
    #          './input/Benchmark_original/blasted_case_3_b14_1.cnf',
    #          './input/Benchmark_original/fs-01.net.cnf',
    #          './input/Benchmark_original/nocountdump14.cnf',
    #          './input/Benchmark_original/or-100-10-10-UC-20.cnf',
    #          './input/Benchmark_original/or-100-10-1-UC-50.cnf',
    #          './input/Benchmark_original/or-100-10-2-UC-60.cnf',
    #          './input/Benchmark_original/or-50-5-1-UC-40.cnf',
    #          './input/Benchmark_original/or-60-20-3-UC-20.cnf',
    #          './input/Benchmark_original/or-60-5-7-UC-20.cnf']
    not_compiling = ['./input/Benchmark_preproc/20_rd_r45.cnf', './input/Benchmark_preproc/23_rd_r45.cnf',
                     './input/Benchmark_preproc/2bitadd_11.cnf',
                     './input/Benchmark_preproc/2bitadd_12.cnf', './input/Benchmark_preproc/3bitadd_31.cnf',
                     './input/Benchmark_preproc/3bitadd_32.cnf',
                     './input/Benchmark_preproc/5_100_sd_schur.cnf', './input/Benchmark_preproc/5_140_sd_schur.cnf',
                     './input/Benchmark_preproc/54.sk_12_97.cnf',
                     './input/Benchmark_preproc/ais12.cnf', './input/Benchmark_preproc/binsearch.16.cnf',
                     './input/Benchmark_preproc/binsearch.32.cnf',
                     './input/Benchmark_preproc/blasted_case_0_b12_even1.cnf',
                     './input/Benchmark_preproc/blasted_case_0_b12_even2.cnf',
                     './input/Benchmark_preproc/blasted_case_0_b12_even3.cnf',
                     './input/Benchmark_preproc/blasted_case_0_ptb_1.cnf',
                     './input/Benchmark_preproc/blasted_case_0_ptb_2.cnf',
                     './input/Benchmark_preproc/blasted_case104.cnf',
                     './input/Benchmark_preproc/blasted_case12.cnf', './input/Benchmark_preproc/blasted_case141.cnf',
                     './input/Benchmark_preproc/blasted_case142.cnf',
                     './input/Benchmark_preproc/blasted_case_1_4_b14_even.cnf',
                     './input/Benchmark_preproc/blasted_case_1_b12_even1.cnf',
                     './input/Benchmark_preproc/blasted_case_1_b12_even2.cnf',
                     './input/Benchmark_preproc/blasted_case_1_b12_even3.cnf',
                     './input/Benchmark_preproc/blasted_case1_b14_even3.cnf',
                     './input/Benchmark_preproc/blasted_case_1_b14_even.cnf',
                     './input/Benchmark_preproc/blasted_case_1_ptb_2.cnf',
                     './input/Benchmark_preproc/blasted_case_2_b12_even1.cnf',
                     './input/Benchmark_preproc/blasted_case_2_b12_even2.cnf',
                     './input/Benchmark_preproc/blasted_case_2_b12_even3.cnf',
                     './input/Benchmark_preproc/blasted_case_2_b14_even.cnf',
                     './input/Benchmark_preproc/blasted_case_2_ptb_2.cnf',
                     './input/Benchmark_preproc/blasted_case_3_4_b14_even.cnf',
                     './input/Benchmark_preproc/blasted_case37.cnf',
                     './input/Benchmark_preproc/blasted_case3_b14_even3.cnf',
                     './input/Benchmark_preproc/blasted_case42.cnf', './input/Benchmark_preproc/blasted_squaring10.cnf',
                     './input/Benchmark_preproc/blasted_squaring11.cnf',
                     './input/Benchmark_preproc/blasted_squaring12.cnf',
                     './input/Benchmark_preproc/blasted_squaring14.cnf',
                     './input/Benchmark_preproc/blasted_squaring16.cnf',
                     './input/Benchmark_preproc/blasted_squaring1.cnf',
                     './input/Benchmark_preproc/blasted_squaring2.cnf',
                     './input/Benchmark_preproc/blasted_squaring30.cnf',
                     './input/Benchmark_preproc/blasted_squaring3.cnf',
                     './input/Benchmark_preproc/blasted_squaring4.cnf',
                     './input/Benchmark_preproc/blasted_squaring5.cnf',
                     './input/Benchmark_preproc/blasted_squaring60.cnf',
                     './input/Benchmark_preproc/blasted_squaring6.cnf',
                     './input/Benchmark_preproc/blasted_squaring70.cnf',
                     './input/Benchmark_preproc/blasted_squaring7.cnf',
                     './input/Benchmark_preproc/blasted_squaring8.cnf',
                     './input/Benchmark_preproc/blasted_squaring9.cnf',
                     './input/Benchmark_preproc/blasted_TR_b12_1_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b12_2_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b12_even2_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b12_even3_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b12_even7_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b14_2_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b14_3_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b14_even2_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b14_even3_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_b14_even_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_device_1_even_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_ptb_1_linear.cnf',
                     './input/Benchmark_preproc/blasted_TR_ptb_2_linear.cnf',
                     './input/Benchmark_preproc/blockmap_20_03.net.cnf',
                     './input/Benchmark_preproc/blockmap_22_02.net.cnf', './input/Benchmark_preproc/bmc-galileo-8.cnf',
                     './input/Benchmark_preproc/bmc-galileo-9.cnf',
                     './input/Benchmark_preproc/bmc-ibm-10.cnf', './input/Benchmark_preproc/bmc-ibm-11.cnf',
                     './input/Benchmark_preproc/bmc-ibm-12.cnf',
                     './input/Benchmark_preproc/bmc-ibm-6.cnf', './input/Benchmark_preproc/bmc-ibm-7.cnf',
                     './input/Benchmark_preproc/C129_FR.cnf',
                     './input/Benchmark_preproc/c1355.isc.cnf', './input/Benchmark_preproc/C140_FV.cnf',
                     './input/Benchmark_preproc/C140_FW.cnf',
                     './input/Benchmark_preproc/C168_FW.cnf', './input/Benchmark_preproc/c1908.isc.cnf',
                     './input/Benchmark_preproc/C202_FS.cnf',
                     './input/Benchmark_preproc/C202_FW.cnf', './input/Benchmark_preproc/C203_FCL.cnf',
                     './input/Benchmark_preproc/C203_FS.cnf',
                     './input/Benchmark_preproc/C203_FW.cnf', './input/Benchmark_preproc/C208_FA.cnf',
                     './input/Benchmark_preproc/C208_FC.cnf',
                     './input/Benchmark_preproc/C210_FS.cnf', './input/Benchmark_preproc/C210_FW.cnf',
                     './input/Benchmark_preproc/C220_FV.cnf',
                     './input/Benchmark_preproc/C220_FW.cnf', './input/Benchmark_preproc/c2670.isc.cnf',
                     './input/Benchmark_preproc/c7552.isc.cnf',
                     './input/Benchmark_preproc/c880.isc.cnf', './input/Benchmark_preproc/compress.sk_17_291.cnf',
                     './input/Benchmark_preproc/ConcreteRoleAffectationService.sk_119_273.cnf',
                     './input/Benchmark_preproc/diagStencilClean.sk_41_36.cnf',
                     './input/Benchmark_preproc/diagStencil.sk_35_36.cnf',
                     './input/Benchmark_preproc/enqueueSeqSK.sk_10_42.cnf',
                     './input/Benchmark_preproc/fphp-010-020.cnf', './input/Benchmark_preproc/fphp-015-020.cnf',
                     './input/Benchmark_preproc/fs-07.net.cnf',
                     './input/Benchmark_preproc/fs-10.net.cnf', './input/Benchmark_preproc/fs-13.net.cnf',
                     './input/Benchmark_preproc/fs-16.net.cnf',
                     './input/Benchmark_preproc/fs-19.net.cnf', './input/Benchmark_preproc/fs-22.net.cnf',
                     './input/Benchmark_preproc/fs-25.net.cnf',
                     './input/Benchmark_preproc/fs-28.net.cnf', './input/Benchmark_preproc/fs-29.net.cnf',
                     './input/Benchmark_preproc/isolateRightmost.sk_7_481.cnf',
                     './input/Benchmark_preproc/jburnim_morton.sk_13_530.cnf',
                     './input/Benchmark_preproc/karatsuba.sk_7_41.cnf', './input/Benchmark_preproc/lang12.cnf',
                     './input/Benchmark_preproc/lang15.cnf', './input/Benchmark_preproc/lang16.cnf',
                     './input/Benchmark_preproc/lang19.cnf',
                     './input/Benchmark_preproc/lang20.cnf', './input/Benchmark_preproc/lang23.cnf',
                     './input/Benchmark_preproc/lang24.cnf',
                     './input/Benchmark_preproc/lang27.cnf', './input/Benchmark_preproc/lang28.cnf',
                     './input/Benchmark_preproc/listReverse.sk_11_43.cnf',
                     './input/Benchmark_preproc/log2.sk_72_391.cnf', './input/Benchmark_preproc/log-5.cnf',
                     './input/Benchmark_preproc/logcount.sk_16_86.cnf',
                     './input/Benchmark_preproc/LoginService2.sk_23_36.cnf',
                     './input/Benchmark_preproc/logistics.c.cnf', './input/Benchmark_preproc/logistics.d.cnf',
                     './input/Benchmark_preproc/ls10-normalized.cnf', './input/Benchmark_preproc/ls11-normalized.cnf',
                     './input/Benchmark_preproc/ls12-normalized.cnf',
                     './input/Benchmark_preproc/ls13-normalized.cnf', './input/Benchmark_preproc/ls14-normalized.cnf',
                     './input/Benchmark_preproc/ls15-normalized.cnf',
                     './input/Benchmark_preproc/ls16-normalized.cnf', './input/Benchmark_preproc/ls8-normalized.cnf',
                     './input/Benchmark_preproc/ls9-normalized.cnf',
                     './input/Benchmark_preproc/lss.sk_6_7.cnf',
                     './input/Benchmark_preproc/mastermind_03_08_05.net.cnf',
                     './input/Benchmark_preproc/mastermind_10_08_03.net.cnf',
                     './input/Benchmark_preproc/nocountdump4.cnf', './input/Benchmark_preproc/or-100-10-10.cnf',
                     './input/Benchmark_preproc/or-100-10-10-UC-10.cnf', './input/Benchmark_preproc/or-100-10-1.cnf',
                     './input/Benchmark_preproc/or-100-10-1-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-2.cnf', './input/Benchmark_preproc/or-100-10-2-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-3.cnf',
                     './input/Benchmark_preproc/or-100-10-3-UC-10.cnf', './input/Benchmark_preproc/or-100-10-4.cnf',
                     './input/Benchmark_preproc/or-100-10-4-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-5.cnf', './input/Benchmark_preproc/or-100-10-5-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-6.cnf',
                     './input/Benchmark_preproc/or-100-10-6-UC-10.cnf', './input/Benchmark_preproc/or-100-10-7.cnf',
                     './input/Benchmark_preproc/or-100-10-7-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-8.cnf', './input/Benchmark_preproc/or-100-10-8-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-10-9.cnf',
                     './input/Benchmark_preproc/or-100-10-9-UC-10.cnf', './input/Benchmark_preproc/or-100-20-10.cnf',
                     './input/Benchmark_preproc/or-100-20-10-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-10-UC-20.cnf', './input/Benchmark_preproc/or-100-20-1.cnf',
                     './input/Benchmark_preproc/or-100-20-1-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-1-UC-20.cnf',
                     './input/Benchmark_preproc/or-100-20-1-UC-30.cnf', './input/Benchmark_preproc/or-100-20-2.cnf',
                     './input/Benchmark_preproc/or-100-20-2-UC-10.cnf', './input/Benchmark_preproc/or-100-20-3.cnf',
                     './input/Benchmark_preproc/or-100-20-3-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-3-UC-20.cnf', './input/Benchmark_preproc/or-100-20-4.cnf',
                     './input/Benchmark_preproc/or-100-20-4-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-4-UC-30.cnf', './input/Benchmark_preproc/or-100-20-5.cnf',
                     './input/Benchmark_preproc/or-100-20-5-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-5-UC-20.cnf', './input/Benchmark_preproc/or-100-20-6.cnf',
                     './input/Benchmark_preproc/or-100-20-6-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-6-UC-20.cnf', './input/Benchmark_preproc/or-100-20-7.cnf',
                     './input/Benchmark_preproc/or-100-20-7-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-7-UC-20.cnf', './input/Benchmark_preproc/or-100-20-8.cnf',
                     './input/Benchmark_preproc/or-100-20-8-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-20-8-UC-20.cnf', './input/Benchmark_preproc/or-100-20-9.cnf',
                     './input/Benchmark_preproc/or-100-20-9-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-10.cnf', './input/Benchmark_preproc/or-100-5-10-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-1.cnf',
                     './input/Benchmark_preproc/or-100-5-2.cnf', './input/Benchmark_preproc/or-100-5-2-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-3.cnf',
                     './input/Benchmark_preproc/or-100-5-4.cnf', './input/Benchmark_preproc/or-100-5-4-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-4-UC-20.cnf',
                     './input/Benchmark_preproc/or-100-5-5.cnf', './input/Benchmark_preproc/or-100-5-5-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-6.cnf',
                     './input/Benchmark_preproc/or-100-5-7.cnf', './input/Benchmark_preproc/or-100-5-7-UC-10.cnf',
                     './input/Benchmark_preproc/or-100-5-8.cnf',
                     './input/Benchmark_preproc/or-100-5-8-UC-10.cnf', './input/Benchmark_preproc/or-100-5-9.cnf',
                     './input/Benchmark_preproc/or-60-10-1.cnf', './input/Benchmark_preproc/or-60-10-3.cnf',
                     './input/Benchmark_preproc/or-60-10-4.cnf', './input/Benchmark_preproc/or-60-10-6.cnf',
                     './input/Benchmark_preproc/or-60-10-7.cnf', './input/Benchmark_preproc/or-60-10-8.cnf',
                     './input/Benchmark_preproc/or-60-10-9.cnf', './input/Benchmark_preproc/or-60-20-10.cnf',
                     './input/Benchmark_preproc/or-60-20-2.cnf', './input/Benchmark_preproc/or-60-20-3.cnf',
                     './input/Benchmark_preproc/or-60-20-4.cnf', './input/Benchmark_preproc/or-60-20-6.cnf',
                     './input/Benchmark_preproc/or-60-20-7.cnf', './input/Benchmark_preproc/or-60-20-9.cnf',
                     './input/Benchmark_preproc/or-60-5-1.cnf', './input/Benchmark_preproc/or-60-5-2.cnf',
                     './input/Benchmark_preproc/or-60-5-3.cnf', './input/Benchmark_preproc/or-60-5-6.cnf',
                     './input/Benchmark_preproc/or-60-5-8.cnf', './input/Benchmark_preproc/or-60-5-9.cnf',
                     './input/Benchmark_preproc/or-70-10-10.cnf', './input/Benchmark_preproc/or-70-10-1.cnf',
                     './input/Benchmark_preproc/or-70-10-2.cnf', './input/Benchmark_preproc/or-70-10-3.cnf',
                     './input/Benchmark_preproc/or-70-10-4.cnf', './input/Benchmark_preproc/or-70-10-5.cnf',
                     './input/Benchmark_preproc/or-70-10-6.cnf', './input/Benchmark_preproc/or-70-10-7.cnf',
                     './input/Benchmark_preproc/or-70-10-8.cnf', './input/Benchmark_preproc/or-70-10-9.cnf',
                     './input/Benchmark_preproc/or-70-20-10.cnf', './input/Benchmark_preproc/or-70-20-10-UC-20.cnf',
                     './input/Benchmark_preproc/or-70-20-1.cnf', './input/Benchmark_preproc/or-70-20-2.cnf',
                     './input/Benchmark_preproc/or-70-20-3.cnf', './input/Benchmark_preproc/or-70-20-3-UC-10.cnf',
                     './input/Benchmark_preproc/or-70-20-4.cnf', './input/Benchmark_preproc/or-70-20-4-UC-10.cnf',
                     './input/Benchmark_preproc/or-70-20-5.cnf', './input/Benchmark_preproc/or-70-20-6.cnf',
                     './input/Benchmark_preproc/or-70-20-7.cnf', './input/Benchmark_preproc/or-70-20-8.cnf',
                     './input/Benchmark_preproc/or-70-20-8-UC-10.cnf', './input/Benchmark_preproc/or-70-20-9.cnf',
                     './input/Benchmark_preproc/or-70-5-10.cnf', './input/Benchmark_preproc/or-70-5-1.cnf',
                     './input/Benchmark_preproc/or-70-5-2.cnf', './input/Benchmark_preproc/or-70-5-3.cnf',
                     './input/Benchmark_preproc/or-70-5-4.cnf', './input/Benchmark_preproc/or-70-5-5.cnf',
                     './input/Benchmark_preproc/or-70-5-6.cnf', './input/Benchmark_preproc/or-70-5-6-UC-10.cnf',
                     './input/Benchmark_preproc/or-70-5-7.cnf', './input/Benchmark_preproc/or-70-5-8.cnf',
                     './input/Benchmark_preproc/or-70-5-9.cnf', './input/Benchmark_preproc/par32-1-c.cnf',
                     './input/Benchmark_preproc/par32-1.cnf', './input/Benchmark_preproc/par32-2-c.cnf',
                     './input/Benchmark_preproc/par32-2.cnf', './input/Benchmark_preproc/par32-3-c.cnf',
                     './input/Benchmark_preproc/par32-3.cnf', './input/Benchmark_preproc/par32-4-c.cnf',
                     './input/Benchmark_preproc/par32-4.cnf', './input/Benchmark_preproc/par32-5-c.cnf',
                     './input/Benchmark_preproc/par32-5.cnf', './input/Benchmark_preproc/parity.sk_11_11.cnf',
                     './input/Benchmark_preproc/partition.sk_22_155.cnf',
                     './input/Benchmark_preproc/Pollard.sk_1_10.cnf', './input/Benchmark_preproc/prob005.pddl.cnf',
                     './input/Benchmark_preproc/prob012.pddl.cnf', './input/Benchmark_preproc/reverse.sk_11_258.cnf',
                     './input/Benchmark_preproc/s13207a_15_7.cnf', './input/Benchmark_preproc/s13207a_3_2.cnf',
                     './input/Benchmark_preproc/s13207a_7_4.cnf', './input/Benchmark_preproc/s15850a_15_7.cnf',
                     './input/Benchmark_preproc/s15850a_3_2.cnf', './input/Benchmark_preproc/s15850a_7_4.cnf',
                     './input/Benchmark_preproc/s38417_15_7.cnf', './input/Benchmark_preproc/s38417_3_2.cnf',
                     './input/Benchmark_preproc/s38417_7_4.cnf', './input/Benchmark_preproc/s38584_15_7.cnf',
                     './input/Benchmark_preproc/s38584_3_2.cnf', './input/Benchmark_preproc/s38584_7_4.cnf',
                     './input/Benchmark_preproc/s5378a_15_7.cnf', './input/Benchmark_preproc/s5378a_3_2.cnf',
                     './input/Benchmark_preproc/s5378a_7_4.cnf', './input/Benchmark_preproc/s9234a_15_7.cnf',
                     './input/Benchmark_preproc/s9234a_3_2.cnf', './input/Benchmark_preproc/s9234a_7_4.cnf',
                     './input/Benchmark_preproc/sat-grid-pbl-0030.cnf',
                     './input/Benchmark_preproc/scenarios_aig_traverse.sb.pl.sk_5_102.cnf',
                     './input/Benchmark_preproc/scenarios_lldelete1.sb.pl.sk_6_409.cnf',
                     './input/Benchmark_preproc/scenarios_llinsert2.sb.pl.sk_6_407.cnf',
                     './input/Benchmark_preproc/scenarios_llreverse.sb.pl.sk_8_25.cnf',
                     './input/Benchmark_preproc/scenarios_lltraversal.sb.pl.sk_5_23.cnf',
                     './input/Benchmark_preproc/scenarios_tree_delete4.sb.pl.sk_4_114.cnf']

    # new = []
    diff = {}
    for filename in files:
        f = os.path.join(d, filename)
        print("--------------------------------------------", filename)
        # if filename in already_processed_files:
        # if filename not in already_processed_files or './input/Benchmark_preproc/'+filename in not_compiling :
            # new.append(filename)
        var_diff, cls_diff = preprocess(f)
        diff[f] = [var_diff, cls_diff]
        print("--------------------------------------------")
    # print(new)
    # print(len(already_processed_files), len(new))
    print(diff)


def order_files_based_on_nb_vars():
    input = "./input/"
    folder = "Dataset_preproc/"
    # folder = "Benchmark_preproc2/"
    # folder = "Benchmark_original/"
    d = input + folder
    temp_files = [f for f in os.listdir(d) if re.match('.*\.cnf', f) and "temp" not in f ]
    # ecai23 = ['04_iscas89_s400_bench.cnf', '04_iscas89_s420_1_bench.cnf',
    #           '04_iscas89_s444_bench.cnf',
    #           '04_iscas89_s526_bench.cnf', '04_iscas89_s526n_bench.cnf', '05_iscas93_s344_bench.cnf',
    #           '05_iscas93_s499_bench.cnf', '06_iscas99_b01.cnf', '06_iscas99_b02.cnf', '06_iscas99_b03.cnf',
    #           '06_iscas99_b06.cnf',
    #           '06_iscas99_b08.cnf', '06_iscas99_b09.cnf', '06_iscas99_b10.cnf']
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
    files = []
    for e in temp_files:
        if e not in ecai23:
            files.append(e)

    files_order = {}
    eliminated = []
    for filename in files:
        f = os.path.join(d, filename)
        with open(f, "r") as freader:
            content = freader.readlines()
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
        # if nb_vars >= 15:
        #     files_order[f] = nb_vars
        # else:
        #     eliminated.append(f)
        files_order[f] = nb_vars

    sorted_files = sorted(files_order.items(), key=lambda x:x[1])
    print(len(sorted_files), len(files))
    print(sorted_files)
    print([f[0].split("/")[-1] for f in sorted_files])
    print("eliminated: ", eliminated)

if __name__ == "__main__":
    order_files_based_on_nb_vars()
    exit(1)

    ordered_instances = ['./input/Benchmark_original/blasted_case28.cnf', './input/Benchmark_original/blasted_case33.cnf', './input/Benchmark_original/blasted_case27.cnf', './input/Benchmark_original/blasted_case32.cnf', './input/Benchmark_original/blasted_case26.cnf', './input/Benchmark_original/blasted_case31.cnf', './input/Benchmark_original/blasted_case134.cnf', './input/Benchmark_original/blasted_case137.cnf', './input/Benchmark_original/ais6.cnf', './input/Benchmark_original/blasted_case36.cnf', './input/Benchmark_original/par8-1-c.cnf', './input/Benchmark_original/blasted_case29.cnf', './input/Benchmark_original/blasted_case24.cnf', './input/Benchmark_original/par8-4-c.cnf', './input/Benchmark_original/tutorial1.sk_1_1.cnf', './input/Benchmark_original/blasted_case30.cnf', './input/Benchmark_original/blasted_case25.cnf', './input/Benchmark_original/par8-2-c.cnf', './input/Benchmark_original/blasted_case100.cnf', './input/Benchmark_original/blasted_case101.cnf', './input/Benchmark_original/par8-3-c.cnf', './input/Benchmark_original/par8-5-c.cnf', './input/Benchmark_original/blasted_case17.cnf', './input/Benchmark_original/blasted_case23.cnf', './input/Benchmark_original/blasted_case59.cnf', './input/Benchmark_original/blasted_case59_1.cnf', './input/Benchmark_original/blasted_case64.cnf', './input/Benchmark_original/blasted_case58.cnf', './input/Benchmark_original/blasted_case63.cnf', './input/Benchmark_original/or-50-5-4-UC-30.cnf', './input/Benchmark_original/or-50-5-4-UC-40.cnf', './input/Benchmark_original/or-50-5-4.cnf', './input/Benchmark_original/or-50-5-5-UC-10.cnf', './input/Benchmark_original/or-50-5-5-UC-20.cnf', './input/Benchmark_original/or-50-5-5-UC-30.cnf', './input/Benchmark_original/or-50-5-5-UC-40.cnf', './input/Benchmark_original/or-50-5-5.cnf', './input/Benchmark_original/or-50-5-6-UC-10.cnf', './input/Benchmark_original/or-50-5-6-UC-20.cnf', './input/Benchmark_original/or-50-5-6-UC-30.cnf', './input/Benchmark_original/or-50-5-6-UC-40.cnf', './input/Benchmark_original/or-50-5-6.cnf', './input/Benchmark_original/or-50-5-7-UC-10.cnf', './input/Benchmark_original/or-50-5-7-UC-20.cnf', './input/Benchmark_original/or-50-5-7-UC-30.cnf', './input/Benchmark_original/or-50-5-7-UC-40.cnf', './input/Benchmark_original/or-50-5-10-UC-10.cnf', './input/Benchmark_original/or-50-5-10-UC-20.cnf', './input/Benchmark_original/or-50-5-10-UC-30.cnf', './input/Benchmark_original/or-50-5-10-UC-40.cnf', './input/Benchmark_original/or-50-5-10.cnf', './input/Benchmark_original/or-50-5-2-UC-10.cnf', './input/Benchmark_original/or-50-5-2-UC-20.cnf', './input/Benchmark_original/or-50-5-2-UC-30.cnf', './input/Benchmark_original/or-50-5-2-UC-40.cnf', './input/Benchmark_original/or-50-5-2.cnf', './input/Benchmark_original/or-50-5-3-UC-10.cnf', './input/Benchmark_original/or-50-5-3-UC-20.cnf', './input/Benchmark_original/or-50-5-3-UC-30.cnf', './input/Benchmark_original/or-50-5-3-UC-40.cnf', './input/Benchmark_original/or-50-5-3.cnf', './input/Benchmark_original/or-50-5-4-UC-10.cnf', './input/Benchmark_original/or-50-10-4-UC-20.cnf', './input/Benchmark_original/or-50-10-4-UC-30.cnf', './input/Benchmark_original/or-50-10-4-UC-40.cnf', './input/Benchmark_original/or-50-10-4.cnf', './input/Benchmark_original/or-50-10-5-UC-10.cnf', './input/Benchmark_original/or-50-10-5-UC-20.cnf', './input/Benchmark_original/or-50-10-5-UC-30.cnf', './input/Benchmark_original/or-50-10-5-UC-40.cnf', './input/Benchmark_original/or-50-10-5.cnf', './input/Benchmark_original/or-50-10-6-UC-10.cnf', './input/Benchmark_original/or-50-10-6-UC-20.cnf', './input/Benchmark_original/or-50-10-6-UC-30.cnf', './input/Benchmark_original/or-50-10-6-UC-40.cnf', './input/Benchmark_original/or-50-10-6.cnf', './input/Benchmark_original/or-50-10-7-UC-10.cnf', './input/Benchmark_original/or-50-10-7-UC-20.cnf', './input/Benchmark_original/or-50-10-7-UC-40.cnf', './input/Benchmark_original/or-50-10-7.cnf', './input/Benchmark_original/or-50-10-8-UC-10.cnf', './input/Benchmark_original/or-50-10-8-UC-20.cnf', './input/Benchmark_original/or-50-10-8-UC-30.cnf', './input/Benchmark_original/or-50-10-8-UC-40.cnf', './input/Benchmark_original/or-50-10-8.cnf', './input/Benchmark_original/or-50-10-9-UC-10.cnf', './input/Benchmark_original/or-50-10-9-UC-20.cnf', './input/Benchmark_original/or-50-10-9-UC-30.cnf', './input/Benchmark_original/or-50-10-9-UC-40.cnf', './input/Benchmark_original/or-50-10-9.cnf', './input/Benchmark_original/or-50-20-1-UC-10.cnf', './input/Benchmark_original/or-50-20-1-UC-20.cnf', './input/Benchmark_original/or-50-20-1-UC-30.cnf', './input/Benchmark_original/or-50-20-1-UC-40.cnf', './input/Benchmark_original/or-50-20-10-UC-10.cnf', './input/Benchmark_original/or-50-20-10-UC-20.cnf', './input/Benchmark_original/or-50-20-10-UC-30.cnf', './input/Benchmark_original/or-50-20-10-UC-40.cnf', './input/Benchmark_original/or-50-20-10.cnf', './input/Benchmark_original/or-50-20-2-UC-10.cnf', './input/Benchmark_original/or-50-20-2-UC-20.cnf', './input/Benchmark_original/or-50-20-2-UC-30.cnf', './input/Benchmark_original/or-50-20-2-UC-40.cnf', './input/Benchmark_original/or-50-20-2.cnf', './input/Benchmark_original/or-50-20-3-UC-10.cnf', './input/Benchmark_original/or-50-20-3-UC-20.cnf', './input/Benchmark_original/or-50-20-3-UC-30.cnf', './input/Benchmark_original/or-50-20-3-UC-40.cnf', './input/Benchmark_original/or-50-20-3.cnf', './input/Benchmark_original/or-50-20-4-UC-10.cnf', './input/Benchmark_original/or-50-20-4-UC-20.cnf', './input/Benchmark_original/or-50-20-4-UC-30.cnf', './input/Benchmark_original/or-50-20-4-UC-40.cnf', './input/Benchmark_original/or-50-20-4.cnf', './input/Benchmark_original/or-50-20-5-UC-10.cnf', './input/Benchmark_original/or-50-20-5-UC-30.cnf', './input/Benchmark_original/or-50-20-5-UC-40.cnf', './input/Benchmark_original/or-50-20-5.cnf', './input/Benchmark_original/or-50-20-6-UC-10.cnf', './input/Benchmark_original/or-50-20-6-UC-20.cnf', './input/Benchmark_original/or-50-20-6-UC-30.cnf', './input/Benchmark_original/or-50-20-6-UC-40.cnf', './input/Benchmark_original/or-50-20-6.cnf', './input/Benchmark_original/or-50-20-7-UC-10.cnf', './input/Benchmark_original/or-50-20-7-UC-20.cnf', './input/Benchmark_original/or-50-20-7-UC-30.cnf', './input/Benchmark_original/or-50-20-7-UC-40.cnf', './input/Benchmark_original/or-50-20-7.cnf', './input/Benchmark_original/or-50-20-8-UC-10.cnf', './input/Benchmark_original/or-50-20-8-UC-20.cnf', './input/Benchmark_original/or-50-20-8-UC-30.cnf', './input/Benchmark_original/or-50-20-8-UC-40.cnf', './input/Benchmark_original/or-50-20-8.cnf', './input/Benchmark_original/or-50-20-9-UC-10.cnf', './input/Benchmark_original/or-50-20-9-UC-20.cnf', './input/Benchmark_original/or-50-20-9-UC-30.cnf', './input/Benchmark_original/or-50-20-9-UC-40.cnf', './input/Benchmark_original/or-50-20-9.cnf', './input/Benchmark_original/or-50-5-1-UC-10.cnf', './input/Benchmark_original/or-50-5-1-UC-20.cnf', './input/Benchmark_original/or-50-5-1-UC-30.cnf', './input/Benchmark_original/or-50-5-1-UC-40.cnf', './input/Benchmark_original/or-50-10-1.cnf', './input/Benchmark_original/or-50-10-4-UC-10.cnf', './input/Benchmark_original/or-50-10-7-UC-30.cnf', './input/Benchmark_original/or-50-20-1.cnf', './input/Benchmark_original/or-50-20-5-UC-20.cnf', './input/Benchmark_original/or-50-5-1.cnf', './input/Benchmark_original/or-50-5-8-UC-10.cnf', './input/Benchmark_original/or-50-5-8-UC-20.cnf', './input/Benchmark_original/or-50-5-8-UC-30.cnf', './input/Benchmark_original/or-50-5-8-UC-40.cnf', './input/Benchmark_original/or-50-5-8.cnf', './input/Benchmark_original/or-50-5-9-UC-10.cnf', './input/Benchmark_original/or-50-5-9-UC-20.cnf', './input/Benchmark_original/or-50-5-9-UC-30.cnf', './input/Benchmark_original/or-50-5-9-UC-40.cnf', './input/Benchmark_original/or-50-5-9.cnf', './input/Benchmark_original/or-50-10-10-UC-10.cnf', './input/Benchmark_original/or-50-10-10-UC-20.cnf', './input/Benchmark_original/or-50-10-10-UC-30.cnf', './input/Benchmark_original/or-50-10-10-UC-40.cnf', './input/Benchmark_original/or-50-10-10.cnf', './input/Benchmark_original/or-50-10-2-UC-10.cnf', './input/Benchmark_original/or-50-10-2-UC-20.cnf', './input/Benchmark_original/or-50-10-2-UC-30.cnf', './input/Benchmark_original/or-50-10-2-UC-40.cnf', './input/Benchmark_original/or-50-10-2.cnf', './input/Benchmark_original/or-50-10-3-UC-10.cnf', './input/Benchmark_original/or-50-10-3-UC-20.cnf', './input/Benchmark_original/or-50-10-3-UC-30.cnf', './input/Benchmark_original/or-50-10-3-UC-40.cnf', './input/Benchmark_original/or-50-10-3.cnf', './input/Benchmark_original/or-50-5-7.cnf', './input/Benchmark_original/or-50-10-1-UC-10.cnf', './input/Benchmark_original/or-50-10-1-UC-20.cnf', './input/Benchmark_original/or-50-10-1-UC-30.cnf', './input/Benchmark_original/or-50-10-1-UC-40.cnf', './input/Benchmark_original/or-50-5-4-UC-20.cnf', './input/Benchmark_original/blasted_case4.cnf', './input/Benchmark_original/blasted_case11.cnf', './input/Benchmark_original/sat-grid-pbl-0010.cnf', './input/Benchmark_original/ais8.cnf', './input/Benchmark_original/blasted_case43.cnf', './input/Benchmark_original/blasted_case45.cnf', './input/Benchmark_original/blasted_case7.cnf', './input/Benchmark_original/medium.cnf', './input/Benchmark_original/blasted_case47.cnf', './input/Benchmark_original/or-60-10-1-UC-10.cnf', './input/Benchmark_original/or-60-10-1-UC-20.cnf', './input/Benchmark_original/or-60-10-1-UC-30.cnf', './input/Benchmark_original/or-60-10-1-UC-40.cnf', './input/Benchmark_original/or-60-10-1.cnf', './input/Benchmark_original/or-60-10-10-UC-10.cnf', './input/Benchmark_original/or-60-10-10-UC-20.cnf', './input/Benchmark_original/or-60-10-10-UC-30.cnf', './input/Benchmark_original/or-60-10-10-UC-40.cnf', './input/Benchmark_original/or-60-10-10.cnf', './input/Benchmark_original/or-60-10-2-UC-10.cnf', './input/Benchmark_original/or-60-5-2-UC-30.cnf', './input/Benchmark_original/or-60-5-2-UC-40.cnf', './input/Benchmark_original/or-60-5-2.cnf', './input/Benchmark_original/or-60-5-3-UC-10.cnf', './input/Benchmark_original/or-60-5-3-UC-20.cnf', './input/Benchmark_original/or-60-5-3-UC-30.cnf', './input/Benchmark_original/or-60-5-3-UC-40.cnf', './input/Benchmark_original/or-60-5-3.cnf', './input/Benchmark_original/or-60-5-4-UC-10.cnf', './input/Benchmark_original/or-60-5-4-UC-20.cnf', './input/Benchmark_original/or-60-5-4-UC-30.cnf', './input/Benchmark_original/or-60-5-4-UC-40.cnf', './input/Benchmark_original/or-60-5-4.cnf', './input/Benchmark_original/or-60-5-5-UC-10.cnf', './input/Benchmark_original/or-60-5-5-UC-20.cnf', './input/Benchmark_original/or-60-5-5-UC-30.cnf', './input/Benchmark_original/or-60-5-5-UC-40.cnf', './input/Benchmark_original/or-60-5-6-UC-10.cnf', './input/Benchmark_original/or-60-5-6-UC-20.cnf', './input/Benchmark_original/or-60-5-6-UC-30.cnf', './input/Benchmark_original/or-60-5-6-UC-40.cnf', './input/Benchmark_original/or-60-5-6.cnf', './input/Benchmark_original/or-60-5-7-UC-10.cnf', './input/Benchmark_original/or-60-5-7-UC-20.cnf', './input/Benchmark_original/or-60-5-7-UC-30.cnf', './input/Benchmark_original/or-60-5-7-UC-40.cnf', './input/Benchmark_original/or-60-5-7.cnf', './input/Benchmark_original/or-60-5-8-UC-10.cnf', './input/Benchmark_original/or-60-5-8-UC-20.cnf', './input/Benchmark_original/or-60-5-8-UC-30.cnf', './input/Benchmark_original/or-60-5-8-UC-40.cnf', './input/Benchmark_original/or-60-5-8.cnf', './input/Benchmark_original/or-60-5-9-UC-10.cnf', './input/Benchmark_original/or-60-10-2-UC-30.cnf', './input/Benchmark_original/or-60-10-2-UC-40.cnf', './input/Benchmark_original/or-60-10-2.cnf', './input/Benchmark_original/or-60-10-3-UC-10.cnf', './input/Benchmark_original/or-60-10-3-UC-20.cnf', './input/Benchmark_original/or-60-10-3-UC-30.cnf', './input/Benchmark_original/or-60-10-3-UC-40.cnf', './input/Benchmark_original/or-60-10-3.cnf', './input/Benchmark_original/or-60-10-4-UC-10.cnf', './input/Benchmark_original/or-60-10-4-UC-20.cnf', './input/Benchmark_original/or-60-10-4-UC-30.cnf', './input/Benchmark_original/or-60-10-4-UC-40.cnf', './input/Benchmark_original/or-60-10-4.cnf', './input/Benchmark_original/or-60-10-5-UC-10.cnf', './input/Benchmark_original/or-60-10-5-UC-20.cnf', './input/Benchmark_original/or-60-10-5-UC-30.cnf', './input/Benchmark_original/or-60-10-5.cnf', './input/Benchmark_original/or-60-10-6-UC-10.cnf', './input/Benchmark_original/or-60-10-6-UC-20.cnf', './input/Benchmark_original/or-60-10-6-UC-30.cnf', './input/Benchmark_original/or-60-10-6-UC-40.cnf', './input/Benchmark_original/or-60-10-6.cnf', './input/Benchmark_original/or-60-10-7-UC-10.cnf', './input/Benchmark_original/or-60-10-7-UC-20.cnf', './input/Benchmark_original/or-60-10-7-UC-30.cnf', './input/Benchmark_original/or-60-10-7-UC-40.cnf', './input/Benchmark_original/or-60-10-7.cnf', './input/Benchmark_original/or-60-10-8-UC-10.cnf', './input/Benchmark_original/or-60-10-8-UC-20.cnf', './input/Benchmark_original/or-60-10-8-UC-30.cnf', './input/Benchmark_original/or-60-10-8-UC-40.cnf', './input/Benchmark_original/or-60-10-8.cnf', './input/Benchmark_original/or-60-10-9-UC-20.cnf', './input/Benchmark_original/or-60-10-9-UC-30.cnf', './input/Benchmark_original/or-60-10-9-UC-40.cnf', './input/Benchmark_original/or-60-10-9.cnf', './input/Benchmark_original/or-60-20-1-UC-10.cnf', './input/Benchmark_original/or-60-20-1-UC-20.cnf', './input/Benchmark_original/or-60-20-1-UC-30.cnf', './input/Benchmark_original/or-60-20-1-UC-40.cnf', './input/Benchmark_original/or-60-20-1.cnf', './input/Benchmark_original/or-60-20-10-UC-10.cnf', './input/Benchmark_original/or-60-20-10-UC-20.cnf', './input/Benchmark_original/or-60-20-10-UC-30.cnf', './input/Benchmark_original/or-60-20-10-UC-40.cnf', './input/Benchmark_original/or-60-20-10.cnf', './input/Benchmark_original/or-60-20-2-UC-10.cnf', './input/Benchmark_original/or-60-20-2-UC-20.cnf', './input/Benchmark_original/or-60-20-2-UC-30.cnf', './input/Benchmark_original/or-60-20-2-UC-40.cnf', './input/Benchmark_original/or-60-20-2.cnf', './input/Benchmark_original/or-60-20-3-UC-10.cnf', './input/Benchmark_original/or-60-20-3-UC-20.cnf', './input/Benchmark_original/or-60-20-3-UC-40.cnf', './input/Benchmark_original/or-60-20-3.cnf', './input/Benchmark_original/or-60-20-4-UC-10.cnf', './input/Benchmark_original/or-60-20-4-UC-20.cnf', './input/Benchmark_original/or-60-20-4-UC-30.cnf', './input/Benchmark_original/or-60-20-4-UC-40.cnf', './input/Benchmark_original/or-60-20-4.cnf', './input/Benchmark_original/or-60-20-5-UC-10.cnf', './input/Benchmark_original/or-60-20-5-UC-20.cnf', './input/Benchmark_original/or-60-20-5-UC-30.cnf', './input/Benchmark_original/or-60-20-5-UC-40.cnf', './input/Benchmark_original/or-60-20-5.cnf', './input/Benchmark_original/or-60-20-6-UC-10.cnf', './input/Benchmark_original/or-60-20-6-UC-20.cnf', './input/Benchmark_original/or-60-20-6-UC-30.cnf', './input/Benchmark_original/or-60-20-6-UC-40.cnf', './input/Benchmark_original/or-60-20-7-UC-10.cnf', './input/Benchmark_original/or-60-20-7-UC-20.cnf', './input/Benchmark_original/or-60-20-7-UC-30.cnf', './input/Benchmark_original/or-60-20-7-UC-40.cnf', './input/Benchmark_original/or-60-20-7.cnf', './input/Benchmark_original/or-60-20-8-UC-10.cnf', './input/Benchmark_original/or-60-20-8-UC-20.cnf', './input/Benchmark_original/or-60-20-8-UC-30.cnf', './input/Benchmark_original/or-60-20-8-UC-40.cnf', './input/Benchmark_original/or-60-20-8.cnf', './input/Benchmark_original/or-60-20-9-UC-10.cnf', './input/Benchmark_original/or-60-20-9-UC-20.cnf', './input/Benchmark_original/or-60-20-9-UC-30.cnf', './input/Benchmark_original/or-60-20-9-UC-40.cnf', './input/Benchmark_original/or-60-20-9.cnf', './input/Benchmark_original/or-60-5-1-UC-10.cnf', './input/Benchmark_original/or-60-5-1-UC-20.cnf', './input/Benchmark_original/or-60-5-1-UC-30.cnf', './input/Benchmark_original/or-60-5-1-UC-40.cnf', './input/Benchmark_original/or-60-5-1.cnf', './input/Benchmark_original/or-60-5-10-UC-10.cnf', './input/Benchmark_original/or-60-5-10-UC-20.cnf', './input/Benchmark_original/or-60-5-10-UC-30.cnf', './input/Benchmark_original/or-60-5-10-UC-40.cnf', './input/Benchmark_original/or-60-5-10.cnf', './input/Benchmark_original/or-60-5-2-UC-10.cnf', './input/Benchmark_original/or-60-5-9-UC-30.cnf', './input/Benchmark_original/or-60-5-9-UC-40.cnf', './input/Benchmark_original/or-60-5-9.cnf', './input/Benchmark_original/or-60-10-2-UC-20.cnf', './input/Benchmark_original/or-60-10-5-UC-40.cnf', './input/Benchmark_original/or-60-10-9-UC-10.cnf', './input/Benchmark_original/or-60-20-3-UC-30.cnf', './input/Benchmark_original/or-60-20-6.cnf', './input/Benchmark_original/or-60-5-2-UC-20.cnf', './input/Benchmark_original/or-60-5-5.cnf', './input/Benchmark_original/or-60-5-9-UC-20.cnf', './input/Benchmark_original/2bitcomp_5.cnf', './input/Benchmark_original/blasted_case21.cnf', './input/Benchmark_original/blasted_case22.cnf', './input/Benchmark_original/blasted_case51.cnf', './input/Benchmark_original/blasted_case52.cnf', './input/Benchmark_original/blasted_case53.cnf', './input/Benchmark_original/blasted_case124.cnf', './input/Benchmark_original/blasted_case112.cnf', './input/Benchmark_original/or-70-10-1-UC-10.cnf', './input/Benchmark_original/or-70-10-1-UC-20.cnf', './input/Benchmark_original/or-70-10-1-UC-30.cnf', './input/Benchmark_original/or-70-10-1-UC-40.cnf', './input/Benchmark_original/or-70-10-1.cnf', './input/Benchmark_original/or-70-10-10-UC-10.cnf', './input/Benchmark_original/or-70-10-10-UC-20.cnf', './input/Benchmark_original/or-70-10-10-UC-30.cnf', './input/Benchmark_original/or-70-10-10-UC-40.cnf', './input/Benchmark_original/or-70-10-10.cnf', './input/Benchmark_original/or-70-10-2-UC-10.cnf', './input/Benchmark_original/or-70-10-2-UC-20.cnf', './input/Benchmark_original/or-70-10-2-UC-30.cnf', './input/Benchmark_original/or-70-10-2-UC-40.cnf', './input/Benchmark_original/or-70-10-2.cnf', './input/Benchmark_original/or-70-10-3-UC-10.cnf', './input/Benchmark_original/or-70-10-3-UC-20.cnf', './input/Benchmark_original/or-70-10-3-UC-30.cnf', './input/Benchmark_original/or-70-5-3-UC-40.cnf', './input/Benchmark_original/or-70-5-3.cnf', './input/Benchmark_original/or-70-5-4-UC-10.cnf', './input/Benchmark_original/or-70-5-4-UC-20.cnf', './input/Benchmark_original/or-70-5-4-UC-30.cnf', './input/Benchmark_original/or-70-5-4-UC-40.cnf', './input/Benchmark_original/or-70-5-4.cnf', './input/Benchmark_original/or-70-5-5-UC-10.cnf', './input/Benchmark_original/or-70-5-5-UC-20.cnf', './input/Benchmark_original/or-70-5-5-UC-30.cnf', './input/Benchmark_original/or-70-5-5-UC-40.cnf', './input/Benchmark_original/or-70-5-5.cnf', './input/Benchmark_original/or-70-5-6-UC-10.cnf', './input/Benchmark_original/or-70-5-6-UC-20.cnf', './input/Benchmark_original/or-70-5-6-UC-30.cnf', './input/Benchmark_original/or-70-5-6-UC-40.cnf', './input/Benchmark_original/or-70-5-6.cnf', './input/Benchmark_original/or-70-5-1-UC-20.cnf', './input/Benchmark_original/or-70-5-1-UC-30.cnf', './input/Benchmark_original/or-70-5-1-UC-40.cnf', './input/Benchmark_original/or-70-5-1.cnf', './input/Benchmark_original/or-70-5-10-UC-10.cnf', './input/Benchmark_original/or-70-5-10-UC-20.cnf', './input/Benchmark_original/or-70-5-10-UC-30.cnf', './input/Benchmark_original/or-70-5-10-UC-40.cnf', './input/Benchmark_original/or-70-5-10.cnf', './input/Benchmark_original/or-70-5-2-UC-10.cnf', './input/Benchmark_original/or-70-5-2-UC-20.cnf', './input/Benchmark_original/or-70-5-2-UC-30.cnf', './input/Benchmark_original/or-70-5-2-UC-40.cnf', './input/Benchmark_original/or-70-5-2.cnf', './input/Benchmark_original/or-70-5-3-UC-10.cnf', './input/Benchmark_original/or-70-5-3-UC-20.cnf', './input/Benchmark_original/or-70-10-3.cnf', './input/Benchmark_original/or-70-10-4-UC-10.cnf', './input/Benchmark_original/or-70-10-4-UC-20.cnf', './input/Benchmark_original/or-70-10-4-UC-30.cnf', './input/Benchmark_original/or-70-10-4-UC-40.cnf', './input/Benchmark_original/or-70-10-4.cnf', './input/Benchmark_original/or-70-10-5-UC-10.cnf', './input/Benchmark_original/or-70-10-5-UC-20.cnf', './input/Benchmark_original/or-70-10-5-UC-30.cnf', './input/Benchmark_original/or-70-10-5-UC-40.cnf', './input/Benchmark_original/or-70-10-5.cnf', './input/Benchmark_original/or-70-10-6-UC-10.cnf', './input/Benchmark_original/or-70-10-6-UC-20.cnf', './input/Benchmark_original/or-70-10-6-UC-30.cnf', './input/Benchmark_original/or-70-10-6-UC-40.cnf', './input/Benchmark_original/or-70-10-6.cnf', './input/Benchmark_original/or-70-10-7-UC-20.cnf', './input/Benchmark_original/or-70-10-7-UC-30.cnf', './input/Benchmark_original/or-70-10-7-UC-40.cnf', './input/Benchmark_original/or-70-10-7.cnf', './input/Benchmark_original/or-70-10-8-UC-10.cnf', './input/Benchmark_original/or-70-10-8-UC-20.cnf', './input/Benchmark_original/or-70-10-8-UC-30.cnf', './input/Benchmark_original/or-70-10-8-UC-40.cnf', './input/Benchmark_original/or-70-10-8.cnf', './input/Benchmark_original/or-70-10-9-UC-10.cnf', './input/Benchmark_original/or-70-10-9-UC-20.cnf', './input/Benchmark_original/or-70-10-9-UC-30.cnf', './input/Benchmark_original/or-70-10-9-UC-40.cnf', './input/Benchmark_original/or-70-10-9.cnf', './input/Benchmark_original/or-70-20-1-UC-10.cnf', './input/Benchmark_original/or-70-20-1-UC-20.cnf', './input/Benchmark_original/or-70-20-1-UC-40.cnf', './input/Benchmark_original/or-70-20-1.cnf', './input/Benchmark_original/or-70-20-10-UC-10.cnf', './input/Benchmark_original/or-70-20-10-UC-20.cnf', './input/Benchmark_original/or-70-20-10-UC-30.cnf', './input/Benchmark_original/or-70-20-10-UC-40.cnf', './input/Benchmark_original/or-70-20-10.cnf', './input/Benchmark_original/or-70-20-2-UC-10.cnf', './input/Benchmark_original/or-70-20-2-UC-20.cnf', './input/Benchmark_original/or-70-20-2-UC-30.cnf', './input/Benchmark_original/or-70-20-2-UC-40.cnf', './input/Benchmark_original/or-70-20-2.cnf', './input/Benchmark_original/or-70-20-3-UC-10.cnf', './input/Benchmark_original/or-70-20-3-UC-20.cnf', './input/Benchmark_original/or-70-20-3-UC-30.cnf', './input/Benchmark_original/or-70-20-3-UC-40.cnf', './input/Benchmark_original/or-70-20-3.cnf', './input/Benchmark_original/or-70-20-4-UC-10.cnf', './input/Benchmark_original/or-70-20-4-UC-20.cnf', './input/Benchmark_original/or-70-20-4-UC-30.cnf', './input/Benchmark_original/or-70-20-4-UC-40.cnf', './input/Benchmark_original/or-70-20-5-UC-10.cnf', './input/Benchmark_original/or-70-20-5-UC-20.cnf', './input/Benchmark_original/or-70-20-5-UC-30.cnf', './input/Benchmark_original/or-70-20-5-UC-40.cnf', './input/Benchmark_original/or-70-20-5.cnf', './input/Benchmark_original/or-70-20-6-UC-10.cnf', './input/Benchmark_original/or-70-20-6-UC-20.cnf', './input/Benchmark_original/or-70-20-6-UC-30.cnf', './input/Benchmark_original/or-70-20-6-UC-40.cnf', './input/Benchmark_original/or-70-20-6.cnf', './input/Benchmark_original/or-70-20-7-UC-10.cnf', './input/Benchmark_original/or-70-20-7-UC-20.cnf', './input/Benchmark_original/or-70-20-7-UC-30.cnf', './input/Benchmark_original/or-70-20-7-UC-40.cnf', './input/Benchmark_original/or-70-20-7.cnf', './input/Benchmark_original/or-70-20-8-UC-10.cnf', './input/Benchmark_original/or-70-20-8-UC-20.cnf', './input/Benchmark_original/or-70-20-8-UC-30.cnf', './input/Benchmark_original/or-70-20-8-UC-40.cnf', './input/Benchmark_original/or-70-20-8.cnf', './input/Benchmark_original/or-70-20-9-UC-10.cnf', './input/Benchmark_original/or-70-20-9-UC-20.cnf', './input/Benchmark_original/or-70-20-9-UC-30.cnf', './input/Benchmark_original/or-70-20-9-UC-40.cnf', './input/Benchmark_original/or-70-20-9.cnf', './input/Benchmark_original/or-70-10-3-UC-40.cnf', './input/Benchmark_original/or-70-10-7-UC-10.cnf', './input/Benchmark_original/or-70-20-1-UC-30.cnf', './input/Benchmark_original/or-70-20-4.cnf', './input/Benchmark_original/or-70-5-1-UC-10.cnf', './input/Benchmark_original/or-70-5-3-UC-30.cnf', './input/Benchmark_original/or-70-5-7-UC-20.cnf', './input/Benchmark_original/or-70-5-7-UC-30.cnf', './input/Benchmark_original/or-70-5-7-UC-40.cnf', './input/Benchmark_original/or-70-5-7.cnf', './input/Benchmark_original/or-70-5-8-UC-10.cnf', './input/Benchmark_original/or-70-5-8-UC-20.cnf', './input/Benchmark_original/or-70-5-8-UC-30.cnf', './input/Benchmark_original/or-70-5-8-UC-40.cnf', './input/Benchmark_original/or-70-5-8.cnf', './input/Benchmark_original/or-70-5-9-UC-10.cnf', './input/Benchmark_original/or-70-5-9-UC-20.cnf', './input/Benchmark_original/or-70-5-9-UC-30.cnf', './input/Benchmark_original/or-70-5-9-UC-40.cnf', './input/Benchmark_original/or-70-5-9.cnf', './input/Benchmark_original/or-70-5-7-UC-10.cnf', './input/Benchmark_original/blasted_case38.cnf', './input/Benchmark_original/blasted_case55.cnf', './input/Benchmark_original/blasted_case8.cnf', './input/Benchmark_original/4step.cnf', './input/Benchmark_original/blasted_case105.cnf', './input/Benchmark_original/blasted_case44.cnf', './input/Benchmark_original/blasted_case46.cnf', './input/Benchmark_original/blasted_case5.cnf', './input/Benchmark_original/5step.cnf', './input/Benchmark_original/blasted_case68.cnf', './input/Benchmark_original/ais10.cnf', './input/Benchmark_original/blasted_case1.cnf', './input/Benchmark_original/s400.bench.cnf', './input/Benchmark_original/20_rd_r45.cnf', './input/Benchmark_original/countdump2.cnf', './input/Benchmark_original/nocountdump26.cnf', './input/Benchmark_original/nocountdump27.cnf', './input/Benchmark_original/nocountdump28.cnf', './input/Benchmark_original/c432.isc.cnf', './input/Benchmark_original/s344_3_2.cnf', './input/Benchmark_original/s349_3_2.cnf', './input/Benchmark_original/blasted_case201.cnf', './input/Benchmark_original/blasted_case202.cnf', './input/Benchmark_original/fphp-010-020.cnf', './input/Benchmark_original/or-100-10-1-UC-10.cnf', './input/Benchmark_original/or-100-10-1-UC-20.cnf', './input/Benchmark_original/or-100-10-1-UC-30.cnf', './input/Benchmark_original/or-100-10-1-UC-40.cnf', './input/Benchmark_original/or-100-10-1-UC-50.cnf', './input/Benchmark_original/or-100-10-1-UC-60.cnf', './input/Benchmark_original/or-100-10-1.cnf', './input/Benchmark_original/or-100-10-10-UC-10.cnf', './input/Benchmark_original/or-100-10-10-UC-20.cnf', './input/Benchmark_original/or-100-10-10-UC-30.cnf', './input/Benchmark_original/or-100-10-10-UC-40.cnf', './input/Benchmark_original/or-100-10-10-UC-50.cnf', './input/Benchmark_original/or-100-10-10-UC-60.cnf', './input/Benchmark_original/or-100-10-10.cnf', './input/Benchmark_original/or-100-10-2-UC-10.cnf', './input/Benchmark_original/or-100-10-2-UC-20.cnf', './input/Benchmark_original/or-100-10-2-UC-30.cnf', './input/Benchmark_original/or-100-10-2-UC-40.cnf', './input/Benchmark_original/or-100-10-2-UC-50.cnf', './input/Benchmark_original/or-100-10-2-UC-60.cnf', './input/Benchmark_original/or-100-10-2.cnf', './input/Benchmark_original/or-100-10-3-UC-10.cnf', './input/Benchmark_original/or-100-10-5-UC-40.cnf', './input/Benchmark_original/or-100-10-7.cnf', './input/Benchmark_original/or-100-20-1-UC-30.cnf', './input/Benchmark_original/or-100-20-3-UC-60.cnf', './input/Benchmark_original/or-100-20-6-UC-20.cnf', './input/Benchmark_original/or-100-20-8-UC-50.cnf', './input/Benchmark_original/or-100-5-10-UC-20.cnf', './input/Benchmark_original/or-100-5-3-UC-40.cnf', './input/Benchmark_original/or-100-5-5.cnf', './input/Benchmark_original/or-100-5-8-UC-20.cnf', './input/Benchmark_original/or-100-5-3-UC-50.cnf', './input/Benchmark_original/or-100-5-3-UC-60.cnf', './input/Benchmark_original/or-100-5-3.cnf', './input/Benchmark_original/or-100-5-4-UC-10.cnf', './input/Benchmark_original/or-100-5-4-UC-20.cnf', './input/Benchmark_original/or-100-5-4-UC-30.cnf', './input/Benchmark_original/or-100-5-4-UC-40.cnf', './input/Benchmark_original/or-100-5-4-UC-50.cnf', './input/Benchmark_original/or-100-5-4-UC-60.cnf', './input/Benchmark_original/or-100-5-4.cnf', './input/Benchmark_original/or-100-5-5-UC-10.cnf', './input/Benchmark_original/or-100-5-5-UC-20.cnf', './input/Benchmark_original/or-100-5-5-UC-30.cnf', './input/Benchmark_original/or-100-5-5-UC-40.cnf', './input/Benchmark_original/or-100-5-5-UC-50.cnf', './input/Benchmark_original/or-100-5-5-UC-60.cnf', './input/Benchmark_original/or-100-5-6-UC-10.cnf', './input/Benchmark_original/or-100-5-6-UC-20.cnf', './input/Benchmark_original/or-100-5-6-UC-30.cnf', './input/Benchmark_original/or-100-5-6-UC-40.cnf', './input/Benchmark_original/or-100-5-6-UC-50.cnf', './input/Benchmark_original/or-100-5-6-UC-60.cnf', './input/Benchmark_original/or-100-5-6.cnf', './input/Benchmark_original/or-100-5-7-UC-10.cnf', './input/Benchmark_original/or-100-5-7-UC-20.cnf', './input/Benchmark_original/or-100-5-7-UC-30.cnf', './input/Benchmark_original/or-100-5-7-UC-40.cnf', './input/Benchmark_original/or-100-5-7-UC-50.cnf', './input/Benchmark_original/or-100-5-7-UC-60.cnf', './input/Benchmark_original/or-100-5-7.cnf', './input/Benchmark_original/or-100-5-8-UC-10.cnf', './input/Benchmark_original/or-100-5-8-UC-30.cnf', './input/Benchmark_original/or-100-5-8-UC-40.cnf', './input/Benchmark_original/or-100-5-8-UC-50.cnf', './input/Benchmark_original/or-100-5-8-UC-60.cnf', './input/Benchmark_original/or-100-5-8.cnf', './input/Benchmark_original/or-100-5-9-UC-10.cnf', './input/Benchmark_original/or-100-5-9-UC-20.cnf', './input/Benchmark_original/or-100-5-9-UC-30.cnf', './input/Benchmark_original/or-100-5-9-UC-40.cnf', './input/Benchmark_original/or-100-5-9-UC-50.cnf', './input/Benchmark_original/or-100-5-9-UC-60.cnf', './input/Benchmark_original/or-100-5-9.cnf', './input/Benchmark_original/or-100-5-10-UC-30.cnf', './input/Benchmark_original/or-100-5-10-UC-40.cnf', './input/Benchmark_original/or-100-5-10-UC-50.cnf', './input/Benchmark_original/or-100-5-10-UC-60.cnf', './input/Benchmark_original/or-100-5-10.cnf', './input/Benchmark_original/or-100-5-2-UC-10.cnf', './input/Benchmark_original/or-100-5-2-UC-20.cnf', './input/Benchmark_original/or-100-5-2-UC-30.cnf', './input/Benchmark_original/or-100-5-2-UC-40.cnf', './input/Benchmark_original/or-100-5-2-UC-50.cnf', './input/Benchmark_original/or-100-5-2-UC-60.cnf', './input/Benchmark_original/or-100-5-2.cnf', './input/Benchmark_original/or-100-5-3-UC-10.cnf', './input/Benchmark_original/or-100-5-3-UC-20.cnf', './input/Benchmark_original/or-100-5-3-UC-30.cnf', './input/Benchmark_original/or-100-10-3-UC-20.cnf', './input/Benchmark_original/or-100-10-3-UC-30.cnf', './input/Benchmark_original/or-100-10-3-UC-40.cnf', './input/Benchmark_original/or-100-10-3-UC-50.cnf', './input/Benchmark_original/or-100-10-3-UC-60.cnf', './input/Benchmark_original/or-100-10-3.cnf', './input/Benchmark_original/or-100-10-4-UC-10.cnf', './input/Benchmark_original/or-100-10-4-UC-20.cnf', './input/Benchmark_original/or-100-10-4-UC-30.cnf', './input/Benchmark_original/or-100-10-4-UC-40.cnf', './input/Benchmark_original/or-100-10-4-UC-50.cnf', './input/Benchmark_original/or-100-10-4-UC-60.cnf', './input/Benchmark_original/or-100-10-4.cnf', './input/Benchmark_original/or-100-10-5-UC-10.cnf', './input/Benchmark_original/or-100-10-5-UC-20.cnf', './input/Benchmark_original/or-100-10-5-UC-30.cnf', './input/Benchmark_original/or-100-10-5-UC-50.cnf', './input/Benchmark_original/or-100-10-5-UC-60.cnf', './input/Benchmark_original/or-100-10-5.cnf', './input/Benchmark_original/or-100-10-6-UC-10.cnf', './input/Benchmark_original/or-100-10-6-UC-20.cnf', './input/Benchmark_original/or-100-10-6-UC-30.cnf', './input/Benchmark_original/or-100-10-6-UC-40.cnf', './input/Benchmark_original/or-100-10-6-UC-50.cnf', './input/Benchmark_original/or-100-10-6-UC-60.cnf', './input/Benchmark_original/or-100-10-6.cnf', './input/Benchmark_original/or-100-10-7-UC-10.cnf', './input/Benchmark_original/or-100-10-7-UC-20.cnf', './input/Benchmark_original/or-100-10-7-UC-30.cnf', './input/Benchmark_original/or-100-10-7-UC-40.cnf', './input/Benchmark_original/or-100-10-7-UC-50.cnf', './input/Benchmark_original/or-100-10-7-UC-60.cnf', './input/Benchmark_original/or-100-10-8-UC-10.cnf', './input/Benchmark_original/or-100-10-8-UC-20.cnf', './input/Benchmark_original/or-100-10-8-UC-30.cnf', './input/Benchmark_original/or-100-10-8-UC-40.cnf', './input/Benchmark_original/or-100-10-8-UC-50.cnf', './input/Benchmark_original/or-100-10-8-UC-60.cnf', './input/Benchmark_original/or-100-10-8.cnf', './input/Benchmark_original/or-100-10-9-UC-10.cnf', './input/Benchmark_original/or-100-10-9-UC-20.cnf', './input/Benchmark_original/or-100-10-9-UC-30.cnf', './input/Benchmark_original/or-100-10-9-UC-40.cnf', './input/Benchmark_original/or-100-10-9-UC-50.cnf', './input/Benchmark_original/or-100-10-9-UC-60.cnf', './input/Benchmark_original/or-100-10-9.cnf', './input/Benchmark_original/or-100-20-1-UC-10.cnf', './input/Benchmark_original/or-100-20-1-UC-20.cnf', './input/Benchmark_original/or-100-20-1-UC-40.cnf', './input/Benchmark_original/or-100-20-1-UC-50.cnf', './input/Benchmark_original/or-100-20-1-UC-60.cnf', './input/Benchmark_original/or-100-20-1.cnf', './input/Benchmark_original/or-100-20-10-UC-10.cnf', './input/Benchmark_original/or-100-20-10-UC-20.cnf', './input/Benchmark_original/or-100-20-10-UC-30.cnf', './input/Benchmark_original/or-100-20-10-UC-40.cnf', './input/Benchmark_original/or-100-20-10-UC-50.cnf', './input/Benchmark_original/or-100-20-10-UC-60.cnf', './input/Benchmark_original/or-100-20-10.cnf', './input/Benchmark_original/or-100-20-2-UC-10.cnf', './input/Benchmark_original/or-100-20-2-UC-20.cnf', './input/Benchmark_original/or-100-20-2-UC-30.cnf', './input/Benchmark_original/or-100-20-2-UC-40.cnf', './input/Benchmark_original/or-100-20-2-UC-50.cnf', './input/Benchmark_original/or-100-20-2-UC-60.cnf', './input/Benchmark_original/or-100-20-2.cnf', './input/Benchmark_original/or-100-20-3-UC-10.cnf', './input/Benchmark_original/or-100-20-3-UC-20.cnf', './input/Benchmark_original/or-100-20-3-UC-30.cnf', './input/Benchmark_original/or-100-20-3-UC-40.cnf', './input/Benchmark_original/or-100-20-3-UC-50.cnf', './input/Benchmark_original/or-100-20-3.cnf', './input/Benchmark_original/or-100-20-4-UC-10.cnf', './input/Benchmark_original/or-100-20-4-UC-20.cnf', './input/Benchmark_original/or-100-20-4-UC-30.cnf', './input/Benchmark_original/or-100-20-4-UC-40.cnf', './input/Benchmark_original/or-100-20-4-UC-50.cnf', './input/Benchmark_original/or-100-20-4-UC-60.cnf', './input/Benchmark_original/or-100-20-4.cnf', './input/Benchmark_original/or-100-20-5-UC-10.cnf', './input/Benchmark_original/or-100-20-5-UC-20.cnf', './input/Benchmark_original/or-100-20-5-UC-30.cnf', './input/Benchmark_original/or-100-20-5-UC-40.cnf', './input/Benchmark_original/or-100-20-5-UC-50.cnf', './input/Benchmark_original/or-100-20-5-UC-60.cnf', './input/Benchmark_original/or-100-20-5.cnf', './input/Benchmark_original/or-100-20-6-UC-10.cnf', './input/Benchmark_original/or-100-20-6-UC-30.cnf', './input/Benchmark_original/or-100-20-6-UC-40.cnf', './input/Benchmark_original/or-100-20-6-UC-50.cnf', './input/Benchmark_original/or-100-20-6-UC-60.cnf', './input/Benchmark_original/or-100-20-6.cnf', './input/Benchmark_original/or-100-20-7-UC-10.cnf', './input/Benchmark_original/or-100-20-7-UC-20.cnf', './input/Benchmark_original/or-100-20-7-UC-30.cnf', './input/Benchmark_original/or-100-20-7-UC-40.cnf', './input/Benchmark_original/or-100-20-7-UC-50.cnf', './input/Benchmark_original/or-100-20-7-UC-60.cnf', './input/Benchmark_original/or-100-20-7.cnf', './input/Benchmark_original/or-100-20-8-UC-10.cnf', './input/Benchmark_original/or-100-20-8-UC-20.cnf', './input/Benchmark_original/or-100-20-8-UC-30.cnf', './input/Benchmark_original/or-100-20-8-UC-40.cnf', './input/Benchmark_original/or-100-20-8-UC-60.cnf', './input/Benchmark_original/or-100-20-8.cnf', './input/Benchmark_original/or-100-20-9-UC-10.cnf', './input/Benchmark_original/or-100-20-9-UC-20.cnf', './input/Benchmark_original/or-100-20-9-UC-30.cnf', './input/Benchmark_original/or-100-20-9-UC-40.cnf', './input/Benchmark_original/or-100-20-9-UC-50.cnf', './input/Benchmark_original/or-100-20-9-UC-60.cnf', './input/Benchmark_original/or-100-20-9.cnf', './input/Benchmark_original/or-100-5-1-UC-10.cnf', './input/Benchmark_original/or-100-5-1-UC-20.cnf', './input/Benchmark_original/or-100-5-1-UC-30.cnf', './input/Benchmark_original/or-100-5-1-UC-40.cnf', './input/Benchmark_original/or-100-5-1-UC-50.cnf', './input/Benchmark_original/or-100-5-1-UC-60.cnf', './input/Benchmark_original/or-100-5-1.cnf', './input/Benchmark_original/or-100-5-10-UC-10.cnf', './input/Benchmark_original/blasted_case56.cnf', './input/Benchmark_original/blasted_case54.cnf', './input/Benchmark_original/blasted_case106.cnf', './input/Benchmark_original/blasted_case108.cnf', './input/Benchmark_original/s444.bench.cnf', './input/Benchmark_original/s298_3_2.cnf', './input/Benchmark_original/blasted_case133.cnf', './input/Benchmark_original/blasted_case136.cnf', './input/Benchmark_original/blasted_case203.cnf', './input/Benchmark_original/blasted_case204.cnf', './input/Benchmark_original/blasted_case205.cnf', './input/Benchmark_original/s344_7_4.cnf', './input/Benchmark_original/s349_7_4.cnf', './input/Benchmark_original/s526.bench.cnf', './input/Benchmark_original/s526n.bench.cnf', './input/Benchmark_original/blasted_case145.cnf', './input/Benchmark_original/blasted_case146.cnf', './input/Benchmark_original/nocountdump5.cnf', './input/Benchmark_original/s298_7_4.cnf', './input/Benchmark_original/countdump7.cnf', './input/Benchmark_original/blasted_case132.cnf', './input/Benchmark_original/blasted_case135.cnf', './input/Benchmark_original/s510.bench.cnf', './input/Benchmark_original/blasted_case_1_b14_1.cnf', './input/Benchmark_original/blasted_case_2_b14_1.cnf', './input/Benchmark_original/blasted_case_3_b14_1.cnf', './input/Benchmark_original/sat-grid-pbl-0015.cnf', './input/Benchmark_original/blasted_case109.cnf', './input/Benchmark_original/c499.isc.cnf', './input/Benchmark_original/blasted_case40.cnf', './input/Benchmark_original/blasted_case39.cnf', './input/Benchmark_original/blasted_case41.cnf', './input/Benchmark_original/blasted_case14.cnf', './input/Benchmark_original/uf250-01.cnf', './input/Benchmark_original/uf250-010.cnf', './input/Benchmark_original/uf250-0100.cnf', './input/Benchmark_original/uf250-011.cnf', './input/Benchmark_original/uf250-012.cnf', './input/Benchmark_original/uf250-014.cnf', './input/Benchmark_original/uf250-015.cnf', './input/Benchmark_original/uf250-016.cnf', './input/Benchmark_original/uf250-017.cnf', './input/Benchmark_original/uf250-018.cnf', './input/Benchmark_original/uf250-019.cnf', './input/Benchmark_original/uf250-02.cnf', './input/Benchmark_original/uf250-020.cnf', './input/Benchmark_original/uf250-021.cnf', './input/Benchmark_original/uf250-022.cnf', './input/Benchmark_original/uf250-023.cnf', './input/Benchmark_original/uf250-024.cnf', './input/Benchmark_original/uf250-025.cnf', './input/Benchmark_original/uf250-026.cnf', './input/Benchmark_original/uf250-027.cnf', './input/Benchmark_original/uf250-028.cnf', './input/Benchmark_original/uf250-029.cnf', './input/Benchmark_original/uf250-030.cnf', './input/Benchmark_original/uf250-031.cnf', './input/Benchmark_original/uf250-032.cnf', './input/Benchmark_original/uf250-033.cnf', './input/Benchmark_original/uf250-034.cnf', './input/Benchmark_original/uf250-035.cnf', './input/Benchmark_original/uf250-036.cnf', './input/Benchmark_original/uf250-037.cnf', './input/Benchmark_original/uf250-038.cnf', './input/Benchmark_original/uf250-039.cnf', './input/Benchmark_original/uf250-04.cnf', './input/Benchmark_original/uf250-040.cnf', './input/Benchmark_original/uf250-041.cnf', './input/Benchmark_original/uf250-042.cnf', './input/Benchmark_original/uf250-043.cnf', './input/Benchmark_original/uf250-044.cnf', './input/Benchmark_original/uf250-045.cnf', './input/Benchmark_original/uf250-047.cnf', './input/Benchmark_original/uf250-048.cnf', './input/Benchmark_original/uf250-049.cnf', './input/Benchmark_original/uf250-05.cnf', './input/Benchmark_original/uf250-050.cnf', './input/Benchmark_original/uf250-051.cnf', './input/Benchmark_original/uf250-052.cnf', './input/Benchmark_original/uf250-053.cnf', './input/Benchmark_original/uf250-054.cnf', './input/Benchmark_original/uf250-055.cnf', './input/Benchmark_original/uf250-056.cnf', './input/Benchmark_original/uf250-057.cnf', './input/Benchmark_original/uf250-058.cnf', './input/Benchmark_original/uf250-059.cnf', './input/Benchmark_original/uf250-06.cnf', './input/Benchmark_original/uf250-060.cnf', './input/Benchmark_original/uf250-061.cnf', './input/Benchmark_original/uf250-063.cnf', './input/Benchmark_original/uf250-064.cnf', './input/Benchmark_original/uf250-065.cnf', './input/Benchmark_original/uf250-066.cnf', './input/Benchmark_original/uf250-067.cnf', './input/Benchmark_original/uf250-068.cnf', './input/Benchmark_original/uf250-069.cnf', './input/Benchmark_original/uf250-07.cnf', './input/Benchmark_original/uf250-070.cnf', './input/Benchmark_original/uf250-071.cnf', './input/Benchmark_original/uf250-072.cnf', './input/Benchmark_original/uf250-073.cnf', './input/Benchmark_original/uf250-074.cnf', './input/Benchmark_original/uf250-075.cnf', './input/Benchmark_original/uf250-076.cnf', './input/Benchmark_original/uf250-077.cnf', './input/Benchmark_original/uf250-078.cnf', './input/Benchmark_original/uf250-08.cnf', './input/Benchmark_original/uf250-080.cnf', './input/Benchmark_original/uf250-081.cnf', './input/Benchmark_original/uf250-082.cnf', './input/Benchmark_original/uf250-083.cnf', './input/Benchmark_original/uf250-084.cnf', './input/Benchmark_original/uf250-085.cnf', './input/Benchmark_original/uf250-086.cnf', './input/Benchmark_original/uf250-087.cnf', './input/Benchmark_original/uf250-088.cnf', './input/Benchmark_original/uf250-089.cnf', './input/Benchmark_original/uf250-09.cnf', './input/Benchmark_original/uf250-090.cnf', './input/Benchmark_original/uf250-091.cnf', './input/Benchmark_original/uf250-092.cnf', './input/Benchmark_original/uf250-093.cnf', './input/Benchmark_original/uf250-094.cnf', './input/Benchmark_original/uf250-095.cnf', './input/Benchmark_original/uf250-096.cnf', './input/Benchmark_original/uf250-097.cnf', './input/Benchmark_original/uf250-098.cnf', './input/Benchmark_original/uf250-099.cnf', './input/Benchmark_original/uf250-013.cnf', './input/Benchmark_original/uf250-03.cnf', './input/Benchmark_original/uf250-046.cnf', './input/Benchmark_original/uf250-062.cnf', './input/Benchmark_original/uf250-079.cnf', './input/Benchmark_original/binsearch.16.pp.cnf', './input/Benchmark_original/binsearch.32.pp.cnf', './input/Benchmark_original/nocountdump29.cnf', './input/Benchmark_original/nocountdump30.cnf', './input/Benchmark_original/s420.1.bench.cnf', './input/Benchmark_original/2bitmax_6.cnf', './input/Benchmark_original/nocountdump14.cnf', './input/Benchmark_original/nocountdump15.cnf', './input/Benchmark_original/23_rd_r45.cnf', './input/Benchmark_original/s382_3_2.cnf', './input/Benchmark_original/ais12.cnf', './input/Benchmark_original/blasted_case119.cnf', './input/Benchmark_original/blasted_case123.cnf', './input/Benchmark_original/blasted_case_1_b14_2.cnf', './input/Benchmark_original/blasted_case_2_b14_2.cnf', './input/Benchmark_original/blasted_case_3_b14_2.cnf', './input/Benchmark_original/blasted_case9.cnf', './input/Benchmark_original/s382_7_4.cnf', './input/Benchmark_original/blasted_case61.cnf', './input/Benchmark_original/3blocks.cnf', './input/Benchmark_original/blasted_case120.cnf', './input/Benchmark_original/s344_15_7.cnf', './input/Benchmark_original/s349_15_7.cnf', './input/Benchmark_original/blasted_case110.cnf', './input/Benchmark_original/blasted_case57.cnf', './input/Benchmark_original/s444_3_2.cnf', './input/Benchmark_original/blasted_case121.cnf', './input/Benchmark_original/blasted_case62.cnf', './input/Benchmark_original/s298_15_7.cnf', './input/Benchmark_original/blasted_case3.cnf', './input/Benchmark_original/s420_3_2.cnf', './input/Benchmark_original/s420_new1_3_2.cnf', './input/Benchmark_original/s420_new_3_2.cnf', './input/Benchmark_original/blasted_case15.cnf', './input/Benchmark_original/blasted_case2.cnf', './input/Benchmark_original/s510_3_2.cnf', './input/Benchmark_original/fphp-015-020.cnf', './input/Benchmark_original/ls8-normalized.cnf', './input/Benchmark_original/blasted_case126.cnf', './input/Benchmark_original/blasted_case_1_b14_3.cnf', './input/Benchmark_original/blasted_case_2_b14_3.cnf', './input/Benchmark_original/blasted_case_3_b14_3.cnf', './input/Benchmark_original/blasted_case111.cnf', './input/Benchmark_original/s444_7_4.cnf', './input/Benchmark_original/blasted_case113.cnf', './input/Benchmark_original/blasted_case117.cnf', './input/Benchmark_original/blasted_case118.cnf', './input/Benchmark_original/mixdup.cnf', './input/Benchmark_original/s832.bench.cnf', './input/Benchmark_original/s420_7_4.cnf', './input/Benchmark_original/s420_new1_7_4.cnf', './input/Benchmark_original/s420_new_7_4.cnf', './input/Benchmark_original/s820.bench.cnf', './input/Benchmark_original/polynomial.sk_7_25.cnf', './input/Benchmark_original/blasted_case122.cnf', './input/Benchmark_original/countdump1.cnf', './input/Benchmark_original/countdump10.cnf', './input/Benchmark_original/countdump9.cnf', './input/Benchmark_original/s510_7_4.cnf', './input/Benchmark_original/par16-1-c.cnf', './input/Benchmark_original/par16-4-c.cnf', './input/Benchmark_original/blasted_case10.cnf', './input/Benchmark_original/blasted_case6.cnf', './input/Benchmark_original/par16-3-c.cnf', './input/Benchmark_original/blasted_case_0_b11_1.cnf', './input/Benchmark_original/blasted_case_1_b11_1.cnf', './input/Benchmark_original/s510_15_7.cnf', './input/Benchmark_original/par16-5-c.cnf', './input/Benchmark_original/qg1-07.cnf', './input/Benchmark_original/qg2-07.cnf', './input/Benchmark_original/par16-2-c.cnf', './input/Benchmark_original/par8-1.cnf', './input/Benchmark_original/par8-3.cnf', './input/Benchmark_original/par8-4.cnf', './input/Benchmark_original/par8-5.cnf', './input/Benchmark_original/s382_15_7.cnf', './input/Benchmark_original/par8-2.cnf', './input/Benchmark_original/s420_new_15_7.cnf', './input/Benchmark_original/tire-1.cnf', './input/Benchmark_original/s526_3_2.cnf', './input/Benchmark_original/s420_15_7.cnf', './input/Benchmark_original/s420_new1_15_7.cnf', './input/Benchmark_original/s526a_3_2.cnf', './input/Benchmark_original/registerlesSwap.sk_3_10.cnf', './input/Benchmark_original/nocountdump19.cnf', './input/Benchmark_original/s444_15_7.cnf', './input/Benchmark_original/s526_7_4.cnf', './input/Benchmark_original/s526a_7_4.cnf', './input/Benchmark_original/blasted_case125.cnf', './input/Benchmark_original/blasted_case19.cnf', './input/Benchmark_original/blasted_case20.cnf', './input/Benchmark_original/blasted_case35.cnf', './input/Benchmark_original/blasted_case34.cnf', './input/Benchmark_original/4blocksb.cnf', './input/Benchmark_original/c880.isc.cnf', './input/Benchmark_original/s953.bench.cnf', './input/Benchmark_original/sat-grid-pbl-0020.cnf', './input/Benchmark_original/blasted_case143.cnf', './input/Benchmark_original/blasted_case_0_b12_1.cnf', './input/Benchmark_original/blasted_case_1_b12_1.cnf', './input/Benchmark_original/blasted_case_2_b12_1.cnf', './input/Benchmark_original/blasted_case114.cnf', './input/Benchmark_original/blasted_case115.cnf', './input/Benchmark_original/blasted_case131.cnf', './input/Benchmark_original/s641.bench.cnf', './input/Benchmark_original/blasted_case116.cnf', './input/Benchmark_original/s713.bench.cnf', './input/Benchmark_original/s526_15_7.cnf', './input/Benchmark_original/s526a_15_7.cnf', './input/Benchmark_original/ls9-normalized.cnf', './input/Benchmark_original/bw_large.a.cnf', './input/Benchmark_original/huge.cnf', './input/Benchmark_original/blasted_case140.cnf', './input/Benchmark_original/s641_3_2.cnf', './input/Benchmark_original/blasted_squaring51.cnf', './input/Benchmark_original/5_100_sd_schur.cnf', './input/Benchmark_original/blasted_squaring50.cnf', './input/Benchmark_original/nocountdump20.cnf', './input/Benchmark_original/nocountdump12.cnf', './input/Benchmark_original/nocountdump13.cnf', './input/Benchmark_original/s641_7_4.cnf', './input/Benchmark_original/s713_3_2.cnf', './input/Benchmark_original/qg1-08.cnf', './input/Benchmark_original/qg2-08.cnf', './input/Benchmark_original/qg3-08.cnf', './input/Benchmark_original/s838.1.bench.cnf', './input/Benchmark_original/s953a_3_2.cnf', './input/Benchmark_original/s713_7_4.cnf', './input/Benchmark_original/s953a_7_4.cnf', './input/Benchmark_original/s1238.bench.cnf', './input/Benchmark_original/tire-2.cnf', './input/Benchmark_original/c1355.isc.cnf', './input/Benchmark_original/s1196.bench.cnf', './input/Benchmark_original/nocountdump18.cnf', './input/Benchmark_original/lang12.cnf', './input/Benchmark_original/s641_15_7.cnf', './input/Benchmark_original/tire-3.cnf', './input/Benchmark_original/blasted_case18.cnf', './input/Benchmark_original/10random.cnf', './input/Benchmark_original/s713_15_7.cnf', './input/Benchmark_original/s820a_3_2.cnf', './input/Benchmark_original/s838_3_2.cnf', './input/Benchmark_original/s953a_15_7.cnf', './input/Benchmark_original/s832a_3_2.cnf', './input/Benchmark_original/s820a_7_4.cnf', './input/Benchmark_original/s838_7_4.cnf', './input/Benchmark_original/blasted_case107.cnf', './input/Benchmark_original/s832a_7_4.cnf', './input/Benchmark_original/sum.32.cnf', './input/Benchmark_original/blasted_case130.cnf', './input/Benchmark_original/blasted_case214.cnf', './input/Benchmark_original/blasted_case213.cnf', './input/Benchmark_original/2bitadd_11.cnf', './input/Benchmark_original/sat-grid-pbl-0025.cnf', './input/Benchmark_original/ls10-normalized.cnf', './input/Benchmark_original/nocountdump11.cnf', './input/Benchmark_original/s1494.bench.cnf', './input/Benchmark_original/s1488.bench.cnf', './input/Benchmark_original/s820a_15_7.cnf', './input/Benchmark_original/s838_15_7.cnf', './input/Benchmark_original/s1238a_3_2.cnf', './input/Benchmark_original/s1196a_3_2.cnf', './input/Benchmark_original/s832a_15_7.cnf', './input/Benchmark_original/blasted_squaring24.cnf', './input/Benchmark_original/blasted_squaring22.cnf', './input/Benchmark_original/blasted_squaring20.cnf', './input/Benchmark_original/blasted_squaring21.cnf', './input/Benchmark_original/5_140_sd_schur.cnf', './input/Benchmark_original/s1238a_7_4.cnf', './input/Benchmark_original/s1196a_7_4.cnf', './input/Benchmark_original/2bitadd_12.cnf', './input/Benchmark_original/blasted_squaring23.cnf', './input/Benchmark_original/GuidanceService2.sk_2_27.cnf', './input/Benchmark_original/hanoi4.cnf', './input/Benchmark_original/nocountdump6.cnf', './input/Benchmark_original/qg4-09.cnf', './input/Benchmark_original/qg6-09.cnf', './input/Benchmark_original/qg7-09.cnf', './input/Benchmark_original/nocountdump1.cnf', './input/Benchmark_original/nocountdump3.cnf', './input/Benchmark_original/blasted_case12.cnf', './input/Benchmark_original/nocountdump22.cnf', './input/Benchmark_original/nocountdump23.cnf', './input/Benchmark_original/nocountdump21.cnf', './input/Benchmark_original/s1423.bench.cnf', './input/Benchmark_original/c1908.isc.cnf', './input/Benchmark_original/4blocks.cnf', './input/Benchmark_original/cnt06.shuffled.cnf', './input/Benchmark_original/blasted_case144.cnf', './input/Benchmark_original/s1238a_15_7.cnf', './input/Benchmark_original/s1196a_15_7.cnf', './input/Benchmark_original/s1423a_3_2.cnf', './input/Benchmark_original/nocountdump17.cnf', './input/Benchmark_original/s1423a_7_4.cnf', './input/Benchmark_original/scenarios_tree_delete3.sb.pl.sk_2_32.cnf', './input/Benchmark_original/blasted_case_0_b14_1.cnf', './input/Benchmark_original/tire-4.cnf', './input/Benchmark_original/blasted_case207.cnf', './input/Benchmark_original/blasted_case208.cnf', './input/Benchmark_original/blasted_case_0_b12_2.cnf', './input/Benchmark_original/blasted_case_1_b12_2.cnf', './input/Benchmark_original/blasted_case_2_b12_2.cnf', './input/Benchmark_original/logistics.a.cnf', './input/Benchmark_original/blasted_squaring27.cnf', './input/Benchmark_original/blasted_case50.cnf', './input/Benchmark_original/logistics.b.cnf', './input/Benchmark_original/blasted_case139.cnf', './input/Benchmark_original/blasted_squaring25.cnf', './input/Benchmark_original/blasted_case138.cnf', './input/Benchmark_original/s1488_3_2.cnf', './input/Benchmark_original/nocountdump9.cnf', './input/Benchmark_original/s1423a_15_7.cnf', './input/Benchmark_original/blasted_case211.cnf', './input/Benchmark_original/blasted_case210.cnf', './input/Benchmark_original/s1488_7_4.cnf', './input/Benchmark_original/blasted_squaring70.cnf', './input/Benchmark_original/blasted_squaring2.cnf', './input/Benchmark_original/blasted_squaring3.cnf', './input/Benchmark_original/blasted_squaring5.cnf', './input/Benchmark_original/blasted_squaring6.cnf', './input/Benchmark_original/blasted_squaring1.cnf', './input/Benchmark_original/blasted_squaring4.cnf', './input/Benchmark_original/blasted_squaring26.cnf', './input/Benchmark_original/blasted_case42.cnf', './input/Benchmark_original/ls11-normalized.cnf', './input/Benchmark_original/sat-grid-pbl-0030.cnf', './input/Benchmark_original/log-1.cnf', './input/Benchmark_original/prob001.pddl.cnf', './input/Benchmark_original/s1488_15_7.cnf', './input/Benchmark_original/blasted_case_2_ptb_1.cnf', './input/Benchmark_original/blasted_case_1_ptb_1.cnf', './input/Benchmark_original/blasted_squaring11.cnf', './input/Benchmark_original/GuidanceService.sk_4_27.cnf', './input/Benchmark_original/nocountdump10.cnf', './input/Benchmark_original/par16-1.cnf', './input/Benchmark_original/par16-3.cnf', './input/Benchmark_original/par16-4.cnf', './input/Benchmark_original/par16-5.cnf', './input/Benchmark_original/par16-2.cnf', './input/Benchmark_original/lang15.cnf', './input/Benchmark_original/lang16.cnf', './input/Benchmark_original/tableBasedAddition.sk_240_1024.cnf', './input/Benchmark_original/blasted_squaring30.cnf', './input/Benchmark_original/blasted_squaring28.cnf', './input/Benchmark_original/blasted_case37.cnf', './input/Benchmark_original/bw_large.b.cnf', './input/Benchmark_original/blasted_squaring10.cnf', './input/Benchmark_original/blasted_squaring8.cnf', './input/Benchmark_original/blasted_squaring29.cnf', './input/Benchmark_original/logistics.c.cnf', './input/Benchmark_original/blasted_case209.cnf', './input/Benchmark_original/blasted_case212.cnf', './input/Benchmark_original/ls12-normalized.cnf', './input/Benchmark_original/c2670.isc.cnf', './input/Benchmark_original/ra.cnf', './input/Benchmark_original/nocountdump7.cnf', './input/Benchmark_original/blasted_TR_device_1_linear.cnf', './input/Benchmark_original/nocountdump2.cnf', './input/Benchmark_original/nocountdump4.cnf', './input/Benchmark_original/nocountdump8.cnf', './input/Benchmark_original/blasted_TR_b14_1_linear.cnf', './input/Benchmark_original/par32-2-c.cnf', './input/Benchmark_original/blasted_case3_b14_even3.cnf', './input/Benchmark_original/blasted_case_1_b14_even.cnf', './input/Benchmark_original/blasted_case_2_b14_even.cnf', './input/Benchmark_original/par32-1-c.cnf', './input/Benchmark_original/blasted_case1_b14_even3.cnf', './input/Benchmark_original/par32-3-c.cnf', './input/Benchmark_original/qg5-11.cnf', './input/Benchmark_original/par32-4-c.cnf', './input/Benchmark_original/log-2.cnf', './input/Benchmark_original/prob002.pddl.cnf', './input/Benchmark_original/par32-5-c.cnf', './input/Benchmark_original/ssa7552-158.cnf', './input/Benchmark_original/ssa7552-159.cnf', './input/Benchmark_original/nocountdump31.cnf', './input/Benchmark_original/ssa7552-160.cnf', './input/Benchmark_original/IssueServiceImpl.sk_8_30.cnf', './input/Benchmark_original/D1119_M20.cnf', './input/Benchmark_original/blockmap_05_01.net.cnf', './input/Benchmark_original/C169_FV.cnf', './input/Benchmark_original/C169_FW.cnf', './input/Benchmark_original/log-3.cnf', './input/Benchmark_original/prob003.pddl.cnf', './input/Benchmark_original/D1119_M23.cnf', './input/Benchmark_original/blasted_squaring9.cnf', './input/Benchmark_original/lang19.cnf', './input/Benchmark_original/blasted_squaring14.cnf', './input/Benchmark_original/C250_FV.cnf', './input/Benchmark_original/C250_FW.cnf', './input/Benchmark_original/10.sk_1_46.cnf', './input/Benchmark_original/ssa7552-038.cnf', './input/Benchmark_original/blasted_case_0_ptb_1.cnf', './input/Benchmark_original/blasted_squaring12.cnf', './input/Benchmark_original/UserServiceImpl.sk_8_32.cnf', './input/Benchmark_original/27.sk_3_32.cnf', './input/Benchmark_original/blasted_case49.cnf', './input/Benchmark_original/countdump6.cnf', './input/Benchmark_original/nocountdump25.cnf', './input/Benchmark_original/blasted_case_1_4_b14_even.cnf', './input/Benchmark_original/blasted_case_3_4_b14_even.cnf', './input/Benchmark_original/nocountdump24.cnf', './input/Benchmark_original/blasted_TR_b14_2_linear.cnf', './input/Benchmark_original/ls13-normalized.cnf', './input/Benchmark_original/lang20.cnf', './input/Benchmark_original/blasted_squaring16.cnf', './input/Benchmark_original/blasted_squaring7.cnf', './input/Benchmark_original/C211_FS.cnf', './input/Benchmark_original/C211_FW.cnf', './input/Benchmark_original/PhaseService.sk_14_27.cnf', './input/Benchmark_original/blockmap_05_02.net.cnf', './input/Benchmark_original/C171_FR.cnf', './input/Benchmark_original/C638_FVK.cnf', './input/Benchmark_original/C638_FKB.cnf', './input/Benchmark_original/cnt07.shuffled.cnf', './input/Benchmark_original/C638_FKA.cnf', './input/Benchmark_original/prob004-log-a.cnf', './input/Benchmark_original/C230_FR.cnf', './input/Benchmark_original/C163_FW.cnf', './input/Benchmark_original/C140_FC.cnf', './input/Benchmark_original/ActivityService.sk_11_27.cnf', './input/Benchmark_original/C140_FW.cnf', './input/Benchmark_original/C140_FV.cnf', './input/Benchmark_original/C215_FC.cnf', './input/Benchmark_original/rb.cnf', './input/Benchmark_original/C170_FR.cnf', './input/Benchmark_original/C208_FA.cnf', './input/Benchmark_original/C209_FA.cnf', './input/Benchmark_original/C129_FR.cnf', './input/Benchmark_original/IterationService.sk_12_27.cnf', './input/Benchmark_original/C168_FW.cnf', './input/Benchmark_original/blasted_TR_b12_1_linear.cnf', './input/Benchmark_original/C203_FCL.cnf', './input/Benchmark_original/C208_FC.cnf', './input/Benchmark_original/C209_FC.cnf', './input/Benchmark_original/C203_FS.cnf', './input/Benchmark_original/hanoi5.cnf', './input/Benchmark_original/C220_FW.cnf', './input/Benchmark_original/C220_FV.cnf', './input/Benchmark_original/C210_FVF.cnf', './input/Benchmark_original/blasted_TR_b14_3_linear.cnf', './input/Benchmark_original/ActivityService2.sk_10_27.cnf', './input/Benchmark_original/blasted_TR_ptb_1_linear.cnf', './input/Benchmark_original/C202_FS.cnf', './input/Benchmark_original/C210_FS.cnf', './input/Benchmark_original/binsearch.16.cnf', './input/Benchmark_original/C210_FW.cnf', './input/Benchmark_original/C203_FW.cnf', './input/Benchmark_original/C202_FW.cnf', './input/Benchmark_original/ls14-normalized.cnf', './input/Benchmark_original/blockmap_05_03.net.cnf', './input/Benchmark_original/lang23.cnf', './input/Benchmark_original/qg7-13.cnf', './input/Benchmark_original/scenarios_aig_insertion1.sb.pl.sk_3_60.cnf', './input/Benchmark_original/log-4.cnf', './input/Benchmark_original/prob004.pddl.cnf', './input/Benchmark_original/lang24.cnf', './input/Benchmark_original/prob012.pddl.cnf', './input/Benchmark_original/111.sk_2_36.cnf', './input/Benchmark_original/blasted_TR_b12_2_linear.cnf', './input/Benchmark_original/blasted_TR_device_1_even_linear.cnf', './input/Benchmark_original/blasted_case142.cnf', './input/Benchmark_original/rc.cnf', './input/Benchmark_original/ConcreteActivityService.sk_13_28.cnf', './input/Benchmark_original/countdump8.cnf', './input/Benchmark_original/ls15-normalized.cnf', './input/Benchmark_original/53.sk_4_32.cnf', './input/Benchmark_original/scenarios_aig_insertion2.sb.pl.sk_3_60.cnf', './input/Benchmark_original/blasted_case_0_b12_even2.cnf', './input/Benchmark_original/blasted_case_1_b12_even2.cnf', './input/Benchmark_original/blasted_case_2_b12_even2.cnf', './input/Benchmark_original/blasted_case_0_b12_even1.cnf', './input/Benchmark_original/blasted_case_1_b12_even1.cnf', './input/Benchmark_original/blasted_case_2_b12_even1.cnf', './input/Benchmark_original/log-5.cnf', './input/Benchmark_original/prob005.pddl.cnf', './input/Benchmark_original/bmc-ibm-2.cnf', './input/Benchmark_original/blasted_case_2_ptb_2.cnf', './input/Benchmark_original/blasted_case_1_ptb_2.cnf', './input/Benchmark_original/lang27.cnf', './input/Benchmark_original/nocountdump16.cnf', './input/Benchmark_original/bw_large.c.cnf', './input/Benchmark_original/55.sk_3_46.cnf', './input/Benchmark_original/lang28.cnf', './input/Benchmark_original/ls16-normalized.cnf', './input/Benchmark_original/ProjectService3.sk_12_55.cnf', './input/Benchmark_original/par32-1.cnf', './input/Benchmark_original/par32-2.cnf', './input/Benchmark_original/par32-3.cnf', './input/Benchmark_original/par32-4.cnf', './input/Benchmark_original/par32-5.cnf', './input/Benchmark_original/c7552.isc.cnf', './input/Benchmark_original/blasted_case_0_ptb_2.cnf', './input/Benchmark_original/mastermind_05_08_03.net.cnf', './input/Benchmark_original/NotificationServiceImpl2.sk_10_36.cnf', './input/Benchmark_original/109.sk_4_36.cnf', './input/Benchmark_original/blasted_case104.cnf', './input/Benchmark_original/s5378a_3_2.cnf', './input/Benchmark_original/s5378a_7_4.cnf', './input/Benchmark_original/51.sk_4_38.cnf', './input/Benchmark_original/fs-07.net.cnf', './input/Benchmark_original/s5378a_15_7.cnf', './input/Benchmark_original/32.sk_4_38.cnf', './input/Benchmark_original/blasted_TR_ptb_2_linear.cnf', './input/Benchmark_original/mastermind_06_08_03.net.cnf', './input/Benchmark_original/alu2_gr_rcs_w8.shuffled.cnf', './input/Benchmark_original/cnt08.shuffled.cnf', './input/Benchmark_original/blasted_case141.cnf', './input/Benchmark_original/blasted_case_0_b12_even3.cnf', './input/Benchmark_original/blasted_case_1_b12_even3.cnf', './input/Benchmark_original/blasted_case_2_b12_even3.cnf', './input/Benchmark_original/blasted_squaring40.cnf', './input/Benchmark_original/blasted_squaring42.cnf', './input/Benchmark_original/blasted_squaring41.cnf', './input/Benchmark_original/binsearch.32.cnf', './input/Benchmark_original/c880_gr_rcs_w7.shuffled.cnf', './input/Benchmark_original/70.sk_3_40.cnf', './input/Benchmark_original/logistics.d.cnf', './input/Benchmark_original/mastermind_03_08_04.net.cnf', './input/Benchmark_original/ProcessBean.sk_8_64.cnf', './input/Benchmark_original/56.sk_6_38.cnf', './input/Benchmark_original/35.sk_3_52.cnf', './input/Benchmark_original/80.sk_2_48.cnf', './input/Benchmark_original/countdump4.cnf', './input/Benchmark_original/blasted_squaring60.cnf', './input/Benchmark_original/countdump3.cnf', './input/Benchmark_original/countdump5.cnf', './input/Benchmark_original/mastermind_04_08_04.net.cnf', './input/Benchmark_original/71.sk_3_65.cnf', './input/Benchmark_original/79.sk_4_40.cnf', './input/Benchmark_original/scenarios_tree_delete.sb.pl.sk_3_30.cnf', './input/Benchmark_original/scenarios_tree_insert_insert.sb.pl.sk_3_68.cnf', './input/Benchmark_original/mastermind_10_08_03.net.cnf', './input/Benchmark_original/s9234a_3_2.cnf', './input/Benchmark_original/s9234a_7_4.cnf', './input/Benchmark_original/bw_large.d.cnf', './input/Benchmark_original/s9234a_15_7.cnf', './input/Benchmark_original/7.sk_4_50.cnf', './input/Benchmark_original/doublyLinkedList.sk_8_37.cnf', './input/Benchmark_original/57.sk_4_64.cnf', './input/Benchmark_original/19.sk_3_48.cnf', './input/Benchmark_original/63.sk_3_64.cnf', './input/Benchmark_original/mastermind_03_08_05.net.cnf', './input/Benchmark_original/Pollard.sk_1_10.cnf', './input/Benchmark_original/36.sk_3_77.cnf', './input/Benchmark_original/LoginService.sk_20_34.cnf', './input/Benchmark_original/3bitadd_31.cnf', './input/Benchmark_original/blasted_TR_b12_even2_linear.cnf', './input/Benchmark_original/blasted_TR_b12_even3_linear.cnf', './input/Benchmark_original/blasted_TR_b12_even7_linear.cnf', './input/Benchmark_original/3bitadd_32.cnf', './input/Benchmark_original/bmc-ibm-7.cnf', './input/Benchmark_original/blasted_TR_b14_even_linear.cnf', './input/Benchmark_original/blasted_TR_b14_even3_linear.cnf', './input/Benchmark_original/29.sk_3_45.cnf', './input/Benchmark_original/107.sk_3_90.cnf', './input/Benchmark_original/cnt09.shuffled.cnf', './input/Benchmark_original/s13207a_3_2.cnf', './input/Benchmark_original/s13207a_7_4.cnf', './input/Benchmark_original/bmc-ibm-5.cnf', './input/Benchmark_original/s13207a_15_7.cnf', './input/Benchmark_original/bmc-ibm-1.cnf', './input/Benchmark_original/isolateRightmost.sk_7_481.cnf', './input/Benchmark_original/17.sk_3_45.cnf', './input/Benchmark_original/fs-10.net.cnf', './input/Benchmark_original/blasted_TR_b14_even2_linear.cnf', './input/Benchmark_original/81.sk_5_51.cnf', './input/Benchmark_original/s15850a_3_2.cnf', './input/Benchmark_original/s15850a_7_4.cnf', './input/Benchmark_original/s15850a_15_7.cnf', './input/Benchmark_original/blockmap_10_01.net.cnf', './input/Benchmark_original/LoginService2.sk_23_36.cnf', './input/Benchmark_original/sort.sk_8_52.cnf', './input/Benchmark_original/scenarios_tree_delete4.sb.pl.sk_4_114.cnf', './input/Benchmark_original/blockmap_10_02.net.cnf', './input/Benchmark_original/parity.sk_11_11.cnf', './input/Benchmark_original/bmc-ibm-13.cnf', './input/Benchmark_original/blockmap_10_03.net.cnf', './input/Benchmark_original/77.sk_3_44.cnf', './input/Benchmark_original/bmc-ibm-3.cnf', './input/Benchmark_original/20.sk_1_51.cnf', './input/Benchmark_original/scenarios_tree_delete2.sb.pl.sk_8_114.cnf', './input/Benchmark_original/84.sk_4_77.cnf', './input/Benchmark_original/enqueueSeqSK.sk_10_42.cnf', './input/Benchmark_original/tutorial2.sk_3_4.cnf', './input/Benchmark_original/s35932_3_2.cnf', './input/Benchmark_original/s35932_7_4.cnf', './input/Benchmark_original/s35932_15_7.cnf', './input/Benchmark_original/110.sk_3_88.cnf', './input/Benchmark_original/logcount.sk_16_86.cnf', './input/Benchmark_original/karatsuba.sk_7_41.cnf', './input/Benchmark_original/cnt10.shuffled.cnf', './input/Benchmark_original/54.sk_12_97.cnf', './input/Benchmark_original/fs-13.net.cnf', './input/Benchmark_original/s38584_3_2.cnf', './input/Benchmark_original/s38584_7_4.cnf', './input/Benchmark_original/s38584_15_7.cnf', './input/Benchmark_original/scenarios_treemax.sb.pl.sk_7_19.cnf', './input/Benchmark_original/s38417_3_2.cnf', './input/Benchmark_original/s38417_7_4.cnf', './input/Benchmark_original/s38417_15_7.cnf', './input/Benchmark_original/bmc-ibm-4.cnf', './input/Benchmark_original/30.sk_5_76.cnf', './input/Benchmark_original/signedAvg.sk_8_1020.cnf', './input/Benchmark_original/scenarios_treemin.sb.pl.sk_9_19.cnf', './input/Benchmark_original/bmc-ibm-11.cnf', './input/Benchmark_original/blockmap_15_01.net.cnf', './input/Benchmark_original/SetTest.sk_9_21.cnf', './input/Benchmark_original/scenarios_tree_delete1.sb.pl.sk_3_114.cnf', './input/Benchmark_original/blockmap_15_02.net.cnf', './input/Benchmark_original/xpose.sk_6_134.cnf', './input/Benchmark_original/blockmap_15_03.net.cnf', './input/Benchmark_original/bmc-ibm-12.cnf', './input/Benchmark_original/fs-16.net.cnf', './input/Benchmark_original/scenarios_lltraversal.sb.pl.sk_5_23.cnf', './input/Benchmark_original/compress.sk_17_291.cnf', './input/Benchmark_original/bmc-ibm-6.cnf', './input/Benchmark_original/listReverse.sk_11_43.cnf', './input/Benchmark_original/bmc-galileo-8.cnf', './input/Benchmark_original/bmc-ibm-10.cnf', './input/Benchmark_original/bmc-galileo-9.cnf', './input/Benchmark_original/scenarios_llreverse.sb.pl.sk_8_25.cnf', './input/Benchmark_original/fs-19.net.cnf', './input/Benchmark_original/reverse.sk_11_258.cnf', './input/Benchmark_original/blockmap_20_01.net.cnf', './input/Benchmark_original/scenarios_tree_search.sb.pl.sk_11_136.cnf', './input/Benchmark_original/scenarios_tree_insert_search.sb.pl.sk_11_136.cnf', './input/Benchmark_original/scenarios_aig_traverse.sb.pl.sk_5_102.cnf', './input/Benchmark_original/lss.sk_6_7.cnf', './input/Benchmark_original/blockmap_20_02.net.cnf', './input/Benchmark_original/blockmap_20_03.net.cnf', './input/Benchmark_original/jburnim_morton.sk_13_530.cnf', './input/Benchmark_original/fs-22.net.cnf', './input/Benchmark_original/blockmap_22_01.net.cnf', './input/Benchmark_original/blockmap_22_02.net.cnf', './input/Benchmark_original/scenarios_llinsert2.sb.pl.sk_6_407.cnf', './input/Benchmark_original/fs-25.net.cnf', './input/Benchmark_original/partition.sk_22_155.cnf', './input/Benchmark_original/log2.sk_72_391.cnf', './input/Benchmark_original/scenarios_lldelete1.sb.pl.sk_6_409.cnf', './input/Benchmark_original/fs-28.net.cnf', './input/Benchmark_original/fs-29.net.cnf', './input/Benchmark_original/diagStencil.sk_35_36.cnf', './input/Benchmark_original/diagStencilClean.sk_41_36.cnf', './input/Benchmark_original/ConcreteRoleAffectationService.sk_119_273.cnf', './input/Benchmark_original/tutorial3.sk_4_31.cnf']
    ordered_preproc_instances_filtered = ['./input/Benchmark_preproc2/C169_FV.cnf', './input/Benchmark_preproc2/C169_FW.cnf', './input/Benchmark_preproc2/or-100-5-1-UC-40.cnf',
                                          './input/Benchmark_preproc2/or-60-10-2-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-6-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-2-UC-30.cnf',
                                          './input/Benchmark_preproc2/or-60-20-9-UC-40.cnf', './input/Benchmark_preproc2/or-70-5-5-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-9-UC-40.cnf',
                                          './input/Benchmark_preproc2/or-100-5-10-UC-40.cnf', './input/Benchmark_preproc2/or-50-5-4-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-7-UC-30.cnf',
                                          './input/Benchmark_preproc2/or-50-10-9-UC-40.cnf', './input/Benchmark_preproc2/or-60-5-7-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-10-UC-40.cnf',
                                          './input/Benchmark_preproc2/or-60-5-9-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-3-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-6-UC-50.cnf',
                                          './input/Benchmark_preproc2/or-100-20-9-UC-50.cnf', './input/Benchmark_preproc2/or-100-10-1-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-8-UC-20.cnf',
                                          './input/Benchmark_preproc2/or-50-20-3-UC-30.cnf', './input/Benchmark_preproc2/or-50-5-9-UC-30.cnf', './input/Benchmark_preproc2/or-60-20-9-UC-30.cnf',
                                          './input/Benchmark_preproc2/or-70-5-3-UC-40.cnf', './input/Benchmark_preproc2/or-100-5-9-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-4-UC-40.cnf',
                                          './input/Benchmark_preproc2/or-50-20-8-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-2-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-2-UC-40.cnf',
                                          './input/Benchmark_preproc2/or-50-10-2-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-7-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-2-UC-40.cnf',

                                          './input/Benchmark_preproc2/or-100-10-3-UC-50.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-60.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-60.cnf', './input/Benchmark_preproc2/or-100-20-3-UC-50.cnf', './input/Benchmark_preproc2/or-50-5-6-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-3-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-7-UC-40.cnf', './input/Benchmark_preproc2/or-70-20-4-UC-40.cnf', './input/Benchmark_preproc2/or-70-5-9-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-4-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-10-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-3-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-10-UC-20.cnf', './input/Benchmark_preproc2/blasted_case134.cnf', './input/Benchmark_preproc2/blasted_case137.cnf', './input/Benchmark_preproc2/or-50-10-5-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-9-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-5-UC-40.cnf', './input/Benchmark_preproc2/or-100-10-7-UC-50.cnf', './input/Benchmark_preproc2/or-100-5-7-UC-50.cnf', './input/Benchmark_preproc2/or-60-10-4-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-7-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-10-UC-40.cnf', './input/Benchmark_preproc2/or-70-10-9-UC-40.cnf', './input/Benchmark_preproc2/or-70-5-8-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-8-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-2-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-7-UC-40.cnf', './input/Benchmark_preproc2/blasted_case36.cnf', './input/Benchmark_preproc2/or-50-20-1-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-8-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-8-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-6-UC-40.cnf', './input/Benchmark_preproc2/or-50-20-6-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-3-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-60.cnf', './input/Benchmark_preproc2/ais6.cnf', './input/Benchmark_preproc2/or-50-20-4-UC-40.cnf', './input/Benchmark_preproc2/or-50-20-7-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-7-UC-30.cnf', './input/Benchmark_preproc2/or-60-20-1-UC-40.cnf', './input/Benchmark_preproc2/or-70-10-6-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-6-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-1-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-1-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-2-UC-50.cnf', './input/Benchmark_preproc2/blasted_case29.cnf', './input/Benchmark_preproc2/blasted_case24.cnf', './input/Benchmark_preproc2/countdump1.cnf', './input/Benchmark_preproc2/countdump10.cnf', './input/Benchmark_preproc2/countdump2.cnf', './input/Benchmark_preproc2/countdump9.cnf', './input/Benchmark_preproc2/or-100-10-10-UC-50.cnf', './input/Benchmark_preproc2/or-50-5-10-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-5-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-1-UC-40.cnf', './input/Benchmark_preproc2/or-100-5-8-UC-40.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-60.cnf', './input/Benchmark_preproc2/or-50-20-10-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-2-UC-40.cnf', './input/Benchmark_preproc2/or-50-20-3-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-60.cnf', './input/Benchmark_preproc2/or-50-5-7-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-4-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-5-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-9-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-5-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-3-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-6-UC-30.cnf', './input/Benchmark_preproc2/par8-1-c.cnf', './input/Benchmark_preproc2/blasted_case30.cnf', './input/Benchmark_preproc2/blasted_case25.cnf', './input/Benchmark_preproc2/or-100-10-2-UC-50.cnf', './input/Benchmark_preproc2/or-50-10-6-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-10-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-6-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-7-UC-40.cnf', './input/Benchmark_preproc2/or-70-20-9-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-50.cnf', './input/Benchmark_preproc2/or-50-5-3-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-6-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-5-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-8-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-50.cnf', './input/Benchmark_preproc2/or-70-5-7-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-6-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-7-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-1-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-10-UC-40.cnf', './input/Benchmark_preproc2/or-60-5-9-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-6-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-10-UC-40.cnf', './input/Benchmark_preproc2/or-60-5-5-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-1-UC-30.cnf', './input/Benchmark_preproc2/par8-4-c.cnf', './input/Benchmark_preproc2/or-100-10-10-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-1-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-1-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-3-UC-50.cnf', './input/Benchmark_preproc2/par8-2-c.cnf', './input/Benchmark_preproc2/blasted_case100.cnf', './input/Benchmark_preproc2/blasted_case101.cnf', './input/Benchmark_preproc2/or-50-5-6-UC-10.cnf', './input/Benchmark_preproc2/or-50-5-8-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-3-UC-40.cnf', './input/Benchmark_preproc2/or-70-10-10-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-60.cnf', './input/Benchmark_preproc2/blasted_case17.cnf', './input/Benchmark_preproc2/blasted_case23.cnf', './input/Benchmark_preproc2/or-50-10-7-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-9-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-9-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-2-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-1-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-10-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-1-UC-30.cnf', './input/Benchmark_preproc2/or-60-20-3-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-1-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-60.cnf', './input/Benchmark_preproc2/or-50-20-6-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-8-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-7-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-6-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-7-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-6-UC-40.cnf', './input/Benchmark_preproc2/or-70-10-8-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-9-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-4-UC-20.cnf', './input/Benchmark_preproc2/or-50-10-5-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-2-UC-30.cnf', './input/Benchmark_preproc2/or-50-5-9-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-5-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-8-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-10-UC-40.cnf', './input/Benchmark_preproc2/or-100-5-7-UC-40.cnf', './input/Benchmark_preproc2/or-100-10-3-UC-40.cnf', './input/Benchmark_preproc2/or-50-5-5-UC-20.cnf', './input/Benchmark_preproc2/or-50-10-8-UC-10.cnf', './input/Benchmark_preproc2/or-50-20-4-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-2-UC-40.cnf', './input/Benchmark_preproc2/or-70-20-3-UC-40.cnf', './input/Benchmark_preproc2/or-70-20-4-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-6-UC-40.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-60.cnf', './input/Benchmark_preproc2/or-60-5-2-UC-30.cnf', './input/Benchmark_preproc2/or-60-20-5-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-2-UC-10.cnf', './input/Benchmark_preproc2/or-50-10-3-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-1-UC-40.cnf', './input/Benchmark_preproc2/or-100-20-5-UC-50.cnf', './input/Benchmark_preproc2/par8-3-c.cnf', './input/Benchmark_preproc2/par8-5-c.cnf', './input/Benchmark_preproc2/or-50-5-10-UC-10.cnf', './input/Benchmark_preproc2/or-50-10-10-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-5-UC-50.cnf', './input/Benchmark_preproc2/or-50-10-5-UC-20.cnf', './input/Benchmark_preproc2/or-50-10-9-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-4-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-5-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-1-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-1-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-4-UC-40.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-50.cnf', './input/Benchmark_preproc2/or-100-20-9-UC-40.cnf', './input/Benchmark_preproc2/par16-3.cnf', './input/Benchmark_preproc2/or-50-20-8-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-3-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-7-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-7-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-9-UC-20.cnf', './input/Benchmark_preproc2/par16-1.cnf', './input/Benchmark_preproc2/or-60-20-3-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-4-UC-40.cnf', './input/Benchmark_preproc2/par16-4.cnf', './input/Benchmark_preproc2/or-50-10-6-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-6-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-10-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-8-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-4-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-3-UC-10.cnf', './input/Benchmark_preproc2/or-50-10-5-UC-10.cnf', './input/Benchmark_preproc2/or-50-5-9-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-10-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-6-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-3-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-5-UC-40.cnf', './input/Benchmark_preproc2/or-100-5-2-UC-40.cnf', './input/Benchmark_preproc2/par16-5.cnf', './input/Benchmark_preproc2/or-50-10-4-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-9-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-3-UC-40.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-40.cnf', './input/Benchmark_preproc2/par16-2.cnf', './input/Benchmark_preproc2/or-50-20-4-UC-10.cnf', './input/Benchmark_preproc2/or-50-20-7-UC-10.cnf', './input/Benchmark_preproc2/or-50-5-8-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-2-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-2-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-8-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-7-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-4-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-50.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-50.cnf', './input/Benchmark_preproc2/or-50-5-4-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-5-UC-10.cnf', './input/Benchmark_preproc2/or-50-20-9-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-1-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-10-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-8-UC-40.cnf', './input/Benchmark_preproc2/or-50-10-1-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-2-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-4-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-9-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-2-UC-10.cnf', './input/Benchmark_preproc2/or-50-10-9-UC-10.cnf', './input/Benchmark_preproc2/or-50-20-2-UC-20.cnf', './input/Benchmark_preproc2/or-50-20-6-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-4-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-3-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-2-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-10-UC-30.cnf', './input/Benchmark_preproc2/blasted_case59.cnf', './input/Benchmark_preproc2/blasted_case59_1.cnf', './input/Benchmark_preproc2/blasted_case64.cnf', './input/Benchmark_preproc2/or-50-5-7-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-7-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-10-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-2-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-1-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-10-UC-30.cnf', './input/Benchmark_preproc2/or-50-20-3-UC-10.cnf', './input/Benchmark_preproc2/or-50-5-1-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-5-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-1-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-40.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-50.cnf', './input/Benchmark_preproc2/or-50-20-1-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-10-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-9-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-9-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-4-UC-40.cnf', './input/Benchmark_preproc2/blasted_case58.cnf', './input/Benchmark_preproc2/blasted_case63.cnf', './input/Benchmark_preproc2/or-70-20-1-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-2-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-50.cnf', './input/Benchmark_preproc2/or-50-5-5-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-1-UC-20.cnf', './input/Benchmark_preproc2/or-50-10-7-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-3-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-3-UC-20.cnf', './input/Benchmark_preproc2/or-60-20-6-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-7-UC-40.cnf', './input/Benchmark_preproc2/or-50-5-4-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-3-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-6-UC-50.cnf', './input/Benchmark_preproc2/or-70-5-8-UC-20.cnf', './input/Benchmark_preproc2/blasted_case4.cnf', './input/Benchmark_preproc2/or-70-10-10-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-1-UC-20.cnf', './input/Benchmark_preproc2/or-60-10-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-50.cnf', './input/Benchmark_preproc2/2bitcomp_5.cnf', './input/Benchmark_preproc2/blasted_case21.cnf', './input/Benchmark_preproc2/blasted_case22.cnf', './input/Benchmark_preproc2/or-60-5-6-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-7-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-9-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-3-UC-30.cnf', './input/Benchmark_preproc2/countdump7.cnf', './input/Benchmark_preproc2/or-60-10-7-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-10-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-5-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-2-UC-20.cnf', './input/Benchmark_preproc2/or-50-5-6.cnf', './input/Benchmark_preproc2/or-50-20-5.cnf', './input/Benchmark_preproc2/or-60-10-4-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-6-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-3-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-8-UC-30.cnf', './input/Benchmark_preproc2/or-50-5-3.cnf', './input/Benchmark_preproc2/or-50-10-5.cnf', './input/Benchmark_preproc2/or-50-10-9.cnf', './input/Benchmark_preproc2/or-50-20-1.cnf', './input/Benchmark_preproc2/or-50-20-2-UC-10.cnf', './input/Benchmark_preproc2/or-50-20-9.cnf', './input/Benchmark_preproc2/or-60-5-5-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-2-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-6-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-1-UC-30.cnf', './input/Benchmark_preproc2/or-50-5-5.cnf', './input/Benchmark_preproc2/or-50-10-4.cnf', './input/Benchmark_preproc2/or-50-10-8.cnf', './input/Benchmark_preproc2/or-50-5-7.cnf', './input/Benchmark_preproc2/or-50-5-9.cnf', './input/Benchmark_preproc2/or-50-20-4.cnf', './input/Benchmark_preproc2/or-60-20-9-UC-10.cnf', './input/Benchmark_preproc2/or-70-5-5-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-6-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-7-UC-20.cnf', './input/Benchmark_preproc2/or-50-10-10.cnf', './input/Benchmark_preproc2/or-50-10-2.cnf', './input/Benchmark_preproc2/or-50-5-4.cnf', './input/Benchmark_preproc2/or-50-5-1.cnf', './input/Benchmark_preproc2/or-50-5-10.cnf', './input/Benchmark_preproc2/or-50-5-2.cnf', './input/Benchmark_preproc2/or-50-10-3.cnf', './input/Benchmark_preproc2/or-50-10-6.cnf', './input/Benchmark_preproc2/or-50-10-7.cnf', './input/Benchmark_preproc2/or-50-20-10.cnf', './input/Benchmark_preproc2/or-50-20-2.cnf', './input/Benchmark_preproc2/or-50-20-3.cnf', './input/Benchmark_preproc2/or-50-20-6.cnf', './input/Benchmark_preproc2/or-50-20-7.cnf', './input/Benchmark_preproc2/or-50-20-8.cnf', './input/Benchmark_preproc2/or-50-5-8.cnf', './input/Benchmark_preproc2/or-60-5-2-UC-10.cnf', './input/Benchmark_preproc2/or-70-5-10-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-10-UC-30.cnf', './input/Benchmark_preproc2/or-50-10-1.cnf', './input/Benchmark_preproc2/or-70-5-9-UC-10.cnf', './input/Benchmark_preproc2/blasted_case11.cnf', './input/Benchmark_preproc2/or-100-5-8-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-40.cnf', './input/Benchmark_preproc2/or-70-5-7-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-10-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-2-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-4-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-1-UC-10.cnf', './input/Benchmark_preproc2/or-60-5-8-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-4-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-7-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-7-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-5-UC-40.cnf', './input/Benchmark_preproc2/or-100-20-6-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-8-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-9-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-3-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-40.cnf', './input/Benchmark_preproc2/or-60-20-5-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-2-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-20-5-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-5-UC-40.cnf', './input/Benchmark_preproc2/or-60-10-6-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-9-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-6-UC-30.cnf', './input/Benchmark_preproc2/or-100-5-1-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-3-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-7-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-3-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-8-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-1-UC-10.cnf', './input/Benchmark_preproc2/or-70-5-8-UC-10.cnf', './input/Benchmark_preproc2/or-60-10-10-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-10-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-2-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-4-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-6-UC-10.cnf', './input/Benchmark_preproc2/or-70-5-5-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-3-UC-20.cnf', './input/Benchmark_preproc2/sat-grid-pbl-0010.cnf', './input/Benchmark_preproc2/or-60-10-3-UC-10.cnf', './input/Benchmark_preproc2/or-70-5-10-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-1-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-1-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-7-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-40.cnf', './input/Benchmark_preproc2/ais8.cnf', './input/Benchmark_preproc2/blasted_case43.cnf', './input/Benchmark_preproc2/blasted_case45.cnf', './input/Benchmark_preproc2/blasted_case7.cnf', './input/Benchmark_preproc2/or-70-20-5-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-6-UC-20.cnf', './input/Benchmark_preproc2/blasted_case47.cnf', './input/Benchmark_preproc2/or-70-10-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-2-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-4-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-1-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-2-UC-10.cnf', './input/Benchmark_preproc2/medium.cnf', './input/Benchmark_preproc2/or-60-5-4.cnf', './input/Benchmark_preproc2/or-70-20-10-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-3-UC-30.cnf', './input/Benchmark_preproc2/or-60-10-8.cnf', './input/Benchmark_preproc2/or-60-10-9.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-40.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-40.cnf', './input/Benchmark_preproc2/or-60-5-3.cnf', './input/Benchmark_preproc2/or-60-5-7.cnf', './input/Benchmark_preproc2/or-60-10-5.cnf', './input/Benchmark_preproc2/or-60-10-7.cnf', './input/Benchmark_preproc2/or-60-20-1.cnf', './input/Benchmark_preproc2/or-60-20-4.cnf', './input/Benchmark_preproc2/or-70-20-8-UC-20.cnf', './input/Benchmark_preproc2/or-60-5-8.cnf', './input/Benchmark_preproc2/or-100-5-2-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-7-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-3-UC-40.cnf', './input/Benchmark_preproc2/or-60-5-10.cnf', './input/Benchmark_preproc2/or-60-5-5.cnf', './input/Benchmark_preproc2/or-60-5-6.cnf', './input/Benchmark_preproc2/or-60-10-1.cnf', './input/Benchmark_preproc2/or-60-10-3.cnf', './input/Benchmark_preproc2/or-60-10-6.cnf', './input/Benchmark_preproc2/or-60-20-8.cnf', './input/Benchmark_preproc2/or-70-10-6-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-7-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-10-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-4-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-60.cnf', './input/Benchmark_preproc2/or-100-20-6-UC-30.cnf', './input/Benchmark_preproc2/or-60-5-2.cnf', './input/Benchmark_preproc2/or-60-10-10.cnf', './input/Benchmark_preproc2/or-60-10-2.cnf', './input/Benchmark_preproc2/or-60-10-4.cnf', './input/Benchmark_preproc2/or-60-20-10.cnf', './input/Benchmark_preproc2/or-60-20-3.cnf', './input/Benchmark_preproc2/or-60-20-5.cnf', './input/Benchmark_preproc2/or-60-20-6.cnf', './input/Benchmark_preproc2/or-60-20-7.cnf', './input/Benchmark_preproc2/or-60-20-9.cnf', './input/Benchmark_preproc2/or-60-5-9.cnf', './input/Benchmark_preproc2/or-70-10-10-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-5-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-6-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-7-UC-10.cnf', './input/Benchmark_preproc2/or-60-20-2.cnf', './input/Benchmark_preproc2/or-60-5-1.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-40.cnf', './input/Benchmark_preproc2/or-100-10-1-UC-20.cnf', './input/Benchmark_preproc2/or-70-5-4-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-5-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-10-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-2-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-6-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-30.cnf', './input/Benchmark_preproc2/nocountdump13.cnf', './input/Benchmark_preproc2/nocountdump16.cnf', './input/Benchmark_preproc2/nocountdump26.cnf', './input/Benchmark_preproc2/nocountdump27.cnf', './input/Benchmark_preproc2/nocountdump28.cnf', './input/Benchmark_preproc2/nocountdump29.cnf', './input/Benchmark_preproc2/nocountdump5.cnf', './input/Benchmark_preproc2/or-70-20-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-30.cnf', './input/Benchmark_preproc2/or-70-20-4-UC-10.cnf', './input/Benchmark_preproc2/or-70-20-8-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-30.cnf', './input/Benchmark_preproc2/or-70-10-2-UC-10.cnf', './input/Benchmark_preproc2/or-70-10-4-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-10-UC-20.cnf', './input/Benchmark_preproc2/blasted_case51.cnf', './input/Benchmark_preproc2/blasted_case52.cnf', './input/Benchmark_preproc2/blasted_case53.cnf', './input/Benchmark_preproc2/blasted_case124.cnf', './input/Benchmark_preproc2/C250_FV.cnf', './input/Benchmark_preproc2/C250_FW.cnf', './input/Benchmark_preproc2/or-70-20-3-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-9-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-50.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-9-UC-30.cnf', './input/Benchmark_preproc2/or-100-10-7-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-30.cnf', './input/Benchmark_preproc2/blasted_case112.cnf', './input/Benchmark_preproc2/or-100-5-7-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-8-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-2-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-8.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-20.cnf', './input/Benchmark_preproc2/or-70-10-5.cnf', './input/Benchmark_preproc2/or-70-5-3.cnf', './input/Benchmark_preproc2/or-70-10-6.cnf', './input/Benchmark_preproc2/or-70-5-1.cnf', './input/Benchmark_preproc2/or-70-5-10.cnf', './input/Benchmark_preproc2/or-70-5-2.cnf', './input/Benchmark_preproc2/or-70-20-2.cnf', './input/Benchmark_preproc2/or-70-20-6.cnf', './input/Benchmark_preproc2/or-70-20-7.cnf', './input/Benchmark_preproc2/or-70-20-8.cnf', './input/Benchmark_preproc2/or-70-5-8.cnf', './input/Benchmark_preproc2/or-70-5-9.cnf', './input/Benchmark_preproc2/or-70-10-1.cnf', './input/Benchmark_preproc2/or-70-5-5.cnf', './input/Benchmark_preproc2/or-70-10-7.cnf', './input/Benchmark_preproc2/or-70-10-9.cnf', './input/Benchmark_preproc2/or-70-20-1.cnf', './input/Benchmark_preproc2/or-70-20-10.cnf', './input/Benchmark_preproc2/or-70-20-3.cnf', './input/Benchmark_preproc2/or-70-20-4.cnf', './input/Benchmark_preproc2/or-70-20-5.cnf', './input/Benchmark_preproc2/or-70-10-3.cnf', './input/Benchmark_preproc2/or-70-5-6.cnf', './input/Benchmark_preproc2/or-100-20-3-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-5-UC-30.cnf', './input/Benchmark_preproc2/or-70-5-7.cnf', './input/Benchmark_preproc2/blasted_case38.cnf', './input/Benchmark_preproc2/or-70-10-10.cnf', './input/Benchmark_preproc2/or-70-10-2.cnf', './input/Benchmark_preproc2/or-70-5-4.cnf', './input/Benchmark_preproc2/or-70-10-4.cnf', './input/Benchmark_preproc2/or-70-20-9.cnf', './input/Benchmark_preproc2/or-100-5-2-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-5-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-6-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-4-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-20.cnf', './input/Benchmark_preproc2/4step.cnf', './input/Benchmark_preproc2/or-100-5-3-UC-10.cnf', './input/Benchmark_preproc2/par8-1.cnf', './input/Benchmark_preproc2/or-100-5-6-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-3-UC-20.cnf', './input/Benchmark_preproc2/blasted_case55.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-20.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-9-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-1-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-9-UC-10.cnf', './input/Benchmark_preproc2/par8-2.cnf', './input/Benchmark_preproc2/or-100-20-5-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-4-UC-20.cnf', './input/Benchmark_preproc2/par8-4.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-30.cnf', './input/Benchmark_preproc2/or-100-20-6-UC-20.cnf', './input/Benchmark_preproc2/par8-3.cnf', './input/Benchmark_preproc2/par8-5.cnf', './input/Benchmark_preproc2/blasted_case8.cnf', './input/Benchmark_preproc2/par32-2.cnf', './input/Benchmark_preproc2/or-100-5-7-UC-10.cnf', './input/Benchmark_preproc2/par32-5.cnf', './input/Benchmark_preproc2/5step.cnf', './input/Benchmark_preproc2/or-100-10-7-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-20.cnf', './input/Benchmark_preproc2/or-100-5-10-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-2-UC-10.cnf', './input/Benchmark_preproc2/par32-3.cnf', './input/Benchmark_preproc2/or-100-10-1-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-7-UC-10.cnf', './input/Benchmark_preproc2/blasted_case105.cnf', './input/Benchmark_preproc2/or-100-10-8-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-4-UC-10.cnf', './input/Benchmark_preproc2/par32-1.cnf', './input/Benchmark_preproc2/par32-4.cnf', './input/Benchmark_preproc2/or-100-5-4-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-2-UC-10.cnf', './input/Benchmark_preproc2/qg1-07.cnf', './input/Benchmark_preproc2/or-100-10-3-UC-10.cnf', './input/Benchmark_preproc2/blasted_case44.cnf', './input/Benchmark_preproc2/or-100-10-4-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-3-UC-20.cnf', './input/Benchmark_preproc2/or-100-20-9-UC-10.cnf', './input/Benchmark_preproc2/blasted_case46.cnf', './input/Benchmark_preproc2/blasted_case5.cnf', './input/Benchmark_preproc2/blasted_case68.cnf', './input/Benchmark_preproc2/or-100-10-10-UC-10.cnf', './input/Benchmark_preproc2/or-100-5-8-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-8-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-2-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-1-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-10-UC-10.cnf', './input/Benchmark_preproc2/blasted_case1.cnf', './input/Benchmark_preproc2/or-100-5-5-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-6-UC-10.cnf', './input/Benchmark_preproc2/or-100-20-6-UC-10.cnf', './input/Benchmark_preproc2/or-100-10-9-UC-10.cnf', './input/Benchmark_preproc2/qg2-07.cnf', './input/Benchmark_preproc2/ais10.cnf', './input/Benchmark_preproc2/nocountdump12.cnf', './input/Benchmark_preproc2/nocountdump30.cnf', './input/Benchmark_preproc2/blasted_case108.cnf', './input/Benchmark_preproc2/or-100-20-3-UC-10.cnf', './input/Benchmark_preproc2/s400.bench.cnf', './input/Benchmark_preproc2/20_rd_r45.cnf', './input/Benchmark_preproc2/nocountdump11.cnf', './input/Benchmark_preproc2/blasted_case54.cnf', './input/Benchmark_preproc2/blasted_case56.cnf', './input/Benchmark_preproc2/2bitmax_6.cnf', './input/Benchmark_preproc2/par16-1-c.cnf', './input/Benchmark_preproc2/s344_3_2.cnf', './input/Benchmark_preproc2/s349_3_2.cnf', './input/Benchmark_preproc2/c432.isc.cnf', './input/Benchmark_preproc2/or-100-5-3.cnf', './input/Benchmark_preproc2/or-100-5-1.cnf', './input/Benchmark_preproc2/blasted_case201.cnf', './input/Benchmark_preproc2/blasted_case202.cnf', './input/Benchmark_preproc2/or-100-5-8.cnf', './input/Benchmark_preproc2/or-100-10-7.cnf', './input/Benchmark_preproc2/or-100-20-5.cnf', './input/Benchmark_preproc2/or-100-5-2.cnf', './input/Benchmark_preproc2/or-100-5-5.cnf', './input/Benchmark_preproc2/or-100-5-7.cnf', './input/Benchmark_preproc2/or-100-5-9.cnf', './input/Benchmark_preproc2/or-100-5-10.cnf', './input/Benchmark_preproc2/or-100-10-3.cnf', './input/Benchmark_preproc2/or-100-10-5.cnf', './input/Benchmark_preproc2/or-100-10-9.cnf', './input/Benchmark_preproc2/or-100-20-1.cnf', './input/Benchmark_preproc2/or-100-20-8.cnf', './input/Benchmark_preproc2/or-100-20-9.cnf', './input/Benchmark_preproc2/or-100-10-1.cnf', './input/Benchmark_preproc2/or-100-10-10.cnf', './input/Benchmark_preproc2/or-100-5-6.cnf', './input/Benchmark_preproc2/or-100-10-2.cnf', './input/Benchmark_preproc2/or-100-10-4.cnf', './input/Benchmark_preproc2/or-100-10-6.cnf', './input/Benchmark_preproc2/or-100-20-2.cnf', './input/Benchmark_preproc2/or-100-20-4.cnf', './input/Benchmark_preproc2/or-100-20-6.cnf', './input/Benchmark_preproc2/or-100-20-7.cnf', './input/Benchmark_preproc2/blasted_case106.cnf', './input/Benchmark_preproc2/fphp-010-020.cnf', './input/Benchmark_preproc2/or-100-5-4.cnf', './input/Benchmark_preproc2/or-100-10-8.cnf', './input/Benchmark_preproc2/or-100-20-10.cnf', './input/Benchmark_preproc2/or-100-20-3.cnf', './input/Benchmark_preproc2/par16-4-c.cnf', './input/Benchmark_preproc2/s298_3_2.cnf', './input/Benchmark_preproc2/s444.bench.cnf', './input/Benchmark_preproc2/blasted_case136.cnf', './input/Benchmark_preproc2/blasted_case133.cnf', './input/Benchmark_preproc2/s344_7_4.cnf', './input/Benchmark_preproc2/s349_7_4.cnf', './input/Benchmark_preproc2/par16-3-c.cnf', './input/Benchmark_preproc2/blasted_case203.cnf', './input/Benchmark_preproc2/blasted_case204.cnf', './input/Benchmark_preproc2/blasted_case205.cnf', './input/Benchmark_preproc2/blasted_case145.cnf', './input/Benchmark_preproc2/blasted_case146.cnf', './input/Benchmark_preproc2/s298_7_4.cnf', './input/Benchmark_preproc2/par16-5-c.cnf', './input/Benchmark_preproc2/s526.bench.cnf', './input/Benchmark_preproc2/s526n.bench.cnf', './input/Benchmark_preproc2/nocountdump10.cnf', './input/Benchmark_preproc2/blasted_case109.cnf', './input/Benchmark_preproc2/84.sk_4_77.cnf', './input/Benchmark_preproc2/blasted_case_1_b14_1.cnf', './input/Benchmark_preproc2/blasted_case_2_b14_1.cnf', './input/Benchmark_preproc2/blasted_case_3_b14_1.cnf', './input/Benchmark_preproc2/par16-2-c.cnf', './input/Benchmark_preproc2/blasted_case132.cnf', './input/Benchmark_preproc2/blasted_case135.cnf', './input/Benchmark_preproc2/registerlesSwap.sk_3_10.cnf', './input/Benchmark_preproc2/s510.bench.cnf', './input/Benchmark_preproc2/blasted_case_3_b14_2.cnf', './input/Benchmark_preproc2/blasted_case_1_b14_2.cnf', './input/Benchmark_preproc2/blasted_case_2_b14_2.cnf', './input/Benchmark_preproc2/sat-grid-pbl-0015.cnf', './input/Benchmark_preproc2/blasted_case41.cnf', './input/Benchmark_preproc2/blasted_case39.cnf', './input/Benchmark_preproc2/blasted_case40.cnf', './input/Benchmark_preproc2/c499.isc.cnf', './input/Benchmark_preproc2/blasted_case14.cnf', './input/Benchmark_preproc2/polynomial.sk_7_25.cnf', './input/Benchmark_preproc2/blasted_case61.cnf', './input/Benchmark_preproc2/C211_FS.cnf', './input/Benchmark_preproc2/nocountdump14.cnf', './input/Benchmark_preproc2/nocountdump15.cnf', './input/Benchmark_preproc2/uf250-017.cnf', './input/Benchmark_preproc2/uf250-033.cnf', './input/Benchmark_preproc2/uf250-05.cnf', './input/Benchmark_preproc2/uf250-066.cnf', './input/Benchmark_preproc2/uf250-082.cnf', './input/Benchmark_preproc2/uf250-01.cnf', './input/Benchmark_preproc2/uf250-010.cnf', './input/Benchmark_preproc2/uf250-0100.cnf', './input/Benchmark_preproc2/uf250-011.cnf', './input/Benchmark_preproc2/uf250-012.cnf', './input/Benchmark_preproc2/uf250-013.cnf', './input/Benchmark_preproc2/uf250-014.cnf', './input/Benchmark_preproc2/uf250-015.cnf', './input/Benchmark_preproc2/uf250-016.cnf', './input/Benchmark_preproc2/uf250-018.cnf', './input/Benchmark_preproc2/uf250-019.cnf', './input/Benchmark_preproc2/uf250-02.cnf', './input/Benchmark_preproc2/uf250-020.cnf', './input/Benchmark_preproc2/uf250-021.cnf', './input/Benchmark_preproc2/uf250-022.cnf', './input/Benchmark_preproc2/uf250-023.cnf', './input/Benchmark_preproc2/uf250-024.cnf', './input/Benchmark_preproc2/uf250-025.cnf', './input/Benchmark_preproc2/uf250-026.cnf', './input/Benchmark_preproc2/uf250-027.cnf', './input/Benchmark_preproc2/uf250-028.cnf', './input/Benchmark_preproc2/uf250-029.cnf', './input/Benchmark_preproc2/uf250-03.cnf', './input/Benchmark_preproc2/uf250-030.cnf', './input/Benchmark_preproc2/uf250-031.cnf', './input/Benchmark_preproc2/uf250-032.cnf', './input/Benchmark_preproc2/uf250-034.cnf', './input/Benchmark_preproc2/uf250-035.cnf', './input/Benchmark_preproc2/uf250-036.cnf', './input/Benchmark_preproc2/uf250-037.cnf', './input/Benchmark_preproc2/uf250-038.cnf', './input/Benchmark_preproc2/uf250-039.cnf', './input/Benchmark_preproc2/uf250-04.cnf', './input/Benchmark_preproc2/uf250-040.cnf', './input/Benchmark_preproc2/uf250-041.cnf', './input/Benchmark_preproc2/uf250-042.cnf', './input/Benchmark_preproc2/uf250-043.cnf', './input/Benchmark_preproc2/uf250-044.cnf', './input/Benchmark_preproc2/uf250-045.cnf', './input/Benchmark_preproc2/uf250-046.cnf', './input/Benchmark_preproc2/uf250-047.cnf', './input/Benchmark_preproc2/uf250-048.cnf', './input/Benchmark_preproc2/uf250-049.cnf', './input/Benchmark_preproc2/uf250-050.cnf', './input/Benchmark_preproc2/uf250-051.cnf', './input/Benchmark_preproc2/uf250-052.cnf', './input/Benchmark_preproc2/uf250-053.cnf', './input/Benchmark_preproc2/uf250-054.cnf', './input/Benchmark_preproc2/uf250-055.cnf', './input/Benchmark_preproc2/uf250-056.cnf', './input/Benchmark_preproc2/uf250-057.cnf', './input/Benchmark_preproc2/uf250-058.cnf', './input/Benchmark_preproc2/uf250-059.cnf', './input/Benchmark_preproc2/uf250-06.cnf', './input/Benchmark_preproc2/uf250-060.cnf', './input/Benchmark_preproc2/uf250-061.cnf', './input/Benchmark_preproc2/uf250-062.cnf', './input/Benchmark_preproc2/uf250-063.cnf', './input/Benchmark_preproc2/uf250-064.cnf', './input/Benchmark_preproc2/uf250-065.cnf', './input/Benchmark_preproc2/uf250-067.cnf', './input/Benchmark_preproc2/uf250-068.cnf', './input/Benchmark_preproc2/uf250-069.cnf', './input/Benchmark_preproc2/uf250-07.cnf', './input/Benchmark_preproc2/uf250-070.cnf', './input/Benchmark_preproc2/uf250-071.cnf', './input/Benchmark_preproc2/uf250-072.cnf', './input/Benchmark_preproc2/uf250-073.cnf', './input/Benchmark_preproc2/uf250-074.cnf', './input/Benchmark_preproc2/uf250-075.cnf', './input/Benchmark_preproc2/uf250-076.cnf', './input/Benchmark_preproc2/uf250-077.cnf', './input/Benchmark_preproc2/uf250-078.cnf', './input/Benchmark_preproc2/uf250-079.cnf', './input/Benchmark_preproc2/uf250-08.cnf', './input/Benchmark_preproc2/uf250-080.cnf', './input/Benchmark_preproc2/uf250-081.cnf', './input/Benchmark_preproc2/uf250-083.cnf', './input/Benchmark_preproc2/uf250-084.cnf', './input/Benchmark_preproc2/uf250-085.cnf', './input/Benchmark_preproc2/uf250-086.cnf', './input/Benchmark_preproc2/uf250-087.cnf', './input/Benchmark_preproc2/uf250-088.cnf', './input/Benchmark_preproc2/uf250-089.cnf', './input/Benchmark_preproc2/uf250-09.cnf', './input/Benchmark_preproc2/uf250-090.cnf', './input/Benchmark_preproc2/uf250-091.cnf', './input/Benchmark_preproc2/uf250-092.cnf', './input/Benchmark_preproc2/uf250-093.cnf', './input/Benchmark_preproc2/uf250-094.cnf', './input/Benchmark_preproc2/uf250-095.cnf', './input/Benchmark_preproc2/uf250-096.cnf', './input/Benchmark_preproc2/uf250-097.cnf', './input/Benchmark_preproc2/uf250-098.cnf', './input/Benchmark_preproc2/uf250-099.cnf', './input/Benchmark_preproc2/binsearch.16.pp.cnf', './input/Benchmark_preproc2/binsearch.32.pp.cnf', './input/Benchmark_preproc2/s420.1.bench.cnf', './input/Benchmark_preproc2/23_rd_r45.cnf', './input/Benchmark_preproc2/blasted_case119.cnf', './input/Benchmark_preproc2/blasted_case9.cnf', './input/Benchmark_preproc2/blasted_case_1_b14_3.cnf', './input/Benchmark_preproc2/blasted_case_2_b14_3.cnf', './input/Benchmark_preproc2/blasted_case_3_b14_3.cnf', './input/Benchmark_preproc2/s382_3_2.cnf', './input/Benchmark_preproc2/blasted_case123.cnf', './input/Benchmark_preproc2/ais12.cnf', './input/Benchmark_preproc2/s344_15_7.cnf', './input/Benchmark_preproc2/blasted_case120.cnf', './input/Benchmark_preproc2/s349_15_7.cnf', './input/Benchmark_preproc2/qg3-08.cnf', './input/Benchmark_preproc2/s382_7_4.cnf', './input/Benchmark_preproc2/s298_15_7.cnf', './input/Benchmark_preproc2/blasted_case2.cnf', './input/Benchmark_preproc2/blasted_case3.cnf', './input/Benchmark_preproc2/qg1-08.cnf', './input/Benchmark_preproc2/3blocks.cnf', './input/Benchmark_preproc2/blasted_case110.cnf', './input/Benchmark_preproc2/blasted_case57.cnf', './input/Benchmark_preproc2/s444_3_2.cnf', './input/Benchmark_preproc2/blasted_case121.cnf', './input/Benchmark_preproc2/blasted_case62.cnf', './input/Benchmark_preproc2/s420_3_2.cnf', './input/Benchmark_preproc2/s420_new1_3_2.cnf', './input/Benchmark_preproc2/s420_new_3_2.cnf', './input/Benchmark_preproc2/blasted_case15.cnf', './input/Benchmark_preproc2/s510_3_2.cnf', './input/Benchmark_preproc2/blasted_case_0_b11_1.cnf', './input/Benchmark_preproc2/blasted_case_1_b11_1.cnf', './input/Benchmark_preproc2/qg2-08.cnf', './input/Benchmark_preproc2/blasted_case126.cnf', './input/Benchmark_preproc2/fphp-015-020.cnf', './input/Benchmark_preproc2/ls8-normalized.cnf', './input/Benchmark_preproc2/s444_7_4.cnf', './input/Benchmark_preproc2/blasted_case111.cnf', './input/Benchmark_preproc2/s420_new1_7_4.cnf', './input/Benchmark_preproc2/s420_7_4.cnf', './input/Benchmark_preproc2/s420_new_7_4.cnf', './input/Benchmark_preproc2/blasted_case113.cnf', './input/Benchmark_preproc2/blasted_case117.cnf', './input/Benchmark_preproc2/blasted_case118.cnf', './input/Benchmark_preproc2/blasted_case10.cnf', './input/Benchmark_preproc2/mixdup.cnf', './input/Benchmark_preproc2/s510_7_4.cnf', './input/Benchmark_preproc2/s832.bench.cnf', './input/Benchmark_preproc2/blasted_case122.cnf', './input/Benchmark_preproc2/nocountdump7.cnf', './input/Benchmark_preproc2/s820.bench.cnf', './input/Benchmark_preproc2/blasted_case6.cnf', './input/Benchmark_preproc2/s510_15_7.cnf', './input/Benchmark_preproc2/qg7-09.cnf', './input/Benchmark_preproc2/s382_15_7.cnf', './input/Benchmark_preproc2/s420_new_15_7.cnf', './input/Benchmark_preproc2/tire-1.cnf', './input/Benchmark_preproc2/C211_FW.cnf', './input/Benchmark_preproc2/s420_15_7.cnf', './input/Benchmark_preproc2/s420_new1_15_7.cnf', './input/Benchmark_preproc2/blasted_case_0_b12_1.cnf', './input/Benchmark_preproc2/blasted_case_1_b12_1.cnf', './input/Benchmark_preproc2/blasted_case_2_b12_1.cnf', './input/Benchmark_preproc2/10.sk_1_46.cnf', './input/Benchmark_preproc2/s526_3_2.cnf', './input/Benchmark_preproc2/s444_15_7.cnf', './input/Benchmark_preproc2/2bitadd_11.cnf', './input/Benchmark_preproc2/s526a_3_2.cnf', './input/Benchmark_preproc2/nocountdump6.cnf', './input/Benchmark_preproc2/nocountdump18.cnf', './input/Benchmark_preproc2/nocountdump19.cnf', './input/Benchmark_preproc2/s526_7_4.cnf', './input/Benchmark_preproc2/s526a_7_4.cnf', './input/Benchmark_preproc2/blasted_case19.cnf', './input/Benchmark_preproc2/blasted_case20.cnf', './input/Benchmark_preproc2/blasted_case125.cnf', './input/Benchmark_preproc2/qg6-09.cnf', './input/Benchmark_preproc2/blasted_case143.cnf', './input/Benchmark_preproc2/2bitadd_12.cnf', './input/Benchmark_preproc2/blasted_case35.cnf', './input/Benchmark_preproc2/blasted_case34.cnf', './input/Benchmark_preproc2/4blocksb.cnf', './input/Benchmark_preproc2/c880.isc.cnf', './input/Benchmark_preproc2/s953.bench.cnf', './input/Benchmark_preproc2/nocountdump21.cnf', './input/Benchmark_preproc2/nocountdump23.cnf', './input/Benchmark_preproc2/sat-grid-pbl-0020.cnf', './input/Benchmark_preproc2/blasted_case114.cnf', './input/Benchmark_preproc2/blasted_case115.cnf', './input/Benchmark_preproc2/blasted_case131.cnf', './input/Benchmark_preproc2/s641.bench.cnf', './input/Benchmark_preproc2/27.sk_3_32.cnf', './input/Benchmark_preproc2/blasted_case116.cnf', './input/Benchmark_preproc2/nocountdump20.cnf', './input/Benchmark_preproc2/qg4-09.cnf', './input/Benchmark_preproc2/s526_15_7.cnf', './input/Benchmark_preproc2/s526a_15_7.cnf', './input/Benchmark_preproc2/s713.bench.cnf', './input/Benchmark_preproc2/C171_FR.cnf', './input/Benchmark_preproc2/C638_FVK.cnf', './input/Benchmark_preproc2/ls9-normalized.cnf', './input/Benchmark_preproc2/bw_large.a.cnf', './input/Benchmark_preproc2/huge.cnf', './input/Benchmark_preproc2/nocountdump22.cnf', './input/Benchmark_preproc2/blasted_case140.cnf', './input/Benchmark_preproc2/C140_FC.cnf', './input/Benchmark_preproc2/5_100_sd_schur.cnf', './input/Benchmark_preproc2/lang12.cnf', './input/Benchmark_preproc2/s641_3_2.cnf', './input/Benchmark_preproc2/nocountdump9.cnf', './input/Benchmark_preproc2/35.sk_3_52.cnf', './input/Benchmark_preproc2/s953a_3_2.cnf', './input/Benchmark_preproc2/blasted_squaring51.cnf', './input/Benchmark_preproc2/C210_FVF.cnf', './input/Benchmark_preproc2/C215_FC.cnf', './input/Benchmark_preproc2/blasted_squaring50.cnf', './input/Benchmark_preproc2/s641_7_4.cnf', './input/Benchmark_preproc2/s953a_7_4.cnf', './input/Benchmark_preproc2/C638_FKB.cnf', './input/Benchmark_preproc2/s713_3_2.cnf', './input/Benchmark_preproc2/s838.1.bench.cnf', './input/Benchmark_preproc2/C230_FR.cnf', './input/Benchmark_preproc2/C209_FA.cnf', './input/Benchmark_preproc2/s713_7_4.cnf', './input/Benchmark_preproc2/C638_FKA.cnf', './input/Benchmark_preproc2/tire-2.cnf', './input/Benchmark_preproc2/s1238.bench.cnf', './input/Benchmark_preproc2/hanoi4.cnf', './input/Benchmark_preproc2/C140_FW.cnf', './input/Benchmark_preproc2/C170_FR.cnf', './input/Benchmark_preproc2/c1355.isc.cnf', './input/Benchmark_preproc2/C140_FV.cnf', './input/Benchmark_preproc2/tire-3.cnf', './input/Benchmark_preproc2/s1196.bench.cnf', './input/Benchmark_preproc2/s641_15_7.cnf', './input/Benchmark_preproc2/s953a_15_7.cnf', './input/Benchmark_preproc2/C163_FW.cnf', './input/Benchmark_preproc2/C203_FCL.cnf', './input/Benchmark_preproc2/C209_FC.cnf', './input/Benchmark_preproc2/C129_FR.cnf', './input/Benchmark_preproc2/blasted_case18.cnf', './input/Benchmark_preproc2/10random.cnf', './input/Benchmark_preproc2/s713_15_7.cnf', './input/Benchmark_preproc2/C220_FW.cnf', './input/Benchmark_preproc2/s820a_3_2.cnf', './input/Benchmark_preproc2/s838_3_2.cnf', './input/Benchmark_preproc2/C220_FV.cnf', './input/Benchmark_preproc2/nocountdump1.cnf', './input/Benchmark_preproc2/nocountdump3.cnf', './input/Benchmark_preproc2/s832a_3_2.cnf', './input/Benchmark_preproc2/blasted_case107.cnf', './input/Benchmark_preproc2/s820a_7_4.cnf', './input/Benchmark_preproc2/s838_7_4.cnf', './input/Benchmark_preproc2/nocountdump17.cnf', './input/Benchmark_preproc2/GuidanceService2.sk_2_27.cnf', './input/Benchmark_preproc2/s832a_7_4.cnf', './input/Benchmark_preproc2/C203_FS.cnf', './input/Benchmark_preproc2/blasted_case130.cnf', './input/Benchmark_preproc2/blasted_case_0_b12_2.cnf', './input/Benchmark_preproc2/blasted_case_1_b12_2.cnf', './input/Benchmark_preproc2/blasted_case_2_b12_2.cnf', './input/Benchmark_preproc2/C208_FA.cnf', './input/Benchmark_preproc2/sum.32.cnf', './input/Benchmark_preproc2/blasted_case213.cnf', './input/Benchmark_preproc2/blasted_case214.cnf', './input/Benchmark_preproc2/sat-grid-pbl-0025.cnf', './input/Benchmark_preproc2/C208_FC.cnf', './input/Benchmark_preproc2/C168_FW.cnf', './input/Benchmark_preproc2/ls10-normalized.cnf', './input/Benchmark_preproc2/s1494.bench.cnf', './input/Benchmark_preproc2/C203_FW.cnf', './input/Benchmark_preproc2/s1488.bench.cnf', './input/Benchmark_preproc2/s820a_15_7.cnf', './input/Benchmark_preproc2/s838_15_7.cnf', './input/Benchmark_preproc2/nocountdump2.cnf', './input/Benchmark_preproc2/s832a_15_7.cnf', './input/Benchmark_preproc2/s1238a_3_2.cnf', './input/Benchmark_preproc2/5_140_sd_schur.cnf', './input/Benchmark_preproc2/s1196a_3_2.cnf', './input/Benchmark_preproc2/blasted_squaring22.cnf', './input/Benchmark_preproc2/blasted_squaring24.cnf', './input/Benchmark_preproc2/blasted_squaring20.cnf', './input/Benchmark_preproc2/blasted_squaring21.cnf', './input/Benchmark_preproc2/s1238a_7_4.cnf', './input/Benchmark_preproc2/C202_FS.cnf', './input/Benchmark_preproc2/s1196a_7_4.cnf', './input/Benchmark_preproc2/blasted_case12.cnf', './input/Benchmark_preproc2/blasted_squaring23.cnf', './input/Benchmark_preproc2/scenarios_tree_delete3.sb.pl.sk_2_32.cnf', './input/Benchmark_preproc2/blasted_case144.cnf', './input/Benchmark_preproc2/cnt06.shuffled.cnf', './input/Benchmark_preproc2/C210_FS.cnf', './input/Benchmark_preproc2/C202_FW.cnf', './input/Benchmark_preproc2/s1423.bench.cnf', './input/Benchmark_preproc2/c1908.isc.cnf', './input/Benchmark_preproc2/s1238a_15_7.cnf', './input/Benchmark_preproc2/4blocks.cnf', './input/Benchmark_preproc2/s1196a_15_7.cnf', './input/Benchmark_preproc2/s1423a_3_2.cnf', './input/Benchmark_preproc2/C210_FW.cnf', './input/Benchmark_preproc2/log-1.cnf', './input/Benchmark_preproc2/prob001.pddl.cnf', './input/Benchmark_preproc2/tire-4.cnf', './input/Benchmark_preproc2/s1423a_7_4.cnf', './input/Benchmark_preproc2/blasted_case_0_b14_1.cnf', './input/Benchmark_preproc2/countdump6.cnf', './input/Benchmark_preproc2/blasted_case207.cnf', './input/Benchmark_preproc2/blasted_case208.cnf', './input/Benchmark_preproc2/qg5-11.cnf', './input/Benchmark_preproc2/logistics.a.cnf', './input/Benchmark_preproc2/blasted_squaring27.cnf', './input/Benchmark_preproc2/blasted_case50.cnf', './input/Benchmark_preproc2/blasted_case138.cnf', './input/Benchmark_preproc2/blasted_case139.cnf', './input/Benchmark_preproc2/blasted_squaring25.cnf', './input/Benchmark_preproc2/logistics.b.cnf', './input/Benchmark_preproc2/s1423a_15_7.cnf', './input/Benchmark_preproc2/s1488_3_2.cnf', './input/Benchmark_preproc2/blasted_case210.cnf', './input/Benchmark_preproc2/blasted_case211.cnf', './input/Benchmark_preproc2/s1488_7_4.cnf', './input/Benchmark_preproc2/lang15.cnf', './input/Benchmark_preproc2/lang16.cnf', './input/Benchmark_preproc2/GuidanceService.sk_4_27.cnf', './input/Benchmark_preproc2/107.sk_3_90.cnf', './input/Benchmark_preproc2/blasted_squaring70.cnf', './input/Benchmark_preproc2/blasted_squaring2.cnf', './input/Benchmark_preproc2/blasted_squaring3.cnf', './input/Benchmark_preproc2/blasted_squaring5.cnf', './input/Benchmark_preproc2/blasted_squaring6.cnf', './input/Benchmark_preproc2/blasted_squaring1.cnf', './input/Benchmark_preproc2/blasted_squaring4.cnf', './input/Benchmark_preproc2/blasted_squaring26.cnf', './input/Benchmark_preproc2/blasted_case42.cnf', './input/Benchmark_preproc2/ls11-normalized.cnf', './input/Benchmark_preproc2/blasted_case_1_ptb_1.cnf', './input/Benchmark_preproc2/blasted_case_2_ptb_1.cnf', './input/Benchmark_preproc2/s1488_15_7.cnf', './input/Benchmark_preproc2/par32-2-c.cnf', './input/Benchmark_preproc2/sat-grid-pbl-0030.cnf', './input/Benchmark_preproc2/par32-1-c.cnf', './input/Benchmark_preproc2/par32-3-c.cnf', './input/Benchmark_preproc2/par32-4-c.cnf', './input/Benchmark_preproc2/blasted_squaring11.cnf', './input/Benchmark_preproc2/32.sk_4_38.cnf', './input/Benchmark_preproc2/par32-5-c.cnf', './input/Benchmark_preproc2/79.sk_4_40.cnf', './input/Benchmark_preproc2/nocountdump24.cnf', './input/Benchmark_preproc2/blasted_squaring30.cnf', './input/Benchmark_preproc2/blockmap_05_01.net.cnf', './input/Benchmark_preproc2/blasted_squaring28.cnf', './input/Benchmark_preproc2/blasted_case37.cnf', './input/Benchmark_preproc2/bw_large.b.cnf', './input/Benchmark_preproc2/55.sk_3_46.cnf', './input/Benchmark_preproc2/blasted_squaring10.cnf', './input/Benchmark_preproc2/blasted_squaring8.cnf', './input/Benchmark_preproc2/nocountdump4.cnf', './input/Benchmark_preproc2/blasted_case_1_b14_even.cnf', './input/Benchmark_preproc2/blasted_case3_b14_even3.cnf', './input/Benchmark_preproc2/blasted_case_2_b14_even.cnf', './input/Benchmark_preproc2/blasted_case1_b14_even3.cnf', './input/Benchmark_preproc2/blasted_TR_device_1_linear.cnf', './input/Benchmark_preproc2/bmc-ibm-2.cnf', './input/Benchmark_preproc2/blasted_squaring29.cnf', './input/Benchmark_preproc2/logistics.c.cnf', './input/Benchmark_preproc2/IssueServiceImpl.sk_8_30.cnf', './input/Benchmark_preproc2/c2670.isc.cnf', './input/Benchmark_preproc2/111.sk_2_36.cnf', './input/Benchmark_preproc2/nocountdump31.cnf', './input/Benchmark_preproc2/nocountdump8.cnf', './input/Benchmark_preproc2/blasted_case209.cnf', './input/Benchmark_preproc2/blasted_case212.cnf', './input/Benchmark_preproc2/rb.cnf', './input/Benchmark_preproc2/ls12-normalized.cnf', './input/Benchmark_preproc2/nocountdump25.cnf', './input/Benchmark_preproc2/blasted_TR_b14_1_linear.cnf', './input/Benchmark_preproc2/lang19.cnf', './input/Benchmark_preproc2/ra.cnf', './input/Benchmark_preproc2/UserServiceImpl.sk_8_32.cnf', './input/Benchmark_preproc2/log-2.cnf', './input/Benchmark_preproc2/prob002.pddl.cnf', './input/Benchmark_preproc2/blasted_case_1_4_b14_even.cnf', './input/Benchmark_preproc2/blasted_case_3_4_b14_even.cnf', './input/Benchmark_preproc2/blockmap_05_02.net.cnf', './input/Benchmark_preproc2/lang20.cnf', './input/Benchmark_preproc2/log-3.cnf', './input/Benchmark_preproc2/prob003.pddl.cnf', './input/Benchmark_preproc2/scenarios_aig_insertion1.sb.pl.sk_3_60.cnf', './input/Benchmark_preproc2/qg7-13.cnf', './input/Benchmark_preproc2/blasted_squaring9.cnf', './input/Benchmark_preproc2/blasted_TR_b14_2_linear.cnf', './input/Benchmark_preproc2/blasted_squaring14.cnf', './input/Benchmark_preproc2/blasted_case_0_ptb_1.cnf', './input/Benchmark_preproc2/ActivityService.sk_11_27.cnf', './input/Benchmark_preproc2/53.sk_4_32.cnf', './input/Benchmark_preproc2/hanoi5.cnf', './input/Benchmark_preproc2/blasted_squaring12.cnf', './input/Benchmark_preproc2/blasted_case49.cnf', './input/Benchmark_preproc2/PhaseService.sk_14_27.cnf', './input/Benchmark_preproc2/ls13-normalized.cnf', './input/Benchmark_preproc2/blasted_squaring16.cnf', './input/Benchmark_preproc2/blasted_squaring7.cnf', './input/Benchmark_preproc2/blockmap_05_03.net.cnf', './input/Benchmark_preproc2/blasted_TR_b12_1_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b14_3_linear.cnf', './input/Benchmark_preproc2/cnt07.shuffled.cnf', './input/Benchmark_preproc2/109.sk_4_36.cnf', './input/Benchmark_preproc2/prob004-log-a.cnf', './input/Benchmark_preproc2/IterationService.sk_12_27.cnf', './input/Benchmark_preproc2/51.sk_4_38.cnf', './input/Benchmark_preproc2/lang23.cnf', './input/Benchmark_preproc2/countdump8.cnf', './input/Benchmark_preproc2/ActivityService2.sk_10_27.cnf', './input/Benchmark_preproc2/56.sk_6_38.cnf', './input/Benchmark_preproc2/blasted_TR_ptb_1_linear.cnf', './input/Benchmark_preproc2/rc.cnf', './input/Benchmark_preproc2/binsearch.16.cnf', './input/Benchmark_preproc2/lang24.cnf', './input/Benchmark_preproc2/ls14-normalized.cnf', './input/Benchmark_preproc2/blasted_TR_device_1_even_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b12_2_linear.cnf', './input/Benchmark_preproc2/log-4.cnf', './input/Benchmark_preproc2/prob004.pddl.cnf', './input/Benchmark_preproc2/prob012.pddl.cnf', './input/Benchmark_preproc2/scenarios_aig_insertion2.sb.pl.sk_3_60.cnf', './input/Benchmark_preproc2/ConcreteActivityService.sk_13_28.cnf', './input/Benchmark_preproc2/blasted_case_0_b12_even2.cnf', './input/Benchmark_preproc2/blasted_case_1_b12_even2.cnf', './input/Benchmark_preproc2/blasted_case_2_b12_even2.cnf', './input/Benchmark_preproc2/blasted_case_0_b12_even1.cnf', './input/Benchmark_preproc2/blasted_case_1_b12_even1.cnf', './input/Benchmark_preproc2/blasted_case_2_b12_even1.cnf', './input/Benchmark_preproc2/blasted_case142.cnf', './input/Benchmark_preproc2/lang27.cnf', './input/Benchmark_preproc2/ls15-normalized.cnf', './input/Benchmark_preproc2/log-5.cnf', './input/Benchmark_preproc2/prob005.pddl.cnf', './input/Benchmark_preproc2/blasted_case_1_ptb_2.cnf', './input/Benchmark_preproc2/blasted_case_2_ptb_2.cnf', './input/Benchmark_preproc2/lang28.cnf', './input/Benchmark_preproc2/ProjectService3.sk_12_55.cnf', './input/Benchmark_preproc2/bw_large.c.cnf', './input/Benchmark_preproc2/57.sk_4_64.cnf', './input/Benchmark_preproc2/ls16-normalized.cnf', './input/Benchmark_preproc2/c7552.isc.cnf', './input/Benchmark_preproc2/NotificationServiceImpl2.sk_10_36.cnf', './input/Benchmark_preproc2/blasted_case_0_ptb_2.cnf', './input/Benchmark_preproc2/ProcessBean.sk_8_64.cnf', './input/Benchmark_preproc2/fs-07.net.cnf', './input/Benchmark_preproc2/71.sk_3_65.cnf', './input/Benchmark_preproc2/mastermind_05_08_03.net.cnf', './input/Benchmark_preproc2/blasted_case104.cnf', './input/Benchmark_preproc2/blasted_TR_ptb_2_linear.cnf', './input/Benchmark_preproc2/s5378a_3_2.cnf', './input/Benchmark_preproc2/s5378a_7_4.cnf', './input/Benchmark_preproc2/s5378a_15_7.cnf', './input/Benchmark_preproc2/blasted_case141.cnf', './input/Benchmark_preproc2/bmc-ibm-7.cnf', './input/Benchmark_preproc2/blasted_case_0_b12_even3.cnf', './input/Benchmark_preproc2/blasted_case_1_b12_even3.cnf', './input/Benchmark_preproc2/blasted_case_2_b12_even3.cnf', './input/Benchmark_preproc2/mastermind_06_08_03.net.cnf', './input/Benchmark_preproc2/70.sk_3_40.cnf', './input/Benchmark_preproc2/cnt08.shuffled.cnf', './input/Benchmark_preproc2/alu2_gr_rcs_w8.shuffled.cnf', './input/Benchmark_preproc2/blasted_squaring40.cnf', './input/Benchmark_preproc2/blasted_squaring42.cnf', './input/Benchmark_preproc2/blasted_squaring41.cnf', './input/Benchmark_preproc2/80.sk_2_48.cnf', './input/Benchmark_preproc2/3bitadd_31.cnf', './input/Benchmark_preproc2/countdump3.cnf', './input/Benchmark_preproc2/countdump5.cnf', './input/Benchmark_preproc2/countdump4.cnf', './input/Benchmark_preproc2/binsearch.32.cnf', './input/Benchmark_preproc2/3bitadd_32.cnf', './input/Benchmark_preproc2/c880_gr_rcs_w7.shuffled.cnf', './input/Benchmark_preproc2/logistics.d.cnf', './input/Benchmark_preproc2/mastermind_03_08_04.net.cnf', './input/Benchmark_preproc2/77.sk_3_44.cnf', './input/Benchmark_preproc2/bmc-ibm-5.cnf', './input/Benchmark_preproc2/blasted_squaring60.cnf', './input/Benchmark_preproc2/scenarios_tree_delete.sb.pl.sk_3_30.cnf', './input/Benchmark_preproc2/7.sk_4_50.cnf', './input/Benchmark_preproc2/scenarios_tree_insert_insert.sb.pl.sk_3_68.cnf', './input/Benchmark_preproc2/mastermind_04_08_04.net.cnf', './input/Benchmark_preproc2/36.sk_3_77.cnf', './input/Benchmark_preproc2/mastermind_10_08_03.net.cnf', './input/Benchmark_preproc2/63.sk_3_64.cnf', './input/Benchmark_preproc2/17.sk_3_45.cnf', './input/Benchmark_preproc2/s9234a_3_2.cnf', './input/Benchmark_preproc2/s9234a_7_4.cnf', './input/Benchmark_preproc2/bw_large.d.cnf', './input/Benchmark_preproc2/s9234a_15_7.cnf', './input/Benchmark_preproc2/19.sk_3_48.cnf', './input/Benchmark_preproc2/doublyLinkedList.sk_8_37.cnf', './input/Benchmark_preproc2/LoginService.sk_20_34.cnf', './input/Benchmark_preproc2/bmc-ibm-1.cnf', './input/Benchmark_preproc2/mastermind_03_08_05.net.cnf', './input/Benchmark_preproc2/Pollard.sk_1_10.cnf', './input/Benchmark_preproc2/29.sk_3_45.cnf', './input/Benchmark_preproc2/blasted_TR_b14_even3_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b14_even_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b12_even2_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b12_even3_linear.cnf', './input/Benchmark_preproc2/blasted_TR_b12_even7_linear.cnf', './input/Benchmark_preproc2/81.sk_5_51.cnf', './input/Benchmark_preproc2/blockmap_10_01.net.cnf', './input/Benchmark_preproc2/cnt09.shuffled.cnf', './input/Benchmark_preproc2/s13207a_3_2.cnf', './input/Benchmark_preproc2/isolateRightmost.sk_7_481.cnf', './input/Benchmark_preproc2/s13207a_7_4.cnf', './input/Benchmark_preproc2/fs-10.net.cnf', './input/Benchmark_preproc2/s13207a_15_7.cnf', './input/Benchmark_preproc2/blasted_TR_b14_even2_linear.cnf', './input/Benchmark_preproc2/blockmap_10_02.net.cnf', './input/Benchmark_preproc2/20.sk_1_51.cnf', './input/Benchmark_preproc2/LoginService2.sk_23_36.cnf', './input/Benchmark_preproc2/bmc-ibm-13.cnf', './input/Benchmark_preproc2/s15850a_3_2.cnf', './input/Benchmark_preproc2/s15850a_7_4.cnf', './input/Benchmark_preproc2/s15850a_15_7.cnf', './input/Benchmark_preproc2/sort.sk_8_52.cnf', './input/Benchmark_preproc2/blockmap_10_03.net.cnf', './input/Benchmark_preproc2/scenarios_tree_delete4.sb.pl.sk_4_114.cnf', './input/Benchmark_preproc2/bmc-ibm-3.cnf', './input/Benchmark_preproc2/tutorial2.sk_3_4.cnf', './input/Benchmark_preproc2/parity.sk_11_11.cnf', './input/Benchmark_preproc2/bmc-ibm-4.cnf', './input/Benchmark_preproc2/110.sk_3_88.cnf', './input/Benchmark_preproc2/enqueueSeqSK.sk_10_42.cnf', './input/Benchmark_preproc2/scenarios_tree_delete2.sb.pl.sk_8_114.cnf', './input/Benchmark_preproc2/s35932_3_2.cnf', './input/Benchmark_preproc2/s35932_7_4.cnf', './input/Benchmark_preproc2/s35932_15_7.cnf', './input/Benchmark_preproc2/karatsuba.sk_7_41.cnf', './input/Benchmark_preproc2/logcount.sk_16_86.cnf', './input/Benchmark_preproc2/54.sk_12_97.cnf', './input/Benchmark_preproc2/cnt10.shuffled.cnf', './input/Benchmark_preproc2/fs-13.net.cnf', './input/Benchmark_preproc2/30.sk_5_76.cnf', './input/Benchmark_preproc2/s38584_3_2.cnf', './input/Benchmark_preproc2/s38584_7_4.cnf', './input/Benchmark_preproc2/s38584_15_7.cnf', './input/Benchmark_preproc2/signedAvg.sk_8_1020.cnf', './input/Benchmark_preproc2/scenarios_treemax.sb.pl.sk_7_19.cnf', './input/Benchmark_preproc2/s38417_3_2.cnf', './input/Benchmark_preproc2/s38417_7_4.cnf', './input/Benchmark_preproc2/s38417_15_7.cnf', './input/Benchmark_preproc2/scenarios_treemin.sb.pl.sk_9_19.cnf']

    # count_irrelevant_literals()
    # remove_unconstrained_vars()
    # preprocess_folder()
    # for f in ordered_instances:
    #     preprocess(f)
    # order_files_based_on_nb_vars()
    # exit(19)
    #
    # d = "./input/Benchmark_preproc2/"
    # write_minic2d_file(d+"fs-01.net.cnf", d )
    # exit(1)
    # a = [1,3,7,8,10]
    # b = [1,2,6,9,10]
    # c = [2,3,5]
    # literals = [1,2,3,4,5,6,7,8,9,10]
    # util.create_literal_mapping(literals, a)

    #eliminate experiments that have less then 50 variables

    d = "./input/Dataset_preproc/"
    files = [f for f in os.listdir(d) if re.match('.*\.cnf', f) and "temp" not in f]
    for filename in files:
        f = os.path.join(d, filename)
        # if not os.path.exists(f.replace("cnf", "w")):
        utils.generate_random_uniform_weights(f)

