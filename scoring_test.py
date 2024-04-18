from CNFmodelD4 import WCNF
import os
import greedy_selective_backboneD4 as d4wcnf

def counts():
    wcnf = WCNF(scalar=0)
    wcnf.load_file("./input/test/nqueens_4.cnf")
    nb_nodes, nb_edges, wmc, mc, comp_time = wcnf.compile_d4_wmc("./input/test/nqueens_4.cnf",
                                                              "./input/test/nqueens_4.w")
    print(nb_nodes, nb_edges, wmc, mc, comp_time)
    print(wcnf.occurance(1,0))
    print(wcnf.occurance(1,1))
    print(wcnf.adjusted_occurance(1,0))
    print(wcnf.adjusted_occurance(1,1))

    var = 1
    score_type = "half"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0, s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1,s)
    score_type = "occratio"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0, s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1,s)
    score_type = "adjoccratio"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0,s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1,s)

    var = 7
    score_type = "half"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0, s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1, s)
    score_type = "occratio"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0, s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1, s)
    score_type = "adjoccratio"
    s = wcnf.calculate_score(var, 0, score_type)
    print(score_type, var, 0, s)
    s = wcnf.calculate_score(var, 1, score_type)
    print(score_type, var, 1, s)

def run_expr():
    # inobj = "wscore_half"  # = "comp" -- for whcn the compilation has to be performed  -or "count" for literal and such countign
    # inobj = "wscore_occratio"  # = "comp" -- for whcn the compilation has to be performed  -or "count" for literal and such countign
    inobj = "wscore_adjoccratio"  # = "comp" -- for whcn the compilation has to be performed  -or "count" for literal and such countign
    input = "./input/"
    folder = "test/"
    out_folder = "./results/" + folder.replace("/", "_" + inobj + "/")
    seed = 1234
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    d = input + folder

    files = ["nqueens_4.cnf"]
    print(files)
    f_count = 0
    for filename in files:
        f_count += 1
        f = os.path.join(d, filename)
        print(filename)
        for type in ["random_selection"]:
            d4wcnf.run_sdd(type, d, f, seed, out_folder, inobj, scalar=0)

if __name__ == "__main__":
    # counts()
    run_expr()