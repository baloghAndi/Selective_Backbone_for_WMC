import greedy_selective_backboneD4
import sys
import os

if __name__ == "__main__":
    seed = 1234
    d = sys.argv[1] #"./input/wmc2022_track2_private/"
    folder = d.split("/")[-2]
    filename = sys.argv[2]
    inobj = sys.argv[3]
    alg_type = sys.argv[4]
    part = sys.argv[5]
    out_folder = "./results/" + folder + "_NO_COMPILE" + part+ "_"+ inobj + "/"

    print(alg_type, inobj, filename)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    greedy_selective_backboneD4.run_sdd(alg_type, filename, seed, out_folder, inobj, NO_COMPILE=True)

    exit(0)
