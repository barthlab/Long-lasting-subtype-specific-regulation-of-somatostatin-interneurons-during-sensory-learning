from src import *


if __name__ == "__main__":
    # for exp_id in EXP_LIST:
    # for exp_id in ("Ai148_SAT", "Calb2_SAT"):
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        print(f"Cell num: {len(mitten_data.cells_uid)}", [x for x in mitten_data.cells_uid])
        print([x.name for x in mitten_data.days])



    # plot birdsong raster