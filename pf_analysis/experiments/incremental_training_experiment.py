import multiprocessing as mp
from pathlib import Path
from uuid import uuid4

import peptide_forest

DATA_DIR = Path("../../data")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    output = DATA_DIR / "experiment_outputs" / f"{uuid4()}_output.csv"
    pf = peptide_forest.PeptideForest(
        config_path="./config_files/config_local_export_models.json",  # args.c
        output=output,  # args.o,
        memory_limit=None,  # args.m,
        max_mp_count=1,  # args.mp_limit,
    )
    pf.boost(write_results=False, dump_train_test_data=True)


