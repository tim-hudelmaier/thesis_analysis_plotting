import json
import multiprocessing as mp
from pathlib import Path
from uuid import uuid4

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

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
    pf.boost(write_results=False, dump_train_test_data=True, eval_test_set=False)
    f = list(pf.spectrum_index.keys())[0]

    tt_data_dir = DATA_DIR / "experiment_outputs" / "04854_F1_R8_P0109699E13_TMT10" / "tt_data"
    model_dir = DATA_DIR / "experiment_outputs" / "04854_F1_R8_P0109699E13_TMT10" / "models"

    results = []
    for file in tt_data_dir.glob("*.json"):
        fold = int(str(file.stem).split("_")[-1][-1])
        engine_paths = model_dir.glob(f"*f{fold}.json")

        for engine_path in engine_paths:
            epoch = int(str(engine_path.stem).split("_")[-2][-1])
            spectra = [s for s in json.load(open(file, "r"))["test_spectra"]]
            pf.engine = peptide_forest.training.get_classifier()
            pf.engine.load_model(engine_path)
            eval_gen = pf.get_data_chunk(file=f, reference_spectra=spectra)
            df = pf.get_results(gen=eval_gen, use_disk=False, write_output=False)
            n_psms = df["top_target_peptide_forest"].sum()
            results.append((fold, epoch, n_psms))

    plt_df = pd.DataFrame(results, columns=["fold", "epoch", "n_psms"])
    plt_df = plt_df.groupby("epoch").sum().drop(columns=["fold"]).reset_index()

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=plt_df["epoch"], y=plt_df["n_psms"], marker="o", color="black")

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Total PSMs", fontsize=14)
    plt.title("Total PSMs per Iteration", fontsize=16)
    plt.show()
