import json
import multiprocessing as mp
from pathlib import Path
from uuid import uuid4
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import peptide_forest

DATA_DIR = Path("./data")
PALETTE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
]


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = DATA_DIR / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    output = dir_path / f"{uuid4()}_output.csv"
    pf = peptide_forest.PeptideForest(
        config_path="config_files/config_local_export_models.json",  # args.c
        output=output,  # args.o,
        memory_limit=None,  # args.m,
        max_mp_count=1,  # args.mp_limit,
    )
    pf.boost(
        write_results=False,
        dump_train_test_data=True,
        eval_test_set=False,
        drop_used_spectra=False,
    )
    f = list(pf.spectrum_index.keys())[0]
    filepath = list(pf.spectrum_index.keys())[0].split("/")[-1].split(".")[0]

    tt_data_dir = DATA_DIR / filepath / "tt_data"
    model_dir = DATA_DIR / filepath / "models"

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
            df = pf.get_results(gen=eval_gen, write_output=False)
            q_val_cols = [c for c in df.columns if "q-value_" in c]
            data = []
            for x in np.logspace(-4, -1, 100):
                for engine in q_val_cols:
                    data.append(
                        [
                            x,
                            engine.replace("q-value_", ""),
                            len(df[df[engine] <= x]),
                            epoch,
                            fold,
                        ]
                    )
            df = pd.DataFrame(
                data, columns=["q-value threshold", "Engine", "n PSMs", "epoch", "fold"]
            )
            results.append(df)

    plt_df = pd.concat(results)
    plt_df = plt_df.groupby(["epoch", "q-value threshold", "Engine"]).sum().drop(columns=["fold"]).reset_index()

    epochs = plt_df["epoch"].unique()
    for epoch in epochs:
        sns.lineplot(
            data=plt_df[plt_df["epoch"] == epoch],
            x="q-value threshold",
            y="n PSMs",
            hue="Engine",
            palette=PALETTE,
        )
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=plt_df["epoch"],
        y=plt_df.loc[plt_df["q-value threshold"] == 0.01, "n PSMs"],
        hue=plt_df["Engine"],
        marker="o",
        color="black",
        palette=PALETTE,
    )

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Total PSMs", fontsize=14)
    plt.title("Total PSMs per Iteration", fontsize=16)
    plt.show()
