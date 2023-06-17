import json
import os
import multiprocessing as mp
from pathlib import Path
from uuid import uuid4
import datetime

from google.cloud import storage
from google.cloud import secretmanager

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


def download_bucket(bucket_name, destination_directory):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        destination_file_name = os.path.join(destination_directory, blob.name)
        blob.download_to_filename(destination_file_name)
        print(f"Blob {blob.name} downloaded to {destination_file_name}.")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


def run(config):
    mp.freeze_support()
    mp.set_start_method("fork", force=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = DATA_DIR / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    if 'GCP_PROJECT' in os.environ:
        download_bucket('peptide-forest-raw-data', str(DATA_DIR))

    output = dir_path / f"{uuid4()}_output.csv"
    pf = peptide_forest.PeptideForest(
        config_path="config_files/config_local_export_models.json",  # args.c
        output=output,  # args.o,
        memory_limit=None,  # args.m,
        max_mp_count=1,  # args.mp_limit,
    )
    pf.config = peptide_forest.pf_config.PFConfig(config)
    pf.boost(
        write_results=False,
        dump_train_test_data=True,
        eval_test_set=False,
        drop_used_spectra=False,
    )
    f = list(pf.spectrum_index.keys())[0]
    filepath = list(pf.spectrum_index.keys())[0].split("/")[-1].split(".")[0]

    tt_data_dir = dir_path / filepath / "tt_data"
    model_dir = dir_path / filepath / "models"

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
    plt_df = (
        plt_df.groupby(["epoch", "q-value threshold", "Engine"])
        .sum()
        .drop(columns=["fold"])
        .reset_index()
    )

    plt_df.to_csv(dir_path / "results.csv", index=False)

    # save results to bucket if deployed
    if 'GCP_PROJECT' in os.environ:
        upload_blob(
            "pf-results",
            str(dir_path / "results.csv"),
            str(dir_path / "results.csv"),
        )
        for file in model_dir.glob("*.json"):
            upload_blob(
                "pf-results",
                str(file),
                str(file),
            )
        for file in tt_data_dir.glob("*.json"):
            upload_blob(
                "pf-results",
                str(file),
                str(file),
            )
