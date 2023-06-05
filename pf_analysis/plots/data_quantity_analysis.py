import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path("../../data")

    df = pd.read_csv(
        data_dir / "results_data_quantity_analysis.csv", header=0, index_col=0
    )
    df["q_cut_log10"] = df["q_cut"].apply(lambda x: -1 * np.log10(x))
    df["q_cut_log10_category"] = pd.cut(df["q_cut_log10"], bins=3, labels=[1, 2, 3])
    color_discrete_map = {1: "rgb(255,0,0)", 2: "rgb(255,0,255)", 3: "rgb(0,0,255)"}

    fig = px.scatter(
        df,
        x="n_spectra",
        y="n_psms_1%",
        color="q_cut_log10_category",
        hover_data=["n_psms_1%", "n_estimators", "max_depth"],
        facet_row="max_depth",
        facet_col="n_estimators",
        color_discrete_map=color_discrete_map,
    )
    fig.show()
