import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"CSV読み込み失敗: {e}")

    required_columns = [
        "patient_id",
        "name",
        "disease",
        "start_date",
        "fim_initial",
        "fim_current",
        "therapist"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"不足カラム: {missing}")

    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["start_date"] = pd.to_datetime(
        df["start_date"], errors="coerce"
    )

    df["fim_initial"] = pd.to_numeric(
        df["fim_initial"], errors="coerce"
    )

    df["fim_current"] = pd.to_numeric(
        df["fim_current"], errors="coerce"
    )

    df = df.dropna(
        subset=["start_date", "fim_initial", "fim_current"]
    )

    df = df[df["fim_initial"] > 0]

    return df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today()

    df["fim_gain"] = df["fim_current"] - df["fim_initial"]

    df["improvement_rate"] = (
        df["fim_gain"] / df["fim_initial"]
    )

    df["elapsed_days"] = (
        today - df["start_date"]
    ).dt.days

    return df


def therapist_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("therapist").agg(
        case_count=("patient_id", "count"),
        avg_improvement_rate=("improvement_rate", "mean"),
        median_improvement=("improvement_rate", "median"),
        avg_fim_gain=("fim_gain", "mean")
    ).reset_index()

    return summary


def disease_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("disease").agg(
        case_count=("patient_id", "count"),
        avg_improvement_rate=("improvement_rate", "mean"),
        avg_fim_gain=("fim_gain", "mean")
    ).reset_index()

    return summary


def save_outputs(
    df,
    therapist_df,
    disease_df,
    outdir
):
    Path(outdir).mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        f"{outdir}/patient_progress.csv",
        index=False
    )

    therapist_df.to_csv(
        f"{outdir}/therapist_summary.csv",
        index=False
    )

    disease_df.to_csv(
        f"{outdir}/disease_summary.csv",
        index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rehab Progress Analyzer"
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--outdir",
        default="outputs"
    )

    args = parser.parse_args()

    df = load_data(args.input)
    df = validate_and_clean(df)
    df = calculate_metrics(df)

    therapist_df = therapist_summary(df)
    disease_df = disease_summary(df)

    save_outputs(
        df,
        therapist_df,
        disease_df,
        args.outdir
    )

    print("✅ 分析完了")


if __name__ == "__main__":
    main()