import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.cli import LightningCLI


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def analyze_domain_adaptation(log_dirs):
    results = []

    excluded_dirs = [
        os.path.join("log", "tfsepdualnet_train")
    ]

    valid_log_dirs = [
        log_dir
        for log_dir in log_dirs
        if log_dir not in excluded_dirs
    ]

    for base_dir in valid_log_dirs:
        metric_paths = glob(
            os.path.join(base_dir, "version_*/metrics.csv"),
            recursive=True,
        )

        if not metric_paths:
            print(f"Warning: no metrics.csv found under {base_dir}; skipping.")
            continue

        metric_path = max(
            metric_paths,
            key=lambda path: int(
                os.path.basename(
                    os.path.dirname(path)
                ).split("_")[-1]
            ),
        )

        metrics = pd.read_csv(metric_path)

        valid_val_rows = metrics[
            metrics["source_acc"].notna()
            & metrics["step"].notna()
        ]

        if valid_val_rows.empty:
            print(
                f"Warning: no valid validation rows found in "
                f"{metric_path}; skipping."
            )
            continue

        valid_train_rows = metrics[
            metrics["train/total_loss"].notna()
        ]

        final_train_loss = (
            valid_train_rows["train/total_loss"].iloc[-1]
            if not valid_train_rows.empty
            else 0.0
        )

        last_val_epoch = valid_val_rows.iloc[-1]
        experiment_name = os.path.basename(base_dir)

        results.append(
            {
                "experiment": experiment_name,
                "source_acc": last_val_epoch["source_acc"],
                "target_acc": last_val_epoch["target_acc"],
                "domain_gap": last_val_epoch["domain_gap"],
                "final_val_acc": last_val_epoch["val_acc"],
                "final_train_loss": final_train_loss,
            }
        )

    df = pd.DataFrame(results)

    if df.empty:
        print("No valid experiment data found.")
        return df

    fig, ax = plt.subplots(figsize=(12, 6))

    df.plot(
        x="experiment",
        y=["source_acc", "target_acc"],
        kind="bar",
        ax=ax,
        color=["#1f77b4", "#ff7f0e"],
        width=0.6,
    )

    ax2 = ax.twinx()

    df.plot(
        x="experiment",
        y="domain_gap",
        kind="line",
        ax=ax2,
        color="red",
        marker="o",
        linewidth=2,
    )

    ax.set_ylabel("Accuracy")
    ax2.set_ylabel(
        "Domain Gap (Source - Target)",
        color="red",
    )

    ax.set_title("Domain Adaptation Performance")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax.set_ylim(0, 1)

    gap_min = df["domain_gap"].min()
    gap_max = df["domain_gap"].max()
    gap_padding = max(
        0.01,
        (gap_max - gap_min) * 0.1,
    )

    ax2.set_ylim(
        gap_min - gap_padding,
        gap_max + gap_padding,
    )

    plt.tight_layout()
    plt.savefig(
        "domain_adaptation_comparison.png",
        dpi=300,
    )

    return df


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        log_dirs = glob("log/tfsepdualnet_*")
        result_df = analyze_domain_adaptation(log_dirs)
        print(result_df)
    else:
        LightningCLI()