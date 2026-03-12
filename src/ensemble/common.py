import os
import numpy as np
import pandas as pd


def ensemble_probabilities(prob_list, method="simple", weights=None):
    if len(prob_list) == 0:
        raise ValueError("prob_list is empty")

    prob_list = [np.asarray(p, dtype=np.float32) for p in prob_list]

    lengths = [len(p) for p in prob_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"Probability length mismatch: {lengths}")

    if method == "simple":
        return np.mean(prob_list, axis=0)

    if method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted ensemble")
        if len(weights) != len(prob_list):
            raise ValueError(
                f"weights length {len(weights)} != num models {len(prob_list)}"
            )

        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return sum(w * p for w, p in zip(weights, prob_list))

    raise ValueError(f"Unsupported ensemble method: {method}")


def save_binary_submission_by_language(
    dataframe: pd.DataFrame,
    output_dir: str,
    prediction_col: str = "prediction",
    output_label_col: str = "polarization",
):
    os.makedirs(output_dir, exist_ok=True)

    for lang_code in sorted(dataframe["lang"].unique()):
        out_df = dataframe[dataframe["lang"] == lang_code][["id", prediction_col]].copy()
        out_df.columns = ["id", output_label_col]

        out_path = os.path.join(output_dir, f"pred_{lang_code}.csv")
        out_df.to_csv(out_path, index=False)

        n_pos = int(out_df[output_label_col].sum())
        total = len(out_df)
        ratio = (n_pos / total * 100.0) if total > 0 else 0.0

        print(
            f"{lang_code}: saved {out_path} | "
            f"N={total:>6} | Pos={n_pos:>5} ({ratio:.2f}%)"
        )

    print(f"\nEnsemble submission saved to: {output_dir}")