import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup          import BINARY_LABELS, MULTICLASS_LABELS, RANDOM_STATE
from phase4_data_splitting import load_binary_data, load_multiclass_data

SHAP_SAMPLE_SIZE = 500

def load_best_model_names() -> tuple:
    path = "data/processed/best_model_names.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("best_model_names.csv not found. Run Phase 7 first.")
    s = pd.read_csv(path, header=None, index_col=0).squeeze()
    return str(s["best_binary"]), str(s["best_multiclass"])

def _is_tree_model(model) -> bool:
    from sklearn.ensemble import RandomForestClassifier
    import lightgbm as lgb
    return isinstance(model, (RandomForestClassifier, lgb.LGBMClassifier))

def _uses_scaled(model_name: str) -> bool:
    return "logreg" in model_name

# ══════════════════════════════════════════════════════════
# FEATURE SELECTION & IMPORTANCE EXPORTS
# ══════════════════════════════════════════════════════════

def export_shap_importance(shap_values, feature_names: list, save_name: str):
    """Calculates mean(|SHAP|) and exports to CSV for feature selection analysis."""
    # If 3D (samples, features, classes), take mean absolute across samples and classes
    if len(shap_values.values.shape) == 3:
        importances = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        importances = np.abs(shap_values.values).mean(axis=0)
    
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values(by="importance", ascending=False)
    
    os.makedirs("results/tables", exist_ok=True)
    path = f"results/tables/shap_importance_{save_name}.csv"
    df.to_csv(path, index=False)
    print(f"💾 SHAP importance CSV (for feature selection) → {path}")

def plot_feature_importance(model, feature_names: list, title: str, save_name: str, top_n: int = 25):
    if not hasattr(model, "feature_importances_"):
        print(f"    ⚠️  {title}: no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    top_vals = top_vals / top_vals.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Relative Importance (%)")
    ax.set_title(f"Top {top_n} Feature Importances\n{title}", fontweight="bold")
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    path = f"results/figures/feature_importance_{save_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Feature importance plot → {path}")

# ══════════════════════════════════════════════════════════
# SHAP VISUALIZATIONS
# ══════════════════════════════════════════════════════════

def compute_shap(model, X_sample: np.ndarray, feature_names: list, task_label: str) -> shap.Explanation:
    print(f"\n🔍 Computing SHAP values for {task_label}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values

def plot_shap_summary(shap_values, X_sample: np.ndarray, feature_names: list, task_label: str, save_name: str, class_idx: int = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    sv = shap_values
    if class_idx is not None and len(shap_values.values.shape) == 3:
        sv_vals = shap_values.values[:, :, class_idx]
        sv = shap.Explanation(
            values=sv_vals,
            base_values=shap_values.base_values[:, class_idx] if len(shap_values.base_values.shape) > 1 else shap_values.base_values,
            data=shap_values.data,
            feature_names=feature_names
        )

    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.title(f"SHAP Summary Plot\n{task_label}", fontweight="bold")
    plt.tight_layout()
    path = f"results/figures/shap_summary_{save_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_shap_dependence(shap_values, X_sample: np.ndarray, feature_names: list, feature: str, task_label: str, save_name: str, class_idx: int = None):
    if feature not in feature_names:
        print(f"    ⚠️  Feature '{feature}' not found — skipping.")
        return

    feat_idx = feature_names.index(feature)
    sv_vals = shap_values.values
    
    # Fix: Ensure we slice the 3D array (samples, features, classes) correctly
    if len(sv_vals.shape) == 3:
        idx = class_idx if class_idx is not None else 1
        sv_vals = sv_vals[:, :, idx]

    feat_vals = X_sample[:, feat_idx]
    shap_col = sv_vals[:, feat_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(feat_vals, shap_col, alpha=0.4, s=15, c=shap_col, cmap="RdBu_r")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.colorbar(scatter, ax=ax, label="SHAP value")
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value (impact)")
    ax.set_title(f"SHAP Dependence: {feature}\n{task_label}", fontweight="bold")
    plt.tight_layout()
    path = f"results/figures/shap_dependence_{save_name}_{feature}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_shap_force_examples(model, X_sample: np.ndarray, y_true: np.ndarray, feature_names: list, task_label: str, save_name: str, n_examples: int = 3):
    explainer = shap.TreeExplainer(model)
    y_pred = model.predict(X_sample)
    correct_idx = np.where(y_pred == y_true)[0]
    incorrect_idx = np.where(y_pred != y_true)[0]

    for tag, indices in [("correct", correct_idx), ("incorrect", incorrect_idx)]:
        for rank, idx in enumerate(indices[:n_examples]):
            sv_single = explainer(X_sample[idx:idx+1])
            if len(sv_single.values.shape) == 3:
                cls_p = int(y_pred[idx])
                sv_plot = shap.Explanation(
                    values=sv_single.values[0, :, cls_p],
                    base_values=sv_single.base_values[0, cls_p] if sv_single.base_values.ndim > 1 else sv_single.base_values[0],
                    data=sv_single.data[0], feature_names=feature_names
                )
            else:
                sv_plot = sv_single[0]

            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(sv_plot, max_display=15, show=False)
            plt.title(f"Waterfall — {tag} #{rank+1}\nTrue: {y_true[idx]} Pred: {y_pred[idx]}", fontsize=9)
            plt.savefig(f"results/figures/shap_waterfall_{save_name}_{tag}_{rank+1}.png", bbox_inches="tight")
            plt.close()

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60); print("PHASE 8 — EXPLAINABILITY"); print("=" * 60)

    best_bin_name, best_mc_name = load_best_model_names()
    
    (X_tr_sc, X_tr_un, X_te_sc, X_te_un, y_tr_bin, y_te_bin, feat_names) = load_binary_data()
    (_, _, X_te_sc_mc, X_te_un_mc, y_tr_mc, y_te_mc, _) = load_multiclass_data()

    bin_model = joblib.load(f"models/binary_{best_bin_name}.joblib")
    mc_model = joblib.load(f"models/{best_mc_name}.joblib")

    X_te_bin = X_te_sc if _uses_scaled(best_bin_name) else X_te_un
    X_te_mc = X_te_sc_mc if _uses_scaled(best_mc_name) else X_te_un_mc

    rng = np.random.default_rng(RANDOM_STATE)
    idx_b = rng.choice(len(X_te_bin), min(SHAP_SAMPLE_SIZE, len(X_te_bin)), replace=False)
    idx_m = rng.choice(len(X_te_mc), min(SHAP_SAMPLE_SIZE, len(X_te_mc)), replace=False)

    # BINARY
    print("\n--- BINARY EXPLAINABILITY ---")
    plot_feature_importance(bin_model, feat_names, f"Binary: {best_bin_name}", f"binary_{best_bin_name}")
    if _is_tree_model(bin_model):
        shap_b = compute_shap(bin_model, X_te_bin[idx_b].astype(float), feat_names, "Binary")
        export_shap_importance(shap_b, feat_names, f"binary_{best_bin_name}")
        plot_shap_summary(shap_b, X_te_bin[idx_b], feat_names, "Binary", f"binary_{best_bin_name}")
        # Note: Updated feature names to match OHE (e.g., _1 suffix)
        for f in ["lcDispositionDirection_1", "issueArea_2", "certReason_2"]:
            plot_shap_dependence(shap_b, X_te_bin[idx_b], feat_names, f, "Binary", f"binary_{best_bin_name}")
        plot_shap_force_examples(bin_model, X_te_bin[idx_b], y_te_bin[idx_b], feat_names, "Binary", f"binary_{best_bin_name}")

    # MULTICLASS
    print("\n--- MULTICLASS EXPLAINABILITY ---")
    plot_feature_importance(mc_model, feat_names, f"Multiclass: {best_mc_name}", f"mc_{best_mc_name}")
    if _is_tree_model(mc_model):
        shap_m = compute_shap(mc_model, X_te_mc[idx_m].astype(float), feat_names, "Multiclass")
        export_shap_importance(shap_m, feat_names, f"mc_{best_mc_name}")
        for c in [0, 1]:
            plot_shap_summary(shap_m, X_te_mc[idx_m], feat_names, f"Class {c}", f"mc_{best_mc_name}_c{c}", class_idx=c)
    else:
        print("Note: Best multiclass model is not tree-based; skipping SHAP.")

    print("\n✅ Phase 8 complete.")

if __name__ == "__main__":
    main()