"""
Evaluación del clasificador (requisito 6):
  1) Conjunto de test (split estratificado 60/20/20): curvas ROC por clase
     (one-vs-rest) y AUC por clase para cada vista del clasificador (SVM
     rasgos, RF rasgos, SVM embeddings, CNN), matrices de confusión por clase.
  2) Validación cruzada ESTRATIFICADA de K folds sobre TODO el dataset:
     - modelos clásicos (SVM/RF sobre rasgos): siempre disponible.
     - CNN (transferencia+fine-tuning) con --deep (caro en CPU; épocas
       reducidas por fold). Las probabilidades fuera-de-fold permiten ROC
       por clase con todo el dataset.

Salidas en results/classification/:
  - roc_curves.png / roc_auc_per_class.json   (test, por modelo y clase)
  - confusion_matrices.png / .csv             (test, por modelo)
  - cv_classical.json / cv_deep.json          (CV: folds, media ± std, OOF)
  - cv_roc_curves.png / cv_confusion.png      (ROC y confusión de las
                                               probabilidades out-of-fold)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import CLASSES, IDX_TO_CLASS, RESULTS_DIR, list_dataset, stratified_split  # noqa: E402

DEVICE = "cpu"
N_CLASSES = len(CLASSES)


# ------------------------------------------------------------------ ROC ---
def per_class_roc(y_true, probs):
    """Curvas y AUC one-vs-rest por clase.

    Returns: {class_idx: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true)
    out = {}
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out[c] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan")}
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        out[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y_bin, probs[:, c]))}
    return out


def plot_roc_curves(roc_by_model, path, title):
    """Una subtrama por modelo, una curva por clase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(roc_by_model)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (model, roc) in zip(axes, roc_by_model.items()):
        for c in range(N_CLASSES):
            r = roc[c]
            ax.plot(r["fpr"], r["tpr"], lw=2, label=f"{IDX_TO_CLASS[c]} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR (1 - especificidad)")
        ax.set_ylabel("TPR (sensibilidad)")
        ax.set_title(model)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, path, title, model_names=None):
    """Mapas de calor de matrices de confusión (por modelo o una sola)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if cm.ndim == 2:
        cm = cm[None]
    names = model_names or ["modelo"]
    n = len(cm)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m, name in zip(axes, cm, names):
        m = np.asarray(m, float)
        row = m.sum(axis=1, keepdims=True)
        pct = m / np.maximum(row, 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{int(m[i, j])}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=11)
        ax.set_xticks(range(N_CLASSES), list(IDX_TO_CLASS.values()), rotation=20, ha="right")
        ax.set_yticks(range(N_CLASSES), list(IDX_TO_CLASS.values()))
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        ax.set_title(name)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ modelos de la vista CNN ---
def cnn_predict_proba(img_ids, arch="resnet18"):
    """Probabilidades de la CNN entrenada para una lista de ids."""
    import joblib  # noqa: F401  (consistencia de imports)
    import torch
    from classifier_deep import LesionDataset, build_model
    from torch.utils.data import DataLoader

    ckpt = torch.load(ROOT / "models" / f"classifier_{arch}.pt", map_location=DEVICE, weights_only=False)
    net = build_model(arch).to(DEVICE).eval()
    net.load_state_dict(ckpt["state"])
    loader = DataLoader(LesionDataset(list(img_ids), augment=False), batch_size=16)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs += torch.softmax(net(x.to(DEVICE)), 1).cpu().numpy().tolist()
    return np.array(probs)


def classic_features(ids):
    """Rasgos clásicos y embeddings para una lista de ids (usando GT)."""
    from data_utils import load_image_and_mask
    from features import extract_dataset_features
    from features_deep import extract_embeddings, load_backbone

    prep = ROOT / "data" / "processed" / "preprocessed"
    X, y, used = extract_dataset_features(ids, prep)
    net = load_backbone()
    E = []
    for img_id in used:
        img, mask = load_image_and_mask(img_id, preprocessed_dir="preprocessed")
        E.append(extract_embeddings(net, img, mask))
    return X, y, np.vstack(E), used


def load_classic_models():
    import joblib

    mp = ROOT / "models"
    return (joblib.load(mp / "classic_scaler.joblib"), joblib.load(mp / "svm_features.joblib"),
            joblib.load(mp / "rf_features.joblib"), joblib.load(mp / "embed_scaler.joblib"),
            joblib.load(mp / "svm_embeddings.joblib"))


# --------------------------------------------------------- 1) split test ---
def eval_test_split():
    out = RESULTS_DIR / "classification"
    out.mkdir(parents=True, exist_ok=True)
    splits = stratified_split()
    X, y, E, used = classic_features(splits["test"])
    scaler, svm, rf, e_scaler, esvm = load_classic_models()

    models_probs = {
        "SVM (rasgos clásicos)": svm.predict_proba(scaler.transform(X)),
        "Random Forest (rasgos)": rf.predict_proba(X),
        "SVM (embeddings ResNet18)": esvm.predict_proba(e_scaler.transform(E)),
        "CNN ResNet-18 (fine-tuned)": cnn_predict_proba(used),
    }

    # ---- ROC por clase ----
    roc_by_model = {}
    auc_json = {}
    for name, probs in models_probs.items():
        roc_by_model[name] = per_class_roc(y, probs)
        auc_json[name] = {IDX_TO_CLASS[c]: roc_by_model[name][c]["auc"] for c in range(N_CLASSES)}
        auc_json[name]["macro"] = float(np.nanmean([roc_by_model[name][c]["auc"] for c in range(N_CLASSES)]))
    plot_roc_curves(roc_by_model, out / "roc_curves.png",
                    "Curvas ROC one-vs-rest por clase — split de test (n=%d)" % len(y))
    (out / "roc_auc_per_class.json").write_text(json.dumps(auc_json, indent=2))

    # ---- matrices de confusión por clase ----
    cms, names = [], []
    for name, probs in models_probs.items():
        pred = probs.argmax(1)
        from sklearn.metrics import confusion_matrix
        cms.append(confusion_matrix(y, pred, labels=range(N_CLASSES)))
        names.append(name)
    plot_confusion(np.array(cms), out / "confusion_matrices.png",
                   "Matrices de confusión por clase — split de test", names)
    with (out / "confusion_matrices.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "clase_real"] + [f"pred_{IDX_TO_CLASS[i]}" for i in range(N_CLASSES)])
        for name, m in zip(names, cms):
            for i in range(N_CLASSES):
                w.writerow([name, IDX_TO_CLASS[i]] + m[i].tolist())

    print("=== Test (n=%d) — AUC por clase ===" % len(y))
    for name in models_probs:
        print(f"  {name:30s} " + " ".join(f"{IDX_TO_CLASS[c]}={auc_json[name][IDX_TO_CLASS[c]]:.3f}"
                                          for c in range(N_CLASSES))
              + f" | macro={auc_json[name]['macro']:.3f}")
    print(f"Salidas -> {out}/ (roc_curves.png, confusion_matrices.png, *.json/csv)")
    return auc_json


# ------------------------------------------- 2) CV estratificada clásica ---
def cv_classical(k=5, seed=42):
    """CV estratificada de K folds para SVM/RF sobre rasgos (todo el dataset)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = RESULTS_DIR / "classification"
    ids_all, labels = list_dataset()
    X, y, _, used = classic_features(ids_all)

    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    models = {
        "SVM (rasgos clásicos)": lambda: Pipeline([("scaler", StandardScaler()),
                                                   ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))]),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    results = {}
    oof = {name: np.zeros((len(y), N_CLASSES)) for name in models}
    for name, mk in models.items():
        fold_rows, y_pred_all = [], np.zeros(len(y), int)
        for f, (tr, te) in enumerate(cv.split(X, y), 1):
            m = mk().fit(X[tr], y[tr])
            pred, proba = m.predict(X[te]), m.predict_proba(X[te])
            oof[name][te] = proba
            y_pred_all[te] = pred
            fold_rows.append({
                "fold": f,
                "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                "auc_ovr_macro": roc_auc_score(y[te], proba, multi_class="ovr", average="macro"),
            })
        results[name] = {
            "folds": fold_rows,
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in fold_rows])),
            "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in fold_rows])),
            "auc_ovr_macro_mean": float(np.mean([r["auc_ovr_macro"] for r in fold_rows])),
            "auc_ovr_macro_std": float(np.std([r["auc_ovr_macro"] for r in fold_rows])),
            "confusion_oof": confusion_matrix(y, y_pred_all, labels=range(N_CLASSES)).tolist(),
        }
        print(f"[{name}] CV bAcc={results[name]['balanced_accuracy_mean']:.3f} "
              f"± {results[name]['balanced_accuracy_std']:.3f} | "
              f"AUC={results[name]['auc_ovr_macro_mean']:.3f} ± {results[name]['auc_ovr_macro_std']:.3f}")

    (out / "cv_classical.json").write_text(json.dumps(results, indent=2))

    # ROC y confusión con las probabilidades out-of-fold (todo el dataset)
    roc_by_model = {name: per_class_roc(y, oof[name]) for name in models}
    plot_roc_curves(roc_by_model, out / "cv_roc_curves.png",
                    f"ROC por clase (one-vs-rest) — CV estratificada {k} folds, out-of-fold (n={len(y)})")
    plot_confusion(np.array([results[n]["confusion_oof"] for n in models]), out / "cv_confusion.png",
                   f"Matriz de confusión de predicciones out-of-fold (CV {k} folds)", list(models))
    print(f"Salidas -> {out}/ (cv_classical.json, cv_roc_curves.png, cv_confusion.png)")
    return results


# ----------------------------------------------- 3) CV estratificada CNN ---
def cv_deep(k=5, epochs=6, batch_size=16, arch="resnet18", seed=42):
    """CV estratificada de K folds para la CNN (transferencia+fine-tuning).

    Entrena una red desde cero por fold con épocas reducidas (CPU). Reusa el
    Dataset y el esquema de entrenamiento de classifier_deep.py.
    """
    import torch
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    from classifier_deep import LesionDataset, build_model, unfreeze_last_stage

    torch.manual_seed(seed)
    np.random.seed(seed)

    out = RESULTS_DIR / "classification"
    _, labels = list_dataset()
    ids_all, y_all = [], []
    for img_id, c in sorted(labels.items()):
        ids_all.append(img_id)
        y_all.append(int(c))
    y_all = np.array(y_all)

    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    head_epochs = max(1, epochs // 3)
    fold_rows, oof = [], np.zeros((len(y_all), N_CLASSES))
    for f, (tr, te) in enumerate(cv.split(ids_all, y_all), 1):
        ids_tr = [ids_all[i] for i in tr]
        ids_te = [ids_all[i] for i in te]
        tr_loader = DataLoader(LesionDataset(ids_tr, augment=True), batch_size, shuffle=True)
        va_loader = DataLoader(LesionDataset(ids_te), batch_size)

        model = build_model(arch).to(DEVICE)
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = None
        for ep in range(1, epochs + 1):
            if ep == head_epochs + 1:
                unfreeze_last_stage(model, arch, pct=0.5)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                        lr=1e-4, weight_decay=1e-4)
            elif ep == 1:
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                        lr=1e-3, weight_decay=1e-4)
            model.train()
            for x, yy in tr_loader:
                opt.zero_grad()
                loss = crit(model(x.to(DEVICE)), yy.to(DEVICE))
                loss.backward()
                opt.step()

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, yy in va_loader:
                probs += torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy().tolist()
                ys += yy.tolist()
        probs, ys = np.array(probs), np.array(ys)
        oof[te] = probs
        pred = probs.argmax(1)
        fold_rows.append({
            "fold": f,
            "balanced_accuracy": balanced_accuracy_score(ys, pred),
            "auc_ovr_macro": roc_auc_score(ys, probs, multi_class="ovr", average="macro"),
        })
        print(f"[{arch}] fold {f}/{k}: bAcc={fold_rows[-1]['balanced_accuracy']:.3f} "
              f"AUC={fold_rows[-1]['auc_ovr_macro']:.3f}")
        torch.save({"arch": arch, "fold": f, "state": {kk: vv.cpu() for kk, vv in model.state_dict().items()}},
                   ROOT / "models" / f"cv_{arch}_fold{f}.pt")

    y_pred_all = oof.argmax(1)
    summary = {
        "arch": arch, "folds": k, "epochs_per_fold": epochs,
        "folds_detail": fold_rows,
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in fold_rows])),
        "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in fold_rows])),
        "auc_ovr_macro_mean": float(np.mean([r["auc_ovr_macro"] for r in fold_rows])),
        "auc_ovr_macro_std": float(np.std([r["auc_ovr_macro"] for r in fold_rows])),
        "confusion_oof": confusion_matrix(y_all, y_pred_all, labels=range(N_CLASSES)).tolist(),
    }
    (out / "cv_deep.json").write_text(json.dumps(summary, indent=2))
    # guardar probabilidades out-of-fold para regenerar figuras sin reentrenar
    np.savez(out / "cv_deep_oof.npz", y=y_all, oof=oof)

    roc = per_class_roc(y_all, oof)
    plot_roc_curves({f"CNN {arch} (CV out-of-fold)": roc}, out / "cv_deep_roc.png",
                    f"ROC por clase — CV estratificada {k} folds de la CNN (n={len(y_all)})")
    plot_confusion(summary["confusion_oof"], out / "cv_deep_confusion.png",
                   f"Confusión out-of-fold — CNN {arch}, CV {k} folds")
    print(f"[{arch}] CV bAcc={summary['balanced_accuracy_mean']:.3f} "
          f"± {summary['balanced_accuracy_std']:.3f} | "
          f"AUC={summary['auc_ovr_macro_mean']:.3f} ± {summary['auc_ovr_macro_std']:.3f}")
    print(f"Salidas -> {out}/ (cv_deep.json, cv_deep_roc.png, cv_deep_confusion.png)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-test", action="store_true", help="no reevaluar el split de test")
    ap.add_argument("--cv", type=int, default=5, help="folds de la CV clásica (0 = omitir)")
    ap.add_argument("--deep", action="store_true", help="añadir CV estratificada de la CNN (lento en CPU)")
    ap.add_argument("--deep-folds", type=int, default=5)
    ap.add_argument("--deep-epochs", type=int, default=6, help="épocas por fold de la CNN")
    ap.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    args = ap.parse_args()

    if not args.skip_test:
        eval_test_split()
    if args.cv:
        cv_classical(k=args.cv)
    if args.deep:
        cv_deep(k=args.deep_folds, epochs=args.deep_epochs, arch=args.arch)
