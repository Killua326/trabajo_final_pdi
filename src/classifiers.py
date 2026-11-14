"""
Clasificación con descriptores clásicos:
  - SVM (RBF) y Random Forest sobre color+textura+forma.
  - Tercera vista: SVM sobre embeddings ResNet-18 (512-d).
  - Ensemble por votación suave (media de probabilidades de las 3 vistas)
    y stacking (Regresión Logística sobre las probabilidades).
Métricas: accuracy, balanced accuracy, macro-F1, AUC macro-OVR, matriz de
confusión. Validación de hiperparámetros con CV estratificada de 5 folds en
train; evaluación final en test.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import IDX_TO_CLASS, PROCESSED_DIR, RESULTS_DIR, list_dataset, stratified_split  # noqa: E402
from features import extract_dataset_features  # noqa: E402
from features_deep import extract_embeddings, load_backbone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             classification_report, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def evaluate(model, X, y, name):
    pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro") if proba is not None else float("nan")
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auc_ovr": auc,
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def svm_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])


def main(quick=False):
    rng = np.random.RandomState(42)
    prep_dir = PROCESSED_DIR / "preprocessed"
    splits = stratified_split()

    # ------------------- rasgos clásicos -------------------
    print("Extrayendo rasgos clásicos (color+textura+forma)...")
    X_tr, y_tr, ids_tr = extract_dataset_features(splits["train"], prep_dir)
    X_va, y_va, ids_va = extract_dataset_features(splits["val"], prep_dir)
    X_te, y_te, ids_te = extract_dataset_features(splits["test"], prep_dir)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_vas, X_tes = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    # ------------------- embeddings CNN -------------------
    print("Extrayendo embeddings ResNet-18...")
    net = load_backbone()
    def emb_of(ids):
        X = []
        for img_id in ids:
            img, mask = None, None
            from data_utils import load_image_and_mask
            img, mask = load_image_and_mask(img_id, preprocessed_dir=prep_dir)
            X.append(extract_embeddings(net, img, mask))
        return np.vstack(X)
    E_tr, E_va, E_te = emb_of(ids_tr), emb_of(ids_va), emb_of(ids_te)
    e_scaler = StandardScaler().fit(E_tr)
    E_trs, E_vas, E_tes = e_scaler.transform(E_tr), e_scaler.transform(E_va), e_scaler.transform(E_te)

    # ------------------- SVM sobre rasgos clásicos -------------------
    print("SVM: búsqueda de hiperparámetros (CV 5 folds en train)...")
    param_grid = {"svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.001, 0.01]}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    gs = GridSearchCV(svm_pipeline(), param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(X_trs, y_tr)
    svm = gs.best_estimator_
    print(f"  mejor: C={gs.best_params_['svm__C']}, gamma={gs.best_params_['svm__gamma']} "
          f"(CV bacc={gs.best_score_:.3f})")
    svm.fit(np.vstack([X_trs, X_vas]), np.concatenate([y_tr, y_va]))

    # ------------------- Random Forest -------------------
    print("Random Forest: búsqueda de hiperparámetros...")
    rf_grid = {"n_estimators": [300], "max_depth": [None, 12], "min_samples_leaf": [1, 3]}
    gs_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_grid,
                         cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    gs_rf.fit(X_tr, y_tr)
    rf = gs_rf.best_estimator_
    print(f"  mejor: {gs_rf.best_params_} (CV bacc={gs_rf.best_score_:.3f})")
    rf.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))

    # ------------------- SVM sobre embeddings -------------------
    esvm = svm_pipeline()
    esvm.fit(np.vstack([E_trs, E_vas]), np.concatenate([y_tr, y_va]))

    # ------------------- ensemble -------------------
    stack_features_tr = np.hstack([svm.predict_proba(X_trs), rf.predict_proba(X_tr), esvm.predict_proba(E_trs)])
    stack_features_va = np.hstack([svm.predict_proba(X_vas), rf.predict_proba(X_va), esvm.predict_proba(E_vas)])
    stack_features_te = np.hstack([svm.predict_proba(X_tes), rf.predict_proba(X_te), esvm.predict_proba(E_tes)])
    stack = LogisticRegression(max_iter=2000).fit(np.vstack([stack_features_tr, stack_features_va]),
                                                  np.concatenate([y_tr, y_va]))
    soft_vote = lambda Xs, Xr, Xe: (svm.predict_proba(Xs) + rf.predict_proba(Xr) + esvm.predict_proba(Xe)) / 3.0

    # ------------------- evaluación en test -------------------
    results = {
        "svm_features": evaluate(svm, X_tes, y_te, "SVM (rasgos clásicos)"),
        "rf_features": evaluate(rf, X_te, y_te, "Random Forest (rasgos clásicos)"),
        "svm_embeddings": evaluate(esvm, E_tes, y_te, "SVM (embeddings ResNet18)"),
        "ensemble_soft": {
            "model": "Ensemble (votación suave)",
            "accuracy": accuracy_score(y_te, soft_vote(X_tes, X_te, E_tes).argmax(1)),
            "balanced_accuracy": balanced_accuracy_score(y_te, soft_vote(X_tes, X_te, E_tes).argmax(1)),
            "macro_f1": f1_score(y_te, soft_vote(X_tes, X_te, E_tes).argmax(1), average="macro"),
            "auc_ovr": roc_auc_score(y_te, soft_vote(X_tes, X_te, E_tes), multi_class="ovr", average="macro"),
            "confusion": confusion_matrix(y_te, soft_vote(X_tes, X_te, E_tes).argmax(1)).tolist(),
        },
        "ensemble_stack": evaluate(stack, stack_features_te, y_te, "Ensemble (stacking LR)"),
    }

    out_dir = RESULTS_DIR / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "classic_metrics.json").open("w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in results.items()}, f, indent=2)
    with (out_dir / "classic_confusion.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo"] + [f"pred_{IDX_TO_CLASS[i]}" for i in range(3)])
        for k, v in results.items():
            w.writerow([v["model"]] + np.sum(v["confusion"], axis=0).tolist())

    print("\n===== RESULTADOS EN TEST =====")
    for k, v in results.items():
        print(f"{v['model']:34s} acc={v['accuracy']:.3f} | bacc={v['balanced_accuracy']:.3f} | "
              f"macroF1={v['macro_f1']:.3f} | AUC={v['auc_ovr']:.3f}")
    print("\nReporte SVM+rasgos clásicos:")
    print(classification_report(y_te, svm.predict(X_tes), target_names=list(IDX_TO_CLASS.values())))
    print("Matriz de confusión (ensemble stacking):")
    print(np.array(results["ensemble_stack"]["confusion"]))

    # Persistir modelos para run_pipeline.py (explicabilidad e inferencia)
    import joblib

    mp = ROOT / "models"
    mp.mkdir(exist_ok=True)
    joblib.dump(scaler, mp / "classic_scaler.joblib")
    joblib.dump(svm, mp / "svm_features.joblib")
    joblib.dump(rf, mp / "rf_features.joblib")
    joblib.dump(e_scaler, mp / "embed_scaler.joblib")
    joblib.dump(esvm, mp / "svm_embeddings.joblib")
    print(f"\nModelos guardados en {mp}/")

    # importancia de rasgos del RF (para explicabilidad)
    importances = rf.feature_importances_
    # nombres sobre una máscara NO vacía para que la lista sea la real
    _, names = extract_all(
        np.full((16, 16, 3), 128, np.uint8), np.ones((16, 16), np.uint8)
    )
    order = np.argsort(importances)[::-1][:20]
    print("\nTop-20 rasgos (Random Forest):")
    for i in order:
        print(f"  {names[i]:28s} {importances[i]:.4f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="grid reducido")
    args = ap.parse_args()
    main(quick=args.quick)
