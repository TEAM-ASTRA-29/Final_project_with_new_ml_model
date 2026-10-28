import warnings, numpy as np, optuna, psutil, time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
                               RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCH_REG = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":       lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees":   lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Ridge":         lambda: Ridge(alpha=1.0),
    "SVR":           lambda: SVR(C=1.0),
    "KNN":           lambda: KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
}
BENCH_CLF = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":       lambda: XGBClassifier(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":   lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Reg.": lambda: LogisticRegression(max_iter=300, random_state=42),
    "SVC":           lambda: SVC(C=1.0, probability=True),
    "KNN":           lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
}

def run_benchmark(X, y, task, scale=True, cv_folds=5):
    catalog = BENCH_REG if task == "Regression" else BENCH_CLF
    scoring = "r2" if task == "Regression" else "accuracy"
    Xn = X.values if hasattr(X, "values") else np.array(X)
    yn = y.values if hasattr(y, "values") else np.array(y)
    if scale: Xn = StandardScaler().fit_transform(Xn)
    results = []
    for name, builder in catalog.items():
        try:
            sc = cross_val_score(builder(), Xn, yn, cv=cv_folds, scoring=scoring, n_jobs=-1)
            results.append({"model_name": name, "mean_score": round(float(sc.mean()), 6),
                             "std_score": round(float(sc.std()), 6)})
        except Exception:
            pass
    results.sort(key=lambda r: -r["mean_score"])
    return {"benchmark": results, "metric": "R2" if task == "Regression" else "Accuracy"}

def _make_model(name, trial, task):
    if "Random Forest" in name:
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
             "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 16, 24]),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
             "random_state": 42, "n_jobs": -1}
        return RandomForestRegressor(**p) if task == "Regression" else RandomForestClassifier(**p)
    elif "XGBoost" in name:
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
             "max_depth": trial.suggest_int("max_depth", 3, 12),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10., log=True),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10., log=True),
             "random_state": 42, "n_jobs": -1, "verbosity": 0}
        return XGBRegressor(**p) if task == "Regression" else XGBClassifier(**p, eval_metric="logloss")
    elif "Gradient" in name:
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
             "max_depth": trial.suggest_int("max_depth", 2, 8),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0), "random_state": 42}
        return GradientBoostingRegressor(**p) if task == "Regression" else GradientBoostingClassifier(**p)
    elif "Extra" in name:
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
             "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
             "random_state": 42, "n_jobs": -1}
        return ExtraTreesRegressor(**p) if task == "Regression" else ExtraTreesClassifier(**p)
    elif "Ridge" in name:
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True))
    elif name == "SVR":
        return SVR(C=trial.suggest_float("C", 0.01, 100., log=True),
                   epsilon=trial.suggest_float("epsilon", 0.01, 1.0),
                   kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]))
    elif name == "SVC":
        return SVC(C=trial.suggest_float("C", 0.01, 100., log=True),
                   kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]), probability=True)
    elif "KNN" in name:
        p = {"n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
             "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
        return KNeighborsRegressor(**p) if task == "Regression" else KNeighborsClassifier(**p)
    elif "Decision" in name:
        p = {"max_depth": trial.suggest_int("max_depth", 2, 20),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 16), "random_state": 42}
        return DecisionTreeRegressor(**p) if task == "Regression" else DecisionTreeClassifier(**p)
    elif "Logistic" in name:
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                                   solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                                   max_iter=500, random_state=42)
    else:
        nl = trial.suggest_int("n_layers", 1, 3)
        layers = tuple(trial.suggest_int(f"n_units_{i}", 32, 256) for i in range(nl))
        p = {"hidden_layer_sizes": layers,
             "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
             "learning_rate_init": trial.suggest_float("lr", 1e-4, 0.05, log=True),
             "max_iter": 500, "random_state": 42}
        return MLPRegressor(**p) if task == "Regression" else MLPClassifier(**p)

def focus_tune(model_name, X, y, task, scale=True, n_trials=60,
               timeout_s=120, cv_folds=5, progress_callback=None):
    Xn = X.values if hasattr(X, "values") else np.array(X)
    yn = y.values if hasattr(y, "values") else np.array(y)
    scoring = "r2" if task == "Regression" else "accuracy"
    if scale: Xn = StandardScaler().fit_transform(Xn)
    history = []
    def objective(trial):
        model = _make_model(model_name, trial, task)
        sc = cross_val_score(model, Xn, yn, cv=cv_folds, scoring=scoring, n_jobs=-1)
        val = float(sc.mean())
        history.append({"trial": trial.number, "score": round(val, 6)})
        if progress_callback:
            progress_callback(trial.number, n_trials, val)
        return val
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    fa = _assess_fit(model_name, study.best_params, Xn, yn, task)
    return {"model_name": model_name, "best_params": study.best_params,
            "best_cv_score": round(study.best_value, 6), "n_trials": len(study.trials),
            "trial_history": history, "fit_assessment": fa,
            "metric": "R2" if task == "Regression" else "Accuracy"}

def _assess_fit(model_name, best_params, Xn, yn, task):
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(Xn, yn, test_size=0.2, random_state=99)
        catalog = BENCH_REG if task == "Regression" else BENCH_CLF
        base = catalog.get(model_name, list(catalog.values())[0])()
        try:
            valid = {k: v for k, v in best_params.items() if hasattr(base, k)}
            base.set_params(**valid)
        except Exception:
            pass
        base.fit(X_tr, y_tr)
        if task == "Regression":
            tr_s = r2_score(y_tr, base.predict(X_tr))
            te_s = r2_score(y_te, base.predict(X_te)); baseline = 0.0
        else:
            tr_s = accuracy_score(y_tr, base.predict(X_tr))
            te_s = accuracy_score(y_te, base.predict(X_te))
            baseline = float(np.bincount(y_te.astype(int)).max()) / len(y_te)
        gap = tr_s - te_s
        if te_s < baseline + 0.02:
            verdict = "underfitting"
            rec = "Model barely beats baseline. Try more estimators, deeper trees, or a more complex model."
        elif gap > 0.20:
            verdict = "overfitting"
            rec = "Large train-test gap. Increase regularisation (alpha/C), reduce max_depth, or add more data."
        elif gap > 0.10:
            verdict = "mild_overfitting"
            rec = "Slight overfit. Consider modest regularisation increase."
        else:
            verdict = "well_fitted"
            rec = "Model generalises well. Train and test scores are close."
        return {"train_score": round(tr_s, 4), "test_score": round(te_s, 4),
                "gap": round(gap, 4), "baseline": round(baseline, 4),
                "verdict": verdict, "recommendation": rec}
    except Exception as e:
        return {"verdict": "unknown", "recommendation": str(e),
                "train_score": None, "test_score": None, "gap": None}

def train_focused_model(model_name, best_params, X_tr, X_te, y_tr, y_te, task, scale):
    X_tr2, X_te2 = X_tr.copy(), X_te.copy()
    if scale:
        sc = StandardScaler(); X_tr2 = sc.fit_transform(X_tr2); X_te2 = sc.transform(X_te2)
    catalog = BENCH_REG if task == "Regression" else BENCH_CLF
    base = catalog.get(model_name, list(catalog.values())[0])()
    try:
        valid = {k: v for k, v in best_params.items() if hasattr(base, k)}
        base.set_params(**valid)
    except Exception:
        pass
    proc = psutil.Process(); m0 = proc.memory_info().rss/(1024**2)
    t0 = time.time(); base.fit(X_tr2, y_tr)
    rt = time.time()-t0; mem = max(0, proc.memory_info().rss/(1024**2)-m0)
    fi = None
    if hasattr(base, "feature_importances_"): fi = base.feature_importances_.tolist()
    out = {"name": f"{model_name} (AutoML)", "key": "automl",
           "params": {k: str(v) for k, v in best_params.items()},
           "runtime": round(rt, 4), "memory_mb": round(mem, 2),
           "model_size_mb": 0.0, "feature_importances": fi, "automl": True}
    if task == "Regression":
        yp = base.predict(X_te2)
        out.update({"y_pred": yp.tolist(),
                    "r2": round(float(r2_score(y_te, yp)), 6),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_te, yp))), 6),
                    "mae": round(float(mean_absolute_error(y_te, yp)), 6),
                    "mse": round(float(mean_squared_error(y_te, yp)), 6)})
    else:
        yp = base.predict(X_te2); classes = list(np.unique(y_te))
        avg = "binary" if len(classes) == 2 else "weighted"
        out.update({"y_pred": yp.tolist(),
                    "accuracy": round(float(accuracy_score(y_te, yp)), 6),
                    "f1": round(float(f1_score(y_te, yp, average=avg, zero_division=0)), 6),
                    "precision": round(float(precision_score(y_te, yp, average=avg, zero_division=0)), 6),
                    "recall": round(float(recall_score(y_te, yp, average=avg, zero_division=0)), 6),
                    "auc": None, "conf_matrix": confusion_matrix(y_te, yp).tolist()})
    return out
