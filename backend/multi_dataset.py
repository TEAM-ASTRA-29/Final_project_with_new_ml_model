import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings("ignore")

class DatasetRegistry:
    def __init__(self): self.datasets = {}; self.labels = []
    def add(self, label, df):
        self.datasets[label] = df.copy()
        if label not in self.labels: self.labels.append(label)
    def remove(self, label):
        self.datasets.pop(label, None); self.labels = [l for l in self.labels if l != label]
    def get(self, label): return self.datasets.get(label)
    def all_labels(self): return self.labels
    def summary(self):
        return [{"Dataset":l,"Rows":df.shape[0],"Cols":df.shape[1],
                 "Numeric":df.select_dtypes(include=[np.number]).shape[1],
                 "Categorical":df.select_dtypes(include=["object","category"]).shape[1],
                 "Nulls":int(df.isnull().sum().sum())} for l,df in self.datasets.items()]

def find_common_columns(datasets):
    from collections import defaultdict
    col_map = defaultdict(list)
    for label, df in datasets.items():
        for col in df.columns: col_map[col.strip().lower()].append(label)
    return {col: labels for col, labels in col_map.items() if len(labels) >= 2}

def suggest_merge_keys(datasets):
    common = find_common_columns(datasets); suggestions = []
    for col, labels in common.items():
        cards = []; dtypes = []
        for lbl in labels:
            df = datasets[lbl]; matched = [c for c in df.columns if c.strip().lower() == col]
            if matched: s = df[matched[0]]; dtypes.append(str(s.dtype)); cards.append(s.nunique())
        looks_like_key = any("object" in d for d in dtypes) or (cards and max(cards) < 500)
        suggestions.append({"column":col,"in_datasets":labels,
                             "looks_like_key":looks_like_key,"cardinality":max(cards) if cards else 0})
    suggestions.sort(key=lambda x: (-len(x["in_datasets"]), -int(x["looks_like_key"])))
    return suggestions

def merge_datasets(primary_label, datasets, merge_keys, how="inner"):
    log = []; primary = datasets[primary_label].copy()
    primary.columns = [c.strip() for c in primary.columns]; result = primary
    log.append(f"Primary: {primary_label} ({len(primary):,} rows)")
    for label, df in datasets.items():
        if label == primary_label: continue
        df = df.copy(); df.columns = [c.strip() for c in df.columns]
        valid_keys = [k for k in merge_keys if k in result.columns and k in df.columns]
        if not valid_keys: log.append(f"Skipped {label} — no valid merge keys"); continue
        before = len(result)
        result = result.merge(df, on=valid_keys, how=how, suffixes=("", f"_{label[:6]}"))
        log.append(f"Merged {label} on {valid_keys} ({how}): {before:,} -> {len(result):,} rows")
    log.append(f"Final: {result.shape[0]:,} rows x {result.shape[1]} cols")
    return result, log

def _quick_cv(X, y, task, cv=3):
    try:
        Xs = StandardScaler().fit_transform(X.fillna(0))
        m = (RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) if task == "Regression"
             else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
        return float(cross_val_score(m, Xs, y, cv=cv,
                                      scoring="r2" if task == "Regression" else "accuracy").mean())
    except Exception: return 0.0

def cross_dataset_correlation(primary_df, target_col, external_dfs, task):
    if target_col not in primary_df.columns: return {}
    pnum = primary_df.select_dtypes(include=[np.number]).copy()
    y_raw = primary_df[target_col]
    y = (pd.Series(LabelEncoder().fit_transform(y_raw.astype(str))) if task == "Classification"
         else pd.to_numeric(y_raw, errors="coerce"))
    y = y.fillna(y.mean() if task == "Regression" else y.mode()[0])
    Xb = pnum.drop(columns=[target_col], errors="ignore").fillna(0)
    if Xb.empty or len(Xb) < 20: return {}
    baseline = _quick_cv(Xb, y, task); analysis = {}
    for label, ext_df in external_dfs.items():
        enum = ext_df.select_dtypes(include=[np.number]).copy()
        if enum.empty: analysis[label] = {"error": "No numeric columns"}; continue
        corr_rows = []
        if task == "Regression":
            yal = y.values[:min(len(y), len(enum))]
            for col in enum.columns:
                try:
                    vals = enum[col].fillna(enum[col].mean()).values[:len(yal)]
                    if len(vals) < 5: continue
                    c = np.corrcoef(yal, vals)[0, 1]
                    if not np.isnan(c):
                        corr_rows.append({"feature":col,"pearson_r":round(float(c),4),
                                          "abs_corr":round(abs(float(c)),4)})
                except Exception: pass
            corr_rows.sort(key=lambda x: -x["abs_corr"])
        n = min(len(Xb), len(enum)); ext_imp = {}; aug_score = baseline
        if n >= 20:
            Xaug = pd.concat([Xb.iloc[:n].reset_index(drop=True),
                               enum.iloc[:n].fillna(0).reset_index(drop=True)], axis=1)
            Xaug = Xaug.loc[:, ~Xaug.columns.duplicated()]
            yaug = y.iloc[:n].reset_index(drop=True)
            aug_score = _quick_cv(Xaug, yaug, task)
            try:
                Xs = StandardScaler().fit_transform(Xaug.fillna(0))
                m = (RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) if task == "Regression"
                     else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
                m.fit(Xs, yaug); imp = dict(zip(Xaug.columns, m.feature_importances_))
                ext_imp = dict(sorted({k: round(v, 4) for k, v in imp.items()
                                        if k in enum.columns}.items(), key=lambda x: -x[1])[:10])
            except Exception: pass
        delta = round(aug_score - baseline, 4)
        if delta > 0.03: verdict = "strong_boost"; vt = f"Strong positive impact: +{delta:.4f} improvement when merged"
        elif delta > 0.005: verdict = "mild_boost"; vt = f"Mild improvement: +{delta:.4f} — worth including"
        elif delta > -0.01: verdict = "neutral"; vt = "Neutral — minimal impact on model performance"
        else: verdict = "negative"; vt = f"Slightly hurts performance ({delta:.4f})"
        analysis[label] = {"baseline_score":round(baseline,4),"augmented_score":round(aug_score,4),
                            "score_delta":delta,"verdict":verdict,"verdict_text":vt,
                            "top_correlations":corr_rows[:10],"top_ext_importances":ext_imp,
                            "n_rows_analysed":n}
    return analysis

def recommend_enrichment(analysis, baseline_model_score):
    recs = []
    for label, info in analysis.items():
        if "error" in info: continue
        delta = info["score_delta"]
        tf = list(info.get("top_ext_importances", {}).keys())[:5]
        tc = [r["feature"] for r in info.get("top_correlations", [])[:3]]
        recs.append({"dataset":label,"score_delta":delta,"verdict":info["verdict"],
                     "verdict_text":info["verdict_text"],"key_features":tf if tf else tc,
                     "projected_score":round(baseline_model_score+delta,4),
                     "priority":"HIGH" if delta > 0.03 else "MEDIUM" if delta > 0 else "LOW"})
    recs.sort(key=lambda x: -x["score_delta"]); return recs

def build_enriched_dataset(primary_df, target_col, selected_datasets, selected_cols):
    log = []; result = primary_df.copy(); added = 0
    for label, cols in selected_cols.items():
        ext_df = selected_datasets.get(label)
        if ext_df is None or not cols: continue
        valid_cols = [c for c in cols if c in ext_df.columns]
        if not valid_cols: continue
        n = min(len(result), len(ext_df))
        subset = ext_df[valid_cols].iloc[:n].reset_index(drop=True)
        result = result.iloc[:n].reset_index(drop=True)
        for col in valid_cols:
            new_name = f"{col}__{label[:6]}" if col in result.columns else col
            result[new_name] = subset[col].values
        added += len(valid_cols); log.append(f"Appended {len(valid_cols)} cols from {label}: {valid_cols}")
    log.append(f"Enriched: {result.shape[0]:,} rows x {result.shape[1]} cols (+{added} external cols)")
    return result, log
