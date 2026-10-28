import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class DataProfiler:
    def __init__(self, df): self.df = df
    def profile(self):
        df = self.df; n_rows, n_cols = df.shape
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        nc = df.isnull().sum(); np2 = (nc/n_rows*100).round(2)
        details = []
        for col in df.columns:
            d = {"column":col,"dtype":str(df[col].dtype),"nulls":int(nc[col]),
                 "null_pct":round(float(np2[col]),2),"unique":int(df[col].nunique()),
                 "is_numeric":col in num,"is_categorical":col in cat}
            if col in num:
                d.update({"mean":round(float(df[col].mean()),4),"std":round(float(df[col].std()),4),
                           "min":round(float(df[col].min()),4),"max":round(float(df[col].max()),4),
                           "median":round(float(df[col].median()),4),"skew":round(float(df[col].skew()),4)})
            details.append(d)
        return {"n_rows":n_rows,"n_cols":n_cols,"numeric_cols":num,"categorical_cols":cat,
                "null_total":int(nc.sum()),"duplicate_rows":int(df.duplicated().sum()),
                "col_details":details,"looks_preprocessed":len(cat)==0 and nc.sum()==0}

class PreprocessingPipeline:
    def __init__(self, config): self.cfg=config; self.encoders={}; self.scaler=None; self.log=[]
    def fit_transform(self, df):
        cfg=self.cfg; target=cfg["target_col"]; df=df.copy()
        drop=[c for c in cfg.get("drop_cols",[]) if c in df.columns and c!=target]
        if drop: df.drop(columns=drop,inplace=True); self.log.append(f"Dropped: {drop}")
        if target not in df.columns: raise ValueError(f"Target '{target}' not found.")
        y=df[target].copy(); X=df.drop(columns=[target])
        ac=cfg.get("already_clean",False)
        if not ac and cfg.get("handle_dups",False):
            before=len(X); mask=~X.duplicated()
            X,y=X[mask].reset_index(drop=True),y[mask].reset_index(drop=True)
            self.log.append(f"Removed {before-len(X)} duplicates")
        cats=X.select_dtypes(include=["object","category"]).columns.tolist()
        hc=cfg.get("handle_cats","label")
        if cats and hc!="none":
            if ac: X.drop(columns=cats,inplace=True,errors="ignore")
            elif hc=="label":
                for col in cats:
                    le=LabelEncoder(); X[col]=le.fit_transform(X[col].astype(str))
                    self.encoders[col]=list(le.classes_)
                self.log.append(f"Label-encoded: {cats}")
            elif hc=="onehot": X=pd.get_dummies(X,columns=cats,drop_first=True); self.log.append(f"One-hot: {cats}")
            elif hc=="drop": X.drop(columns=cats,inplace=True,errors="ignore"); self.log.append(f"Dropped cats: {cats}")
        hn=cfg.get("handle_nulls","mean")
        if not ac and hn!="none":
            nc2=X.columns[X.isnull().any()].tolist()
            if nc2:
                if hn=="drop":
                    before=len(X); mask=X.notna().all(axis=1)
                    X,y=X[mask].reset_index(drop=True),y[mask].reset_index(drop=True)
                    self.log.append(f"Dropped {before-len(X)} null rows")
                else:
                    for c in nc2:
                        if X[c].dtype in [np.float64,np.int64,float,int]:
                            if hn=="mean": X[c].fillna(X[c].mean(),inplace=True)
                            elif hn=="median": X[c].fillna(X[c].median(),inplace=True)
                            elif hn=="mode": X[c].fillna(X[c].mode()[0],inplace=True)
                    self.log.append(f"Filled nulls ({hn}): {nc2}")
        ny=y.isnull()
        if ny.any():
            X=X[~ny].reset_index(drop=True); y=y[~ny].reset_index(drop=True)
            self.log.append(f"Dropped {ny.sum()} null target rows")
        for c in X.select_dtypes(include=[np.number]).columns:
            if X[c].isnull().any(): X[c].fillna(X[c].mean(),inplace=True)
        if not ac and cfg.get("drop_low_var",False):
            lv=X.var()[X.var()<1e-6].index.tolist()
            if lv: X.drop(columns=lv,inplace=True); self.log.append(f"Dropped low-var: {lv}")
        sm=cfg.get("scale_method","none")
        if sm=="standard":
            self.scaler=StandardScaler(); X=pd.DataFrame(self.scaler.fit_transform(X),columns=X.columns); self.log.append("StandardScaler applied")
        elif sm=="minmax":
            self.scaler=MinMaxScaler(); X=pd.DataFrame(self.scaler.fit_transform(X),columns=X.columns); self.log.append("MinMaxScaler applied")
        self.feature_names=list(X.columns); return X,y
    def get_log(self): return self.log
