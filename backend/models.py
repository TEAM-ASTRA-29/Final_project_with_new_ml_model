import time, os, joblib
import numpy as np, pandas as pd, psutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, f1_score, precision_score, recall_score,
                              confusion_matrix, roc_auc_score)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor, ExtraTreesRegressor,
                               RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings; warnings.filterwarnings("ignore")

REGRESSION_MODELS = {
    "Random Forest":"rf","XGBoost":"xgb","Gradient Boosting":"gbm",
    "Extra Trees":"et","AdaBoost":"ada","Decision Tree":"dt",
    "Ridge Regression":"ridge","Lasso Regression":"lasso","ElasticNet":"enet",
    "Linear Regression":"lr","SVR (SVM)":"svr",
    "K-Nearest Neighbors":"knn","MLP Neural Network":"mlp",
}
CLASSIFICATION_MODELS = {
    "Random Forest":"rf","XGBoost":"xgb","Gradient Boosting":"gbm",
    "Extra Trees":"et","AdaBoost":"ada","Decision Tree":"dt",
    "Logistic Regression":"logreg","SVC (SVM)":"svc",
    "K-Nearest Neighbors":"knn","MLP Neural Network":"mlp",
}
MODEL_KNOWLEDGE = {
    "rf":{"full":"Random Forest","type":"Ensemble — Bagging","year":2001,"author":"Leo Breiman",
          "short":"Builds many decision trees on random data subsets and averages predictions.",
          "how_it_works":"Trains N decision trees on bootstrap samples with random feature subsets. Final prediction is mean (regression) or majority vote (classification).",
          "strengths":["Robust to overfitting","Handles high dimensionality","Built-in feature importance"],
          "weaknesses":["Large model size","Slow on very large datasets"],
          "best_for":"General-purpose baseline on tabular data",
          "hyperparams":{"n_estimators":"Number of trees","max_depth":"Max tree depth","min_samples_split":"Min samples to split"},
          "papers":[("Breiman 2001","https://link.springer.com/article/10.1023/A:1010933404324")],
          "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=J4Wdy0Wc_xQ"),("Sklearn docs","https://scikit-learn.org/stable/modules/ensemble.html#random-forests")]},
    "xgb":{"full":"XGBoost","type":"Ensemble — Gradient Boosting","year":2016,"author":"Chen & Guestrin",
           "short":"Optimised gradient boosting with regularisation.",
           "how_it_works":"Builds trees sequentially correcting prior errors. Uses 2nd-order Taylor loss, L1/L2 regularisation, subsampling.",
           "strengths":["State-of-the-art accuracy","Fast","Handles missing values"],
           "weaknesses":["Many hyperparameters","Can overfit without tuning"],
           "best_for":"Competition-grade accuracy on structured data",
           "hyperparams":{"n_estimators":"Boosting rounds","learning_rate":"Step size","max_depth":"Tree depth","subsample":"Row sampling"},
           "papers":[("Chen & Guestrin 2016","https://arxiv.org/abs/1603.02754")],
           "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=OtD8wVaFm6E"),("XGBoost docs","https://xgboost.readthedocs.io")]},
    "gbm":{"full":"Gradient Boosting","type":"Ensemble — Gradient Boosting","year":1999,"author":"Jerome Friedman",
           "short":"Sequentially builds trees to minimise a differentiable loss function.",
           "how_it_works":"Each tree fits pseudo-residuals (negative gradient of loss) from the previous ensemble.",
           "strengths":["High accuracy","Flexible loss functions"],
           "weaknesses":["Slower than XGBoost","Sequential — hard to parallelise"],
           "best_for":"High-accuracy tasks where training time is not critical",
           "hyperparams":{"n_estimators":"Number of stages","learning_rate":"Shrinkage","max_depth":"Tree depth"},
           "papers":[("Friedman 1999","https://statweb.stanford.edu/~jhf/ftp/trebst.pdf")],
           "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=3CC4N4z3GJc")]},
    "et":{"full":"Extra Trees","type":"Ensemble — Extreme Randomisation","year":2006,"author":"Geurts et al.",
          "short":"Like Random Forest but with fully random split thresholds.",
          "how_it_works":"Picks split thresholds randomly, reducing variance and speeding up training.",
          "strengths":["Very fast","Low variance"],
          "weaknesses":["Slightly less accurate than RF on clean data"],
          "best_for":"Speed-critical tasks with noisy features",
          "hyperparams":{"n_estimators":"Number of trees","max_depth":"Max depth"},
          "papers":[("Geurts et al. 2006","https://link.springer.com/article/10.1007/s10994-006-6226-1")],
          "learn":[("Sklearn docs","https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html")]},
    "ada":{"full":"AdaBoost","type":"Ensemble — Adaptive Boosting","year":1995,"author":"Freund & Schapire",
           "short":"Boosts weak classifiers by focusing on misclassified examples.",
           "how_it_works":"Trains weak learners sequentially, increasing weight of misclassified samples.",
           "strengths":["Simple","Works on clean balanced data"],
           "weaknesses":["Sensitive to noise and outliers"],
           "best_for":"Binary classification with clean balanced data",
           "hyperparams":{"n_estimators":"Number of weak learners","learning_rate":"Shrinkage per learner"},
           "papers":[("Freund & Schapire 1997","https://www.sciencedirect.com/science/article/pii/S002200009791504X")],
           "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=LsK-xG1cLYA")]},
    "dt":{"full":"Decision Tree","type":"Tree-based","year":1986,"author":"Quinlan / Breiman et al.",
          "short":"Recursively splits data on the feature that best reduces impurity.",
          "how_it_works":"At each node picks feature/threshold maximising information gain or minimising MSE.",
          "strengths":["Fully interpretable","No scaling needed","Fast"],
          "weaknesses":["High variance","Prone to overfitting"],
          "best_for":"Interpretability-critical applications",
          "hyperparams":{"max_depth":"Max tree depth","min_samples_split":"Min samples to split"},
          "papers":[("CART 1984","https://www.taylorfrancis.com/books/9781315139470")],
          "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=7VeUPuFGJHk")]},
    "ridge":{"full":"Ridge Regression","type":"Linear — L2","year":1970,"author":"Hoerl & Kennard",
             "short":"Linear regression with L2 penalty to shrink coefficients.",
             "how_it_works":"Minimises RSS + alpha*sum(beta^2). Shrinks all coefficients toward zero.",
             "strengths":["Fast","Handles multicollinearity"],
             "weaknesses":["Linear only","No feature selection"],
             "best_for":"Linear problems with correlated features",
             "hyperparams":{"alpha":"Regularisation strength"},
             "papers":[("Hoerl & Kennard 1970","https://www.jstor.org/stable/1267351")],
             "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=Q81RR3yKn30")]},
    "lasso":{"full":"Lasso Regression","type":"Linear — L1","year":1996,"author":"Robert Tibshirani",
             "short":"Linear regression with L1 penalty — drives some coefficients to exactly zero.",
             "how_it_works":"Minimises RSS + alpha*sum(|beta|). Produces sparse solutions.",
             "strengths":["Automatic feature selection","Sparse model"],
             "weaknesses":["Drops correlated features arbitrarily"],
             "best_for":"High-dimensional data with many irrelevant features",
             "hyperparams":{"alpha":"Regularisation strength"},
             "papers":[("Tibshirani 1996","https://www.jstor.org/stable/2346178")],
             "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=NGf0voTMlcs")]},
    "enet":{"full":"ElasticNet","type":"Linear — L1+L2","year":2005,"author":"Zou & Hastie",
            "short":"Combines Lasso and Ridge for balance of selection and stability.",
            "how_it_works":"Minimises RSS + alpha*l1_ratio*sum(|beta|) + alpha*(1-l1_ratio)*sum(beta^2).",
            "strengths":["Feature selection + stability","Handles correlated groups"],
            "weaknesses":["Two hyperparameters to tune"],
            "best_for":"High-dimensional data with groups of correlated features",
            "hyperparams":{"alpha":"Overall regularisation","l1_ratio":"Mix L1(1.0) vs L2(0.0)"},
            "papers":[("Zou & Hastie 2005","https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x")],
            "learn":[("Sklearn docs","https://scikit-learn.org/stable/modules/linear_model.html#elastic-net")]},
    "lr":{"full":"Linear Regression (OLS)","type":"Linear","year":1805,"author":"Legendre & Gauss",
          "short":"Fits a hyperplane minimising sum of squared residuals.",
          "how_it_works":"Finds beta that minimises sum((y-X*beta)^2). Closed-form solution.",
          "strengths":["Extremely fast","Fully interpretable","No hyperparameters"],
          "weaknesses":["Linear only","No regularisation"],
          "best_for":"Linear relationships — baseline for any regression problem",
          "hyperparams":{},
          "papers":[("Wikipedia Least Squares","https://en.wikipedia.org/wiki/Least_squares")],
          "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=7ArmBVF2dCs")]},
    "svr":{"full":"SVR (Support Vector Regression)","type":"Kernel SVM","year":1997,"author":"Vapnik et al.",
           "short":"Fits a hyperplane within an epsilon-tube using the kernel trick.",
           "how_it_works":"Finds function deviating from actuals by at most epsilon. Kernel maps to higher dims.",
           "strengths":["Non-linear via kernel","Robust to outliers"],
           "weaknesses":["Slow O(n^2)","Requires feature scaling"],
           "best_for":"Small-medium datasets with non-linear patterns",
           "hyperparams":{"C":"Regularisation","kernel":"rbf/linear/poly","epsilon":"Tube width"},
           "papers":[("Drucker et al. 1997","https://papers.nips.cc/paper/1997")],
           "learn":[("StatQuest SVMs","https://www.youtube.com/watch?v=efR1C6CvhmE")]},
    "svc":{"full":"SVC (Support Vector Classification)","type":"Kernel SVM","year":1995,"author":"Cortes & Vapnik",
           "short":"Finds maximum-margin hyperplane separating classes.",
           "how_it_works":"Maximises margin between classes. Kernel handles non-linear boundaries.",
           "strengths":["Strong on small high-dim data","Memory efficient"],
           "weaknesses":["Slow training","Needs scaling"],
           "best_for":"Text classification with small datasets",
           "hyperparams":{"C":"Regularisation","kernel":"rbf/linear/poly/sigmoid"},
           "papers":[("Cortes & Vapnik 1995","https://link.springer.com/article/10.1007/BF00994018")],
           "learn":[("StatQuest SVMs","https://www.youtube.com/watch?v=efR1C6CvhmE")]},
    "knn":{"full":"K-Nearest Neighbors","type":"Instance-based","year":1951,"author":"Fix & Hodges",
           "short":"Predicts based on k most similar training examples.",
           "how_it_works":"At prediction finds k nearest training points by distance and averages or votes.",
           "strengths":["No training time","Non-parametric"],
           "weaknesses":["Slow inference","High memory","Curse of dimensionality"],
           "best_for":"Small datasets, recommendation systems",
           "hyperparams":{"n_neighbors":"k neighbours","weights":"uniform or distance","metric":"Distance function"},
           "papers":[("Fix & Hodges 1951","https://apps.dtic.mil/sti/citations/ADA800276")],
           "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=HVXime0nQeI")]},
    "mlp":{"full":"MLP Neural Network","type":"Neural Network","year":1986,"author":"Rumelhart et al.",
           "short":"Feedforward neural network trained via backpropagation.",
           "how_it_works":"Stacks layers with non-linear activations. Backprop computes gradients; gradient descent updates weights.",
           "strengths":["Learns complex patterns","Universal approximator"],
           "weaknesses":["Black box","Needs tuning and scaling"],
           "best_for":"Complex non-linear problems when tree methods plateau",
           "hyperparams":{"hidden_layer_sizes":"Architecture e.g. (100,50)","activation":"relu/tanh/logistic","learning_rate_init":"Initial step size"},
           "papers":[("Rumelhart et al. 1986","https://www.nature.com/articles/323533a0")],
           "learn":[("3Blue1Brown YouTube","https://www.youtube.com/watch?v=aircAruvnKk"),("NN & DL book","http://neuralnetworksanddeeplearning.com")]},
    "logreg":{"full":"Logistic Regression","type":"Linear Classifier","year":1958,"author":"David Cox",
              "short":"Models class probability as sigmoid of linear combination of features.",
              "how_it_works":"Fits P(y=1|x)=sigmoid(X*beta) via maximum likelihood.",
              "strengths":["Fast","Probabilistic output","Interpretable"],
              "weaknesses":["Linear boundary only"],
              "best_for":"Binary classification baseline, medical risk scoring",
              "hyperparams":{"C":"Inverse regularisation","solver":"Optimisation algorithm"},
              "papers":[("Cox 1958","https://www.jstor.org/stable/2983890")],
              "learn":[("StatQuest YouTube","https://www.youtube.com/watch?v=yIYKR4sgzI8")]},
}

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_sessions")
os.makedirs(PERSIST_DIR, exist_ok=True)

def recommend_models(df, target, task, user_rules=None):
    n_rows, n_cols = df.shape; n_features = n_cols - 1
    has_nulls = df.isnull().sum().sum() > 0
    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    cat_ratio = 1 - (num_cols / max(n_cols, 1))
    registry = REGRESSION_MODELS if task == "Regression" else CLASSIFICATION_MODELS
    scores, reasons = {}, {}
    for name, key in registry.items():
        s, r = 50, []
        if n_rows > 50000:
            if key in ("svr","svc","knn"): s -= 30; r.append("Slow on >50k rows")
            if key in ("rf","xgb","gbm","et"): s += 15; r.append("Scales to large data")
        elif n_rows < 1000:
            if key in ("svr","svc","knn","logreg","ridge","lasso","lr"): s += 15; r.append("Works on small data")
            if key in ("mlp","xgb","gbm"): s -= 10; r.append("May overfit small data")
        if n_features > 50:
            if key in ("lasso","enet"): s += 20; r.append("Auto feature selection")
            if key in ("knn","svc","svr"): s -= 15; r.append("Struggles in high dims")
        if has_nulls:
            if key in ("xgb","gbm","rf","et"): s += 10; r.append("Tolerates missing values")
        if cat_ratio > 0.3:
            if key in ("rf","et","xgb","gbm","dt","ada"): s += 10; r.append("Handles categoricals")
        if user_rules:
            if user_rules.get("prefer_speed"):
                if key in ("lr","ridge","lasso","dt","knn","logreg"): s += 25; r.append("Fast (user pref)")
                if key in ("mlp","svr","svc","gbm"): s -= 15
            if user_rules.get("prefer_accuracy"):
                if key in ("xgb","gbm","rf","et","mlp"): s += 25; r.append("High accuracy (user pref)")
            if user_rules.get("prefer_interpretable"):
                if key in ("lr","ridge","lasso","logreg","dt"): s += 25; r.append("Interpretable (user pref)")
                if key in ("mlp","xgb","gbm","rf"): s -= 15
            if user_rules.get("small_model"):
                if key in ("lr","ridge","lasso","logreg","dt"): s += 20; r.append("Small size (user pref)")
        if not r: r.append("Suitable for this dataset")
        scores[name] = max(0, s); reasons[name] = r
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return {"ranked": ranked, "reasons": reasons,
            "dataset_profile": {"n_rows":n_rows,"n_features":n_features,
                                "has_nulls":has_nulls,"cat_ratio":round(cat_ratio,2)}}

def build_model(key, params, task):
    p = params
    if task == "Regression":
        M = {"rf":lambda:RandomForestRegressor(**p,n_jobs=-1),
             "xgb":lambda:XGBRegressor(**p,verbosity=0,n_jobs=-1),
             "gbm":lambda:GradientBoostingRegressor(**p),
             "et":lambda:ExtraTreesRegressor(**p,n_jobs=-1),
             "ada":lambda:AdaBoostRegressor(**p),"dt":lambda:DecisionTreeRegressor(**p),
             "ridge":lambda:Ridge(**p),"lasso":lambda:Lasso(**p),"enet":lambda:ElasticNet(**p),
             "lr":lambda:LinearRegression(),"svr":lambda:SVR(**p),
             "knn":lambda:KNeighborsRegressor(**p),"mlp":lambda:MLPRegressor(**p,max_iter=500)}
    else:
        M = {"rf":lambda:RandomForestClassifier(**p,n_jobs=-1),
             "xgb":lambda:XGBClassifier(**p,verbosity=0,eval_metric="logloss",n_jobs=-1),
             "gbm":lambda:GradientBoostingClassifier(**p),
             "et":lambda:ExtraTreesClassifier(**p,n_jobs=-1),
             "ada":lambda:AdaBoostClassifier(**p),"dt":lambda:DecisionTreeClassifier(**p),
             "logreg":lambda:LogisticRegression(**p,max_iter=500),
             "svc":lambda:SVC(**p,probability=True),
             "knn":lambda:KNeighborsClassifier(**p),"mlp":lambda:MLPClassifier(**p,max_iter=500)}
    return M[key]()

def train_one(name, mkey, params, X_tr, X_te, y_tr, y_te, task, scale, persist_dir):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    if scale:
        sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    model = build_model(mkey, params, task)
    proc = psutil.Process(); m0 = proc.memory_info().rss/(1024**2)
    t0 = time.time(); model.fit(X_tr, y_tr)
    rt = time.time()-t0; mem = max(0, proc.memory_info().rss/(1024**2)-m0)
    tmp = os.path.join(persist_dir, f"_tmp_{mkey}.pkl"); joblib.dump(model, tmp)
    size = os.path.getsize(tmp)/(1024**2)
    fi = None
    if hasattr(model,"feature_importances_"): fi = model.feature_importances_.tolist()
    elif hasattr(model,"coef_"):
        c = np.abs(model.coef_)
        if c.ndim > 1: c = c.mean(axis=0)
        fi = (c/(c.sum()+1e-9)).tolist()
    out = {"name":name,"key":mkey,"params":{k:str(v) for k,v in params.items()},
           "runtime":round(rt,4),"memory_mb":round(mem,2),"model_size_mb":round(size,4),"feature_importances":fi}
    if task == "Regression":
        yp = model.predict(X_te)
        out.update({"y_pred":yp.tolist(),"r2":round(float(r2_score(y_te,yp)),6),
                    "rmse":round(float(np.sqrt(mean_squared_error(y_te,yp))),6),
                    "mae":round(float(mean_absolute_error(y_te,yp)),6),
                    "mse":round(float(mean_squared_error(y_te,yp)),6)})
    else:
        yp = model.predict(X_te); classes = list(np.unique(y_te))
        avg = "binary" if len(classes)==2 else "weighted"
        y_prob=None; auc=None
        try:
            proba = model.predict_proba(X_te)
            if len(classes)==2: auc=round(float(roc_auc_score(y_te,proba[:,1])),6); y_prob=proba[:,1].tolist()
            else: auc=round(float(roc_auc_score(y_te,proba,multi_class="ovr",average="weighted")),6); y_prob=proba.tolist()
        except: pass
        out.update({"y_pred":yp.tolist(),"y_prob":y_prob,
                    "accuracy":round(float(accuracy_score(y_te,yp)),6),
                    "f1":round(float(f1_score(y_te,yp,average=avg,zero_division=0)),6),
                    "precision":round(float(precision_score(y_te,yp,average=avg,zero_division=0)),6),
                    "recall":round(float(recall_score(y_te,yp,average=avg,zero_division=0)),6),
                    "auc":auc,"conf_matrix":confusion_matrix(y_te,yp).tolist()})
    return out

def simulate_cloud_vs_local(result, task, cloud_spec="t3.medium"):
    SPECS={"t3.micro":{"cpu_mult":.5,"vcpu":2,"ram_gb":1},"t3.medium":{"cpu_mult":1.,"vcpu":2,"ram_gb":4},
           "t3.large":{"cpu_mult":2.,"vcpu":2,"ram_gb":8},"c5.xlarge":{"cpu_mult":4.,"vcpu":4,"ram_gb":8},
           "c5.4xlarge":{"cpu_mult":8.,"vcpu":16,"ram_gb":32},"m5.4xlarge":{"cpu_mult":6.,"vcpu":16,"ram_gb":64},
           "p3.2xlarge":{"cpu_mult":12.,"vcpu":8,"ram_gb":61}}
    COST={"t3.micro":.0104,"t3.medium":.0416,"t3.large":.0832,"c5.xlarge":.17,
          "c5.4xlarge":.68,"m5.4xlarge":.768,"p3.2xlarge":3.06}
    spec=SPECS.get(cloud_spec,SPECS["t3.medium"])
    lv=psutil.cpu_count(logical=True); lr=psutil.virtual_memory().total/(1024**3)
    lrt=result["runtime"]; lm=max(1.,lv/2.); crt=max(.001,lrt*(lm/spec["cpu_mult"]))
    cost=(crt/3600)*COST.get(cloud_spec,.0416)*100
    return {"local":{"runtime_s":lrt,"memory_mb":result["memory_mb"],"vcpu":lv,"ram_gb":round(lr,1),"cost_cents":0.},
            "cloud":{"instance":cloud_spec,"runtime_s":round(crt,4),"memory_mb":result["memory_mb"],
                     "vcpu":spec["vcpu"],"ram_gb":spec["ram_gb"],"cost_cents":round(cost,6),
                     "speedup":round(lrt/max(crt,.001),2)}}
