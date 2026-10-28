import streamlit as st
from backend.models import REGRESSION_MODELS, CLASSIFICATION_MODELS

def hyperparam_ui(model_name, prefix, task):
    reg = (task == "Regression")
    key = (REGRESSION_MODELS if reg else CLASSIFICATION_MODELS).get(model_name, "rf")
    p = {}
    sf = (prefix.replace(" ","_").replace("(","").replace(")","")
                .replace("/","_").replace(".","").replace("-","_"))
    if key in ("rf","et"):
        c1,c2,c3 = st.columns(3)
        p["n_estimators"]      = c1.slider("n_estimators",50,600,200,50,key=f"{sf}_ne")
        p["max_depth"]         = c2.select_slider("max_depth",[None,3,5,8,10,15,20],None,key=f"{sf}_md")
        p["min_samples_split"] = c3.slider("min_samples_split",2,20,2,key=f"{sf}_mss")
        p["random_state"] = 42
    elif key == "xgb":
        c1,c2,c3,c4 = st.columns(4)
        p["n_estimators"]  = c1.slider("n_estimators",50,600,200,50,key=f"{sf}_ne")
        p["learning_rate"] = c2.select_slider("learning_rate",[.01,.05,.1,.15,.2,.3],.1,key=f"{sf}_lr")
        p["max_depth"]     = c3.slider("max_depth",2,12,6,key=f"{sf}_md")
        p["subsample"]     = c4.slider("subsample",.5,1.0,1.0,.05,key=f"{sf}_ss")
        p["random_state"] = 42
    elif key == "gbm":
        c1,c2,c3,c4 = st.columns(4)
        p["n_estimators"]  = c1.slider("n_estimators",50,500,150,50,key=f"{sf}_ne")
        p["learning_rate"] = c2.select_slider("learning_rate",[.01,.05,.1,.2,.3],.1,key=f"{sf}_lr")
        p["max_depth"]     = c3.slider("max_depth",2,10,4,key=f"{sf}_md")
        p["subsample"]     = c4.slider("subsample",.5,1.0,1.0,.05,key=f"{sf}_ss")
        p["random_state"] = 42
    elif key == "ada":
        c1,c2 = st.columns(2)
        p["n_estimators"]  = c1.slider("n_estimators",20,300,100,20,key=f"{sf}_ne")
        p["learning_rate"] = c2.select_slider("learning_rate",[.01,.05,.1,.5,1.0],1.0,key=f"{sf}_lr")
        p["random_state"] = 42
    elif key == "dt":
        c1,c2 = st.columns(2)
        p["max_depth"]         = c1.select_slider("max_depth",[None,3,5,8,10,15,20],None,key=f"{sf}_md")
        p["min_samples_split"] = c2.slider("min_samples_split",2,20,2,key=f"{sf}_mss")
        p["random_state"] = 42
    elif key in ("ridge","lasso"):
        p["alpha"] = st.select_slider("alpha",[.001,.01,.1,1.,10.,100.],1.,key=f"{sf}_a")
    elif key == "enet":
        c1,c2 = st.columns(2)
        p["alpha"]    = c1.select_slider("alpha",[.001,.01,.1,1.,10.,100.],1.,key=f"{sf}_a")
        p["l1_ratio"] = c2.slider("l1_ratio",0.0,1.0,.5,.05,key=f"{sf}_l1")
    elif key == "lr":
        st.caption("Linear Regression — no tunable hyperparameters (OLS).")
    elif key in ("svr","svc"):
        c1,c2,c3 = st.columns(3)
        p["C"]      = c1.select_slider("C",[.01,.1,1.,10.,100.],1.,key=f"{sf}_C")
        p["kernel"] = c2.selectbox("Kernel",["rbf","linear","poly","sigmoid"],key=f"{sf}_ker")
        if key == "svr":
            p["epsilon"] = c3.slider("epsilon",.01,1.0,.1,.01,key=f"{sf}_eps")
    elif key == "knn":
        c1,c2,c3 = st.columns(3)
        p["n_neighbors"] = c1.slider("k (neighbours)",1,30,5,key=f"{sf}_kn")
        p["weights"]     = c2.selectbox("Weights",["uniform","distance"],key=f"{sf}_w")
        p["metric"]      = c3.selectbox("Metric",["minkowski","euclidean","manhattan"],key=f"{sf}_met")
    elif key == "mlp":
        c1,c2,c3 = st.columns(3)
        hl = c1.selectbox("Hidden layers",["(100,)","(100,50)","(100,100)","(200,100,50)","(256,128,64)"],key=f"{sf}_hl")
        p["hidden_layer_sizes"] = eval(hl)
        p["activation"]         = c2.selectbox("Activation",["relu","tanh","logistic"],key=f"{sf}_act")
        p["learning_rate_init"] = c3.select_slider("LR",[.0001,.001,.01,.05],.001,key=f"{sf}_lri")
        p["random_state"] = 42
    elif key == "logreg":
        c1,c2 = st.columns(2)
        p["C"]      = c1.select_slider("C",[.001,.01,.1,1.,10.,100.],1.,key=f"{sf}_C")
        p["solver"] = c2.selectbox("Solver",["lbfgs","liblinear","saga"],key=f"{sf}_sol")
        p["random_state"] = 42
    return key, p
