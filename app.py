"""
CropML Research Studio v5.0
Run:  streamlit run app.py
"""
import sys, os, base64, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

from backend.models       import (REGRESSION_MODELS, CLASSIFICATION_MODELS,
                                   MODEL_KNOWLEDGE, recommend_models,
                                   train_one, simulate_cloud_vs_local)
from backend.preprocessor import DataProfiler, PreprocessingPipeline
from backend.session      import make_sid, save_session, load_sessions, delete_session
from backend.automl       import run_benchmark, focus_tune, train_focused_model
from backend.multi_dataset import (DatasetRegistry, find_common_columns,
                                   suggest_merge_keys, merge_datasets,
                                   cross_dataset_correlation,
                                   recommend_enrichment, build_enriched_dataset)
from frontend.hyperparams import hyperparam_ui

try:
    from backend.report import generate_pdf
    PDF_OK = True
except Exception:
    PDF_OK = False

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_sessions")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CropML Studio", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0d1117;--s1:#161b22;--s2:#1c2128;--bd:#30363d;--bd2:#21262d;
  --blue:#58a6ff;--green:#3fb950;--org:#f78166;--purple:#bc8cff;
  --yellow:#e3b341;--teal:#39d353;--text:#e6edf3;--muted:#7d8590;--r:8px;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);background:var(--bg);}
.block-container{padding:1.4rem 2rem 3rem;max-width:1380px;}
h1{font-size:1.55rem;font-weight:700;letter-spacing:-.025em;margin-bottom:.1rem;}
h2{font-size:1.1rem;font-weight:600;}

/* ── KPI cards ── */
.kc{background:var(--s1);border:1px solid var(--bd2);border-radius:var(--r);
    padding:.85rem 1rem;position:relative;overflow:hidden;}
.kc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--blue);}
.kc.g::after{background:var(--green);}
.kc.o::after{background:var(--org);}
.kc.p::after{background:var(--purple);}
.kc.y::after{background:var(--yellow);}
.kc.t::after{background:var(--teal);}
.kl{font-size:.63rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
    color:var(--muted);margin-bottom:.25rem;}
.kv{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:500;
    color:var(--text);line-height:1.1;}
.ks{font-size:.68rem;color:var(--muted);margin-top:.15rem;}

/* ── Section headers ── */
.sh{display:flex;align-items:center;gap:.5rem;font-size:.95rem;font-weight:600;
    padding:.4rem 0;margin:1.4rem 0 .7rem;border-bottom:1px solid var(--bd2);}
.sh-n{display:flex;align-items:center;justify-content:center;width:20px;height:20px;
    background:var(--blue);color:#0d1117;border-radius:50%;font-size:.68rem;font-weight:700;flex-shrink:0;}

/* ── Badges ── */
.b{display:inline-block;border-radius:4px;padding:.1rem .45rem;
   font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:500;}
.bb{background:rgba(88,166,255,.1);color:var(--blue);border:1px solid rgba(88,166,255,.22);}
.bg{background:rgba(63,185,80,.1);color:var(--green);border:1px solid rgba(63,185,80,.22);}
.bo{background:rgba(247,129,102,.1);color:var(--org);border:1px solid rgba(247,129,102,.22);}
.bp{background:rgba(188,140,255,.1);color:var(--purple);border:1px solid rgba(188,140,255,.22);}
.by{background:rgba(227,179,65,.1);color:var(--yellow);border:1px solid rgba(227,179,65,.22);}
.bt{background:rgba(57,211,83,.1);color:var(--teal);border:1px solid rgba(57,211,83,.22);}

/* ── Info boxes ── */
.ib{background:var(--s2);border:1px solid var(--bd);border-left:3px solid var(--blue);
    border-radius:var(--r);padding:.75rem .95rem;font-size:.83rem;line-height:1.6;margin:.4rem 0;}
.ib.g{border-left-color:var(--green);}
.ib.o{border-left-color:var(--org);}
.ib.y{border-left-color:var(--yellow);}

/* ── Verdict boxes ── */
.vw{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.3);border-radius:var(--r);
    padding:.8rem 1rem;color:var(--green);}
.vo{background:rgba(247,129,102,.08);border:1px solid rgba(247,129,102,.3);border-radius:var(--r);
    padding:.8rem 1rem;color:var(--org);}
.vy{background:rgba(227,179,65,.1);border:1px solid rgba(227,179,65,.3);border-radius:var(--r);
    padding:.8rem 1rem;color:var(--yellow);}

/* ── Metric table ── */
.mt{width:100%;border-collapse:collapse;font-size:.81rem;}
.mt th{background:var(--s2);color:var(--blue);font-weight:600;padding:.45rem .65rem;
    border-bottom:1px solid var(--bd);text-align:left;}
.mt td{padding:.38rem .65rem;border-bottom:1px solid var(--bd2);color:var(--text);}
.mt tr:nth-child(even) td{background:rgba(255,255,255,.02);}
.mt .best{color:var(--green);font-weight:700;}
.mt .worst{color:var(--org);}

/* ── Rec card ── */
.rc{background:var(--s2);border:1px solid var(--bd);border-radius:var(--r);
    padding:.85rem 1rem;margin-bottom:.5rem;}
.rc-rank{font-size:.68rem;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.rc-name{font-size:.9rem;font-weight:600;margin:.15rem 0;}

/* ── Model learn link ── */
.mk-link{display:inline-block;background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.2);
    color:var(--blue)!important;border-radius:5px;padding:.22rem .65rem;
    font-size:.75rem;margin:.2rem .2rem 0 0;text-decoration:none;}

/* ── Dataset card ── */
.ds-card{background:var(--s2);border:1px solid var(--bd);border-radius:var(--r);
    padding:.9rem 1.1rem;margin-bottom:.6rem;}

/* ── Download button ── */
.dlbtn{display:inline-block;background:var(--blue);color:#0d1117!important;
    font-weight:600;font-size:.82rem;padding:.5rem 1.2rem;border-radius:6px;text-decoration:none;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--bd2);}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PAL   = ["#58a6ff","#3fb950","#f78166","#bc8cff","#e3b341",
         "#79c0ff","#56d364","#ffa198","#d2a8ff","#f0883e"]
RFILL = ["rgba(88,166,255,.18)","rgba(63,185,80,.18)","rgba(247,129,102,.18)",
         "rgba(188,140,255,.18)","rgba(227,179,65,.18)","rgba(121,192,255,.18)",
         "rgba(86,211,100,.18)","rgba(255,161,152,.18)","rgba(194,161,245,.18)",
         "rgba(240,136,62,.18)"]
_BG   = "rgba(0,0,0,0)"
_GRID = "#21262d"
_TICK = "#7d8590"
_FT   = dict(family="Inter", color="#e6edf3")


# ── Helper functions ──────────────────────────────────────────────────────────
def clr(i):
    return PAL[i % len(PAL)]


def kpi(label, value, sub="", cls=""):
    """Render a KPI card. cls: 'g'=green, 'o'=orange, 'p'=purple, 'y'=yellow, 't'=teal"""
    sub_html = f"<div class='ks'>{sub}</div>" if sub else ""
    return f"<div class='kc {cls}'><div class='kl'>{label}</div><div class='kv'>{value}</div>{sub_html}</div>"


def badge(text, cls="bb"):
    return f"<span class='b {cls}'>{text}</span>"


def ibox(content, cls=""):
    return f"<div class='ib {cls}'>{content}</div>"


def sh(icon, txt, num=None):
    n = f"<div class='sh-n'>{num}</div>" if num else ""
    return f"<div class='sh'>{n}{icon} {txt}</div>"


def ni(val, best, worst):
    if abs(best - worst) < 1e-9:
        return 0.5
    return float(max(0.0, min(1.0, (worst - val) / (worst - best))))


def PT(h=400):
    return dict(height=h, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FT,
                xaxis=dict(gridcolor=_GRID, linecolor=_GRID,
                           tickfont=dict(color=_TICK), zerolinecolor=_GRID),
                yaxis=dict(gridcolor=_GRID, linecolor=_GRID,
                           tickfont=dict(color=_TICK), zerolinecolor=_GRID),
                legend=dict(font=dict(color="#c9d1d9"), bgcolor=_BG,
                            bordercolor=_GRID, borderwidth=1),
                margin=dict(l=50, r=20, t=45, b=45),
                hoverlabel=dict(bgcolor="#1c2128", font=dict(color="#e6edf3"),
                                bordercolor=_GRID))


def verdict_html(fa):
    v   = fa.get("verdict", "unknown")
    cls = {"well_fitted":"vw","overfitting":"vo",
           "mild_overfitting":"vo","underfitting":"vy"}.get(v, "ib")
    ico = {"well_fitted":"✅","overfitting":"⚠️",
           "mild_overfitting":"🔶","underfitting":"📉"}.get(v, "❓")
    tr  = fa.get("train_score")
    te  = fa.get("test_score")
    gap = fa.get("gap")
    sc  = (f"<br><small>Train: {tr:.4f} | Test: {te:.4f} | Gap: {gap:.4f}</small>"
           if tr is not None else "")
    return (f"<div class='{cls}'><b>{ico} {v.replace('_',' ').title()}</b>"
            f" — {fa.get('recommendation','')}{sc}</div>")


def metric_tbl(results, task):
    if task == "Regression":
        mets = [("R2","r2",True),("RMSE","rmse",False),("MAE","mae",False),
                ("Runtime(s)","runtime",False),("Mem(MB)","memory_mb",False),
                ("Size(MB)","model_size_mb",False)]
    else:
        mets = [("Accuracy","accuracy",True),("F1","f1",True),
                ("Precision","precision",True),("Recall","recall",True),
                ("AUC","auc",True),("Runtime(s)","runtime",False),
                ("Size(MB)","model_size_mb",False)]
    names = [r["name"] for r in results]
    hdr   = "<tr><th>Metric</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    body  = ""
    for label, key, hb in mets:
        vals = [float(r.get(key) or 0) for r in results]
        if not any(v != 0 for v in vals):
            continue
        bv = max(vals) if hb else min(vals)
        wv = min(vals) if hb else max(vals)
        cells = "".join(
            f"<td class='{'best' if v==bv else 'worst' if v==wv else ''}'>{v:.4f}</td>"
            for v in vals)
        body += f"<tr><td><b>{label}</b></td>{cells}</tr>"
    return f"<table class='mt'><thead>{hdr}</thead><tbody>{body}</tbody></table>"


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("sessions", []), ("active_session", None),
              ("ds_registry", None), ("cross_analysis", None),
              ("enriched_df", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.ds_registry is None:
    st.session_state.ds_registry = DatasetRegistry()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 CropML Studio")
    st.caption("v5.0 · Research ML Platform")
    st.divider()
    page = st.radio("",
                    ["🔬 Train", "🤖 AutoML Focus", "🗂️ Multi-Dataset",
                     "📊 Analysis", "📐 Compare", "📚 Model Library", "📁 History"],
                    label_visibility="collapsed")
    st.divider()
    disk     = load_sessions()
    existing = {s["id"] for s in st.session_state.sessions}
    for ds in disk:
        if ds["id"] not in existing:
            st.session_state.sessions.append(ds)
            existing.add(ds["id"])
    st.markdown(f"**Sessions:** `{len(st.session_state.sessions)}`")
    if st.session_state.sessions:
        labels = [f"{s['name']} ({s['timestamp'][:10]})"
                  for s in st.session_state.sessions]
        idx = st.selectbox("Active", range(len(labels)),
                           format_func=lambda i: labels[i])
        st.session_state.active_session = st.session_state.sessions[idx]
    st.divider()
    st.caption("AutoML · Multi-Dataset · PDF Export")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Train":
    st.markdown("# 🔬 Train New Session")
    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to begin. Raw or pre-processed data both supported.")
        st.stop()

    df_raw   = pd.read_csv(uploaded)
    profiler = DataProfiler(df_raw)
    profile  = profiler.profile()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Rows",        f"{profile['n_rows']:,}"),                     unsafe_allow_html=True)
    c2.markdown(kpi("Columns",     profile["n_cols"]),                            unsafe_allow_html=True)
    c3.markdown(kpi("Numeric",     len(profile["numeric_cols"]),     cls="g"),    unsafe_allow_html=True)
    c4.markdown(kpi("Categorical", len(profile["categorical_cols"]), cls="o"),    unsafe_allow_html=True)
    c5.markdown(kpi("Nulls",       profile["null_total"],            cls="p"),    unsafe_allow_html=True)
    st.markdown("")

    with st.expander("Preview & column details"):
        st.dataframe(df_raw.head(8), use_container_width=True)
        st.dataframe(pd.DataFrame(profile["col_details"]), use_container_width=True)

    if profile["looks_preprocessed"]:
        st.success("✅ Data looks pre-processed (all numeric, no nulls).")
    else:
        st.warning("⚠️ Raw data detected — configure preprocessing below.")

    # ── Task & columns ──
    st.markdown(sh("⚙️", "Task & Column Configuration", "1"), unsafe_allow_html=True)
    ca, cb, cc = st.columns([1, 2, 2])
    task      = ca.selectbox("Task Type", ["Regression", "Classification"])
    target    = cb.selectbox("Target Column", df_raw.columns.tolist(),
                              index=len(df_raw.columns)-1)
    drop_cols = cc.multiselect("Drop Columns (IDs, leakage etc.)",
                                [c for c in df_raw.columns if c != target])

    # ── Preprocessing ──
    st.markdown(sh("🔧", "Preprocessing Pipeline", "2"), unsafe_allow_html=True)
    already_clean = st.checkbox(
        "Data is already preprocessed — skip encoding & imputation",
        value=profile["looks_preprocessed"])
    if not already_clean:
        pc1, pc2, pc3, pc4 = st.columns(4)
        hc = pc1.selectbox("Categorical", ["label","onehot","drop","none"],
                            format_func=lambda x: {"label":"Label Encode","onehot":"One-Hot",
                                                    "drop":"Drop","none":"None"}[x])
        hn = pc2.selectbox("Nulls", ["mean","median","mode","drop","none"],
                            format_func=lambda x: {"mean":"Mean Fill","median":"Median",
                                                    "mode":"Mode","drop":"Drop Rows","none":"None"}[x])
        hd = pc3.checkbox("Remove duplicates")
        lv = pc4.checkbox("Drop low-variance features")
        sm = st.selectbox("Scaling", ["none","standard","minmax"],
                           format_func=lambda x: {"none":"No Scaling",
                                                   "standard":"StandardScaler",
                                                   "minmax":"MinMaxScaler"}[x])
    else:
        hc = hn = "none"; hd = lv = False; sm = "none"
        st.info("Preprocessing skipped — features passed directly to models.")

    pipe_cfg = {"target_col":target,"drop_cols":drop_cols,"handle_cats":hc,
                "handle_nulls":hn,"handle_dups":hd,"drop_low_var":lv,
                "scale_method":sm,"already_clean":already_clean}
    pipeline = PreprocessingPipeline(pipe_cfg)
    try:
        X, y = pipeline.fit_transform(df_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}"); st.stop()

    final_feats = pipeline.feature_names
    with st.expander("📝 Preprocessing log"):
        for line in pipeline.get_log():
            st.markdown(f"• {line}")
    st.caption(f"**{len(final_feats)} features** → "
               f"{', '.join(final_feats[:12])}{'...' if len(final_feats)>12 else ''}")
    if task == "Classification":
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

    # ── Smart Recommendation ──
    st.markdown(sh("🤖", "Smart Model Recommendation", "3"), unsafe_allow_html=True)
    rec_mode = st.radio("Mode",
                         ["🧠 Auto (dataset-aware)", "📏 Rule-based", "✋ Manual"],
                         horizontal=True)
    user_rules = {}
    if rec_mode == "📏 Rule-based":
        r1, r2, r3, r4 = st.columns(4)
        user_rules["prefer_speed"]         = r1.checkbox("⚡ Speed")
        user_rules["prefer_accuracy"]      = r2.checkbox("🎯 Accuracy")
        user_rules["prefer_interpretable"] = r3.checkbox("🔍 Interpretable")
        user_rules["small_model"]          = r4.checkbox("💾 Small model")

    registry = REGRESSION_MODELS if task == "Regression" else CLASSIFICATION_MODELS

    if rec_mode != "✋ Manual":
        rec_out = recommend_models(df_raw, target, task, user_rules or None)
        ranked  = rec_out["ranked"]
        top5    = [n for n, _ in ranked[:5]]
        dp      = rec_out["dataset_profile"]

        cat_pct = str(round(dp["cat_ratio"] * 100)) + "%"
        null_b  = badge("Has nulls", "bo") if dp["has_nulls"] else badge("No nulls", "bg")
        cat_b   = badge("Cat ratio " + cat_pct, "bp")
        dp_str  = (f"📊 <b>Dataset profile:</b> {dp['n_rows']:,} rows · "
                   f"{dp['n_features']} features · {null_b} · {cat_b}")
        st.markdown(ibox(dp_str, ""), unsafe_allow_html=True)

        st.markdown("**Top 5 recommended models:**")
        medals   = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        rec_cols = st.columns(5)
        for col, (name, score), medal in zip(rec_cols, ranked[:5], medals):
            mk_info = MODEL_KNOWLEDGE.get(registry.get(name, "rf"), {})
            col.markdown(
                f"<div class='rc'>"
                f"<div class='rc-rank'>{medal} score {score}</div>"
                f"<div class='rc-name'>{name}</div>"
                f"<div style='font-size:.7rem;color:#7d8590;'>{mk_info.get('type','')}</div>"
                f"</div>",
                unsafe_allow_html=True)

        with st.expander("Full recommendation details & reasoning"):
            for name, score in ranked:
                reasons = rec_out["reasons"][name]
                mk_info = MODEL_KNOWLEDGE.get(registry.get(name, "rf"), {})
                st.markdown(f"**{name}** — Score: `{score}` · {mk_info.get('type','')}")
                for r in reasons:
                    st.markdown(f"  • {r}")
                st.markdown(f"  ✅ Best for: {mk_info.get('best_for','')}")
                st.divider()
        default_sel = top5[:4]
    else:
        default_sel = list(registry.keys())[:4]

    # ── Model selection ──
    st.markdown(sh("✅", "Select Models to Train", "4"), unsafe_allow_html=True)
    selected = st.multiselect("Models (select ≥1)", list(registry.keys()),
                               default=[d for d in default_sel if d in registry])
    if not selected:
        st.warning("Select at least one model."); st.stop()

    # ── Hyperparameters ──
    st.markdown(sh("🎛️", "Hyperparameter Configuration", "5"), unsafe_allow_html=True)
    scale_models = st.checkbox(
        "Scale features for model training (recommended for SVR/SVC/KNN/MLP)", value=True)
    model_cfgs = {}
    tabs = st.tabs(selected)
    for tab, mname in zip(tabs, selected):
        with tab:
            mk_info = MODEL_KNOWLEDGE.get(registry.get(mname, "rf"), {})
            if mk_info:
                st.caption(f"**{mk_info.get('type','')}** — {mk_info.get('short','')}")
            mkey, mparams = hyperparam_ui(mname, f"hp_{mname}", task)
            model_cfgs[mname] = (mkey, mparams)

    # ── Launch ──
    st.markdown(sh("🚀", "Launch Training", "6"), unsafe_allow_html=True)
    sn1, sn2, sn3 = st.columns([3, 1, 1])
    sname = sn1.text_input(
        "Session Name",
        value=f"{uploaded.name.replace('.csv','')}_{task[:3]}_{datetime.now().strftime('%H%M')}")
    tsize = sn2.slider("Test size", .10, .40, .20, .05)
    rseed = sn3.number_input("Seed", 0, 9999, 42)

    if st.button("🚀 Train All Selected Models", type="primary", use_container_width=True):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=tsize,
                                                    random_state=int(rseed))
        prog = st.progress(0, "Starting…"); results = []
        for i, (mname, (mkey, mparams)) in enumerate(model_cfgs.items()):
            prog.progress(i / len(model_cfgs), text=f"Training {mname}…")
            try:
                r = train_one(mname, mkey, mparams,
                              X_tr.copy(), X_te.copy(), y_tr, y_te,
                              task, scale_models, PERSIST_DIR)
                results.append(r)
            except Exception as e:
                st.warning(f"⚠️ {mname} failed: {e}")
        prog.progress(1.0, text="✅ Done")

        session = {"id":make_sid(sname),"name":sname,
                   "timestamp":datetime.now().isoformat(),
                   "dataset":uploaded.name,"task":task,
                   "n_samples":len(X),"n_features":len(final_feats),
                   "feature_names":final_feats,"target_col":target,
                   "test_size":tsize,"scale":scale_models,
                   "y_test":y_te.tolist(),"model_results":results,
                   "prep_log":pipeline.get_log()}
        st.session_state.sessions.append(session)
        st.session_state.active_session = session
        save_session(session)
        st.success(f"✅ {len(results)} model(s) trained. Session **{sname}** saved.")

        mk2       = "r2" if task == "Regression" else "accuracy"
        best_n    = max(results, key=lambda r: r.get(mk2, 0))["name"]
        res_cols  = st.columns(min(len(results), 6))
        for col, r in zip(res_cols, results):
            star = " ⭐" if r["name"] == best_n else ""
            c    = "g" if r["name"] == best_n else ""
            lbl  = "R²" if task == "Regression" else "Acc"
            col.markdown(kpi(r["name"]+star, f"{r.get(mk2,0):.4f}", lbl, cls=c),
                         unsafe_allow_html=True)
        st.info("➡️ Go to **📊 Analysis** or **🤖 AutoML Focus** for deeper results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — AUTOML FOCUS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AutoML Focus":
    st.markdown("# 🤖 AutoML Focus Engine")
    st.markdown(ibox(
        "Benchmarks every eligible model via cross-validation → identifies the best → "
        "runs <b>Optuna hyperparameter search</b> focused entirely on it → "
        "diagnoses over/underfitting with a concrete fix recommendation.", ""),
        unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"], key="aml_up")
    if not uploaded:
        st.info("Upload a CSV to run AutoML."); st.stop()

    df_raw   = pd.read_csv(uploaded)
    profiler = DataProfiler(df_raw)
    profile  = profiler.profile()

    ca, cb, cc, cd = st.columns(4)
    task   = ca.selectbox("Task",   ["Regression","Classification"], key="aml_task")
    target = cb.selectbox("Target", df_raw.columns.tolist(),
                           index=len(df_raw.columns)-1, key="aml_tgt")
    tsize  = cc.slider("Test size", .10, .40, .20, .05, key="aml_ts")
    rseed  = cd.number_input("Seed", 0, 9999, 42, key="aml_seed")

    already_clean = st.checkbox("Data already preprocessed",
                                 value=profile["looks_preprocessed"], key="aml_ac")
    pipe_cfg = {"target_col":target,"drop_cols":[],
                "handle_cats":"label" if not already_clean else "none",
                "handle_nulls":"mean" if not already_clean else "none",
                "handle_dups":False,"drop_low_var":False,
                "scale_method":"none","already_clean":already_clean}
    pipeline = PreprocessingPipeline(pipe_cfg)
    try:
        X, y = pipeline.fit_transform(df_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}"); st.stop()
    if task == "Classification":
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))
    final_feats = pipeline.feature_names
    st.caption(f"**{len(final_feats)} features** · **{len(X):,} rows**")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=tsize,
                                               random_state=int(rseed))

    st.markdown(sh("⚙️", "AutoML Settings", "1"), unsafe_allow_html=True)
    ac1, ac2, ac3, ac4 = st.columns(4)
    cv_folds  = ac1.slider("CV Folds",     2,  10,   5, key="aml_cv")
    n_trials  = ac2.slider("Optuna Trials",20, 200, 60, 10, key="aml_nt")
    timeout_s = ac3.slider("Timeout (s)",  30, 300,120, 10, key="aml_to")
    scale_aml = ac4.checkbox("Scale features", value=True, key="aml_sc")

    override       = st.checkbox("Override — manually pick which model to focus on", key="aml_ov")
    override_model = None
    if override:
        reg2           = REGRESSION_MODELS if task=="Regression" else CLASSIFICATION_MODELS
        override_model = st.selectbox("Focus on model", list(reg2.keys()), key="aml_om")

    if st.button("🚀 Run AutoML Focus", type="primary", use_container_width=True):
        with st.spinner("Step 1/3 — Benchmarking all models via cross-validation…"):
            bench  = run_benchmark(X, y, task, scale=scale_aml, cv_folds=cv_folds)
        ranked = bench["benchmark"]; metric = bench["metric"]

        st.markdown(sh("📊", "Benchmark Results", "2"), unsafe_allow_html=True)
        df_bench = pd.DataFrame(ranked)
        fig_bench = go.Figure([go.Bar(
            x=df_bench["model_name"], y=df_bench["mean_score"],
            marker=dict(color=[clr(i) for i in range(len(df_bench))], line=dict(width=0)),
            error_y=dict(type="data", array=df_bench["std_score"].tolist(),
                         visible=True, color="#7d8590"),
            text=[f"{v:.4f}" for v in df_bench["mean_score"]],
            textposition="outside")])
        fig_bench.update_layout(
            title=f"Benchmark {metric} (CV={cv_folds}) — error bars = std deviation",
            **PT(380))
        st.plotly_chart(fig_bench, use_container_width=True)

        best_name = override_model or ranked[0]["model_name"]
        best_cv   = next((r["mean_score"] for r in ranked if r["model_name"]==best_name),
                          ranked[0]["mean_score"])
        st.markdown(ibox(
            f"🎯 <b>Focusing on: {best_name}</b> (CV {metric} = {best_cv:.4f}). "
            f"Running {n_trials} Optuna trials with {timeout_s}s timeout…", "g"),
            unsafe_allow_html=True)

        trial_scores = []
        prog         = st.progress(0, "Starting Optuna…")

        def progress_cb(trial_num, total, score):
            trial_scores.append({"trial": trial_num, "score": score})
            pct      = min(1.0, trial_num / max(total, 1))
            best_now = max(s["score"] for s in trial_scores)
            prog.progress(pct, text=f"Trial {trial_num}/{total} — best: {best_now:.4f}")

        with st.spinner(f"Step 2/3 — Optuna tuning {best_name}…"):
            aml_result = focus_tune(
                best_name, X, y, task, scale=scale_aml,
                n_trials=n_trials, timeout_s=timeout_s,
                cv_folds=cv_folds, progress_callback=progress_cb)
        prog.progress(1.0, text="✅ Tuning complete")

        if "error" in aml_result:
            st.error(aml_result["error"]); st.stop()

        hist = aml_result["trial_history"]
        if hist:
            df_hist     = pd.DataFrame(hist)
            best_so_far = [max(df_hist["score"].iloc[:i+1]) for i in range(len(df_hist))]
            fig_hist    = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df_hist["trial"], y=df_hist["score"], mode="markers",
                marker=dict(color="#58a6ff", size=4, opacity=.5), name="Trial score"))
            fig_hist.add_trace(go.Scatter(
                x=df_hist["trial"], y=best_so_far, mode="lines",
                line=dict(color="#3fb950", width=2), name="Best so far"))
            fig_hist.update_layout(title="Optuna Trial History",
                                    xaxis_title="Trial", yaxis_title=metric, **PT(340))
            st.plotly_chart(fig_hist, use_container_width=True)

        fa = aml_result["fit_assessment"]
        st.markdown(sh("🧪", "Fit Quality Assessment", "3"), unsafe_allow_html=True)
        st.markdown(verdict_html(fa), unsafe_allow_html=True)

        bp = aml_result["best_params"]
        st.markdown(sh("🏆", "Best Hyperparameters Found", ""), unsafe_allow_html=True)
        cp1, cp2 = st.columns([2, 1])
        with cp1:
            st.json(bp)
        with cp2:
            st.markdown(kpi(f"Best CV {metric}",
                             f"{aml_result['best_cv_score']:.4f}",
                             f"{aml_result['n_trials']} trials", cls="g"),
                         unsafe_allow_html=True)
            if fa.get("train_score") is not None:
                st.markdown("")
                st.markdown(kpi("Train Score", f"{fa['train_score']:.4f}", cls=""),
                             unsafe_allow_html=True)
                st.markdown("")
                st.markdown(kpi("Test Score",  f"{fa['test_score']:.4f}", "final eval", cls="g"),
                             unsafe_allow_html=True)

        if st.button("✅ Train & Save Final AutoML Model", key="aml_save"):
            with st.spinner("Training final tuned model…"):
                r_final = train_focused_model(
                    best_name, bp, X_tr.copy(), X_te.copy(),
                    y_tr, y_te, task, scale_aml)
            sname_aml = f"AutoML_{best_name.replace(' ','_')}_{datetime.now().strftime('%H%M')}"
            session   = {"id":make_sid(sname_aml),"name":sname_aml,
                         "timestamp":datetime.now().isoformat(),
                         "dataset":uploaded.name,"task":task,
                         "n_samples":len(X),"n_features":len(final_feats),
                         "feature_names":final_feats,"target_col":target,
                         "test_size":tsize,"scale":scale_aml,
                         "y_test":y_te.tolist(),"model_results":[r_final],
                         "prep_log":pipeline.get_log(),"automl_info":aml_result}
            st.session_state.sessions.append(session)
            st.session_state.active_session = session
            save_session(session)
            mk2 = "r2" if task=="Regression" else "accuracy"
            st.success(f"✅ AutoML model saved! {metric} = {r_final.get(mk2,0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — MULTI-DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Multi-Dataset":
    st.markdown("# 🗂️ Multi-Dataset Analysis Engine")
    st.markdown(ibox(
        "Upload multiple related datasets. Detect common merge keys, analyse "
        "cross-dataset correlations, and find which external features most boost "
        "your model's accuracy.", ""), unsafe_allow_html=True)

    reg: DatasetRegistry = st.session_state.ds_registry

    st.markdown(sh("📁", "Upload Datasets", "1"), unsafe_allow_html=True)
    multi_files = st.file_uploader(
        "Upload one or more CSV files", type=["csv"],
        accept_multiple_files=True, key="multi_up")
    if multi_files:
        for f in multi_files:
            if f.name not in reg.all_labels():
                reg.add(f.name, pd.read_csv(f))
                st.toast(f"Added: {f.name}")

    if not reg.all_labels():
        st.info("Upload at least two CSV files to enable analysis."); st.stop()

    summary = reg.summary()
    st.markdown("**Loaded Datasets:**")
    for row in summary:
        ca2, cb2, cc2 = st.columns([4, 5, 1])
        ca2.markdown(
            f"<div class='ds-card'><b>{row['Dataset']}</b><br>"
            f"<span style='font-size:.75rem;color:#7d8590;'>"
            f"{row['Rows']:,} rows · {row['Cols']} cols · "
            f"{row['Numeric']} numeric · {row['Nulls']} nulls</span></div>",
            unsafe_allow_html=True)
        with cc2:
            if st.button("🗑️", key=f"rm_{row['Dataset']}"):
                reg.remove(row["Dataset"]); st.rerun()

    if len(reg.all_labels()) < 2:
        st.info("Upload at least 2 datasets to enable analysis."); st.stop()

    st.markdown(sh("🔗", "Common Column Detection", "2"), unsafe_allow_html=True)
    common    = find_common_columns(reg.datasets)
    merge_sug = suggest_merge_keys(reg.datasets)
    if common:
        st.markdown(f"Found **{len(common)} common column(s)** across datasets:")
        st.dataframe(
            pd.DataFrame([{"Column":k,"Present in":", ".join(v),"Count":len(v)}
                           for k,v in common.items()]),
            use_container_width=True, hide_index=True)
    else:
        st.warning("No exact column name matches found across datasets.")

    st.markdown(sh("🔀", "Merge Configuration", "3"), unsafe_allow_html=True)
    primary_label = st.selectbox("Primary dataset", reg.all_labels())
    default_keys  = [s["column"] for s in merge_sug if s["looks_like_key"]][:3]
    selected_keys = st.multiselect("Merge keys", list(common.keys()), default=default_keys)
    merge_how     = st.selectbox(
        "Join type", ["inner","left","outer"],
        format_func=lambda x: {"inner":"Inner (matching rows only)",
                                "left":"Left (keep all primary rows)",
                                "outer":"Outer (keep all rows)"}[x])

    if selected_keys and st.button("🔀 Merge Datasets", key="do_merge"):
        merged_df, merge_log = merge_datasets(
            primary_label, reg.datasets, selected_keys, how=merge_how)
        st.session_state.enriched_df = merged_df
        reg.add("__merged__", merged_df)
        for line in merge_log: st.markdown(f"• {line}")
        st.success(f"Merged: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")
        with st.expander("Preview merged dataset"):
            st.dataframe(merged_df.head(10), use_container_width=True)
        st.download_button("⬇ Download Merged CSV",
                            merged_df.to_csv(index=False),
                            file_name="merged_dataset.csv", mime="text/csv")

    st.markdown(sh("🔬", "Cross-Dataset Correlation Analysis", "4"), unsafe_allow_html=True)
    prim_df  = reg.get(primary_label)
    if prim_df is None: st.stop()
    pcols    = prim_df.columns.tolist()
    tgt_cd   = st.selectbox("Target column (what to predict)", pcols,
                              index=len(pcols)-1, key="cd_tgt")
    cd_task  = st.selectbox("Task type", ["Regression","Classification"], key="cd_task")

    if st.button("🔬 Analyse Cross-Dataset Impact", key="do_cross"):
        external = {l: df for l, df in reg.datasets.items()
                    if l != primary_label and l != "__merged__"}
        if not external:
            st.warning("Need at least 2 datasets."); st.stop()
        with st.spinner("Analysing correlation & feature importance across datasets…"):
            analysis = cross_dataset_correlation(prim_df, tgt_cd, external, cd_task)
        st.session_state.cross_analysis = analysis

    if st.session_state.cross_analysis:
        analysis = st.session_state.cross_analysis
        st.markdown(sh("📈", "Impact Analysis Results", ""), unsafe_allow_html=True)

        for label, info in analysis.items():
            if "error" in info:
                st.warning(f"{label}: {info['error']}"); continue
            delta    = info["score_delta"]
            delta_s  = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            bs_b     = f"Base: {info['baseline_score']:.4f}"
            ws_b     = f"With: {info['augmented_score']:.4f}"
            nr_b     = f"{info['n_rows_analysed']:,} rows"
            wc       = "bg" if delta >= 0 else "bo"
            dc       = "bt" if delta >= 0 else "bo"
            st.markdown(
                f"<div class='ds-card'><b>{label}</b><br>"
                f"<span style='font-size:.83rem;color:#c9d1d9;'>{info['verdict_text']}</span><br><br>"
                f"{badge(bs_b,'bb')}&nbsp;{badge(ws_b,wc)}&nbsp;"
                f"{badge('Δ '+delta_s,dc)}&nbsp;{badge(nr_b,'bp')}</div>",
                unsafe_allow_html=True)

            ti = info.get("top_ext_importances", {})
            if ti:
                df_ti = pd.DataFrame({"Feature":list(ti.keys()),
                                       "Importance":list(ti.values())}).sort_values("Importance")
                fig_ti = px.bar(df_ti, x="Importance", y="Feature", orientation="h",
                                 color_discrete_sequence=["#58a6ff"],
                                 title=f"{label} — Top Features when Merged")
                fig_ti.update_layout(**PT(max(200, len(df_ti)*28)))
                st.plotly_chart(fig_ti, use_container_width=True)

            tc = info.get("top_correlations", [])
            if tc and cd_task == "Regression":
                df_corr = pd.DataFrame(tc[:8])
                fig_corr = px.bar(df_corr, x="pearson_r", y="feature", orientation="h",
                                   color="pearson_r", color_continuous_scale="RdBu",
                                   range_color=[-1,1],
                                   title=f"{label} — Pearson Correlation with Target")
                fig_corr.update_coloraxes(showscale=False)
                fig_corr.update_layout(**PT(max(200, len(df_corr)*28)))
                st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(sh("💡", "Enrichment Recommendations", "5"), unsafe_allow_html=True)
        mk2        = "r2" if cd_task == "Regression" else "accuracy"
        sess_score = 0.0
        if st.session_state.active_session:
            sess_score = max(
                (r.get(mk2, 0) for r in st.session_state.active_session.get("model_results",[])),
                default=0.)
        recs = recommend_enrichment(analysis, sess_score)
        for rec in recs:
            pb  = {"HIGH":"bt","MEDIUM":"by","LOW":"bo"}.get(rec["priority"],"bb")
            kf  = ", ".join(rec["key_features"][:5]) or "N/A"
            st.markdown(
                f"<div class='ds-card'>{badge(rec['priority'],pb)}&nbsp;"
                f"<b>{rec['dataset']}</b> — {rec['verdict_text']}<br>"
                f"<small style='color:#7d8590;'>Key features: {kf} · "
                f"Projected score: {rec['projected_score']:.4f}</small></div>",
                unsafe_allow_html=True)

        st.markdown(sh("🧩", "Build Enriched Training Dataset", "6"), unsafe_allow_html=True)
        selected_cols = {}
        for label in [l for l in reg.all_labels()
                       if l != primary_label and l != "__merged__"]:
            edf = reg.get(label)
            if edf is None: continue
            nc      = edf.select_dtypes(include=[np.number]).columns.tolist()
            def_c   = list(analysis.get(label,{}).get("top_ext_importances",{}).keys())[:5]
            chosen  = st.multiselect(f"Columns from **{label}**", nc,
                                      default=def_c, key=f"en_{label}")
            if chosen: selected_cols[label] = chosen

        if selected_cols and st.button("🧩 Build Enriched Dataset", key="do_enrich"):
            enr_df, enr_log = build_enriched_dataset(
                prim_df, tgt_cd,
                {l: reg.get(l) for l in selected_cols}, selected_cols)
            st.session_state.enriched_df = enr_df
            reg.add("__enriched__", enr_df)
            for line in enr_log: st.markdown(f"• {line}")
            st.success("✅ Enriched dataset ready!")
            st.download_button("⬇ Download Enriched CSV",
                                enr_df.to_csv(index=False),
                                file_name="enriched_dataset.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analysis":
    st.markdown("# 📊 Session Analysis")
    if not st.session_state.active_session:
        st.info("No active session — train one first."); st.stop()

    s       = st.session_state.active_session
    results = s["model_results"]
    task    = s["task"]
    feats   = s["feature_names"]
    y_test  = np.array(s.get("y_test", []))

    # Header
    ns = f"{s['n_samples']:,} rows"
    nf = f"{s['n_features']} features"
    nr = f"{len(results)} model(s)"
    am = badge("AutoML","bt") if s.get("automl_info") else ""
    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem;'>
      <div style='flex:1;'>
        <div style='font-size:.68rem;color:#7d8590;text-transform:uppercase;
                    letter-spacing:.1em;'>Active Session</div>
        <div style='font-size:1.5rem;font-weight:700;letter-spacing:-.025em;'>{s['name']}</div>
        <div style='margin-top:.35rem;'>
          {badge(s['dataset'],'bb')}&nbsp;{badge(task,'bg')}&nbsp;
          {badge(ns,'bo')}&nbsp;{badge(nf,'bp')}&nbsp;{badge(nr,'bt')}&nbsp;{am}
        </div>
      </div>
      <div style='text-align:right;font-size:.75rem;color:#7d8590;padding-top:.5rem;'>
        {s['timestamp'][:19]}
      </div>
    </div>""", unsafe_allow_html=True)

    if s.get("automl_info"):
        fa = s["automl_info"].get("fit_assessment", {})
        if fa.get("verdict"):
            st.markdown(verdict_html(fa), unsafe_allow_html=True)

    if PDF_OK and st.button("📥 Download PDF Report"):
        with st.spinner("Building PDF…"):
            try:
                pdf   = generate_pdf(s)
                b64   = base64.b64encode(pdf).decode()
                fname = f"{s['name']}_report.pdf"
                st.markdown(
                    f"<a class='dlbtn' href='data:application/pdf;base64,{b64}' "
                    f"download='{fname}'>⬇ Download {fname}</a>",
                    unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")
    st.divider()

    mk = "r2" if task == "Regression" else "accuracy"
    if task == "Regression":
        br = max(results, key=lambda r: r.get("r2",0))
        be = min(results, key=lambda r: r.get("rmse",999))
        bm = min(results, key=lambda r: r.get("mae",999))
        bf = min(results, key=lambda r: r["runtime"])
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi("Best R2",   f"{br.get('r2',0):.4f}",   br["name"], cls="g"), unsafe_allow_html=True)
        c2.markdown(kpi("Best RMSE", f"{be.get('rmse',0):.4f}", be["name"], cls=""),  unsafe_allow_html=True)
        c3.markdown(kpi("Best MAE",  f"{bm.get('mae',0):.4f}",  bm["name"], cls="p"), unsafe_allow_html=True)
        c4.markdown(kpi("Fastest",   f"{bf['runtime']:.3f}s",   bf["name"], cls="y"), unsafe_allow_html=True)
    else:
        ba  = max(results, key=lambda r: r.get("accuracy",0))
        bf1 = max(results, key=lambda r: r.get("f1",0))
        bft = min(results, key=lambda r: r["runtime"])
        bp2 = max(results, key=lambda r: r.get("precision",0))
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi("Best Accuracy",  f"{ba.get('accuracy',0):.4f}",  ba["name"],  cls="g"), unsafe_allow_html=True)
        c2.markdown(kpi("Best F1",        f"{bf1.get('f1',0):.4f}",       bf1["name"], cls=""),  unsafe_allow_html=True)
        c3.markdown(kpi("Best Precision", f"{bp2.get('precision',0):.4f}",bp2["name"], cls="p"), unsafe_allow_html=True)
        c4.markdown(kpi("Fastest",        f"{bft['runtime']:.3f}s",       bft["name"], cls="y"), unsafe_allow_html=True)
    st.markdown("")

    st.markdown(sh("📋","Metric Comparison Table",""), unsafe_allow_html=True)
    st.markdown("<small style='color:#7d8590;'>🟢 Green = best per metric · 🟠 Orange = worst</small>",
                unsafe_allow_html=True)
    st.markdown(metric_tbl(results, task), unsafe_allow_html=True)
    st.markdown("")

    ml = "R2" if task=="Regression" else "Accuracy"
    st.markdown(sh("📊", f"{ml} Comparison",""), unsafe_allow_html=True)
    df_bar = (pd.DataFrame([{"Model":r["name"], ml:r.get(mk,0)} for r in results])
              .sort_values(ml, ascending=False))
    fig_bar = go.Figure([go.Bar(
        x=df_bar["Model"], y=df_bar[ml],
        marker=dict(color=[clr(i) for i in range(len(df_bar))], line=dict(width=0)),
        text=[f"{v:.4f}" for v in df_bar[ml]], textposition="outside")])
    yr = ([min(0, df_bar[ml].min()-.05), min(1.12, df_bar[ml].max()+.08)]
          if task=="Regression" else [0, min(1.12, df_bar[ml].max()+.08)])
    fig_bar.update_layout(title=f"{ml} — higher is better", yaxis_range=yr, **PT(400))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(sh("📉","Error / Classification Metrics",""), unsafe_allow_html=True)
    if task == "Regression":
        fig_err = make_subplots(rows=1, cols=2,
                                 subplot_titles=["RMSE (lower=better)","MAE (lower=better)"])
        ax = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TICK))
        for i, r in enumerate(results):
            for ci, met in [(1,"rmse"),(2,"mae")]:
                fig_err.add_trace(go.Bar(x=[r["name"]], y=[r.get(met,0)],
                                          marker_color=clr(i), showlegend=False,
                                          text=[f"{r.get(met,0):.4f}"],
                                          textposition="outside"), row=1, col=ci)
        fig_err.update_layout(height=400, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FT,
                               margin=dict(l=40,r=20,t=50,b=50),
                               xaxis=ax,yaxis=ax,xaxis2=ax,yaxis2=ax)
    else:
        fig_err = go.Figure()
        for met, cc2 in [("F1","#58a6ff"),("Precision","#3fb950"),("Recall","#f78166")]:
            fig_err.add_trace(go.Bar(
                name=met, x=[r["name"] for r in results],
                y=[r.get(met.lower(),0) for r in results],
                marker_color=cc2, opacity=.85))
        fig_err.update_layout(barmode="group", **PT(400))
    st.plotly_chart(fig_err, use_container_width=True)

    if task == "Regression":
        st.markdown(sh("🎯","Actual vs Predicted",""), unsafe_allow_html=True)
        sel_m = st.selectbox("Model", [r["name"] for r in results], key="avp")
        r_sel = next(r for r in results if r["name"]==sel_m)
        yp    = np.array(r_sel["y_pred"])
        if len(y_test) == len(yp):
            dmin = float(min(y_test.min(), yp.min()))
            dmax = float(max(y_test.max(), yp.max()))
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(x=y_test, y=yp, mode="markers",
                                          marker=dict(color="#58a6ff",size=5,opacity=.55,
                                                      line=dict(width=0)),name="Predictions"))
            fig_avp.add_trace(go.Scatter(x=[dmin,dmax],y=[dmin,dmax],mode="lines",
                                          line=dict(color="#ef4444",width=1.5,dash="dash"),
                                          name="Perfect fit"))
            fig_avp.update_layout(title=f"{sel_m} — Actual vs Predicted",
                                   xaxis_title="Actual",yaxis_title="Predicted",**PT(420))
            st.plotly_chart(fig_avp, use_container_width=True)

            t1, t2 = st.tabs(["Residual Scatter","Error Distribution"])
            with t1:
                res = y_test - yp
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=yp, y=res, mode="markers",
                                              marker=dict(color="#58a6ff",size=5,
                                                          opacity=.55,line=dict(width=0))))
                fig_res.add_hline(y=0, line_color="#ef4444", line_dash="dash", line_width=1.5)
                fig_res.update_layout(title="Residuals",
                                       xaxis_title="Predicted",yaxis_title="Residual",**PT(380))
                st.plotly_chart(fig_res, use_container_width=True)
            with t2:
                fig_rh = go.Figure()
                for i, r in enumerate(results):
                    if len(y_test) == len(r["y_pred"]):
                        fig_rh.add_trace(go.Histogram(
                            x=y_test-np.array(r["y_pred"]),
                            name=r["name"], opacity=.6, nbinsx=40, marker_color=clr(i)))
                fig_rh.update_layout(barmode="overlay",
                                      title="Residual Distribution",**PT(380))
                st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.markdown(sh("🔲","Confusion Matrix",""), unsafe_allow_html=True)
        cm_sel = st.selectbox("Model",[r["name"] for r in results],key="cm_s")
        r_cm   = next(r for r in results if r["name"]==cm_sel)
        fig_cm = px.imshow(np.array(r_cm["conf_matrix"]),text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted",y="Actual"),
                            title=f"{cm_sel} — Confusion Matrix")
        fig_cm.update_layout(**{k:v for k,v in PT(380).items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig_cm, use_container_width=True)

    fi_res = [r for r in results if r.get("feature_importances")]
    if fi_res:
        st.markdown(sh("🔑","Feature Importance",""), unsafe_allow_html=True)
        fi_tabs = st.tabs([r["name"] for r in fi_res])
        for tab, r in zip(fi_tabs, fi_res):
            with tab:
                df_fi = (pd.DataFrame({"Feature":feats,"Importance":r["feature_importances"]})
                         .nlargest(20,"Importance").sort_values("Importance"))
                fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                                 color="Importance",
                                 color_continuous_scale=[[0,"#1c2128"],[1,clr(results.index(r))]],
                                 title=f"{r['name']} — Feature Importance (Top 20)")
                fig_fi.update_coloraxes(showscale=False)
                fig_fi.update_layout(**PT(max(300, len(df_fi)*26)))
                st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(sh("⚙️","Computational Performance",""), unsafe_allow_html=True)
    fig_comp = make_subplots(rows=1,cols=3,
                              subplot_titles=["Runtime (s)","Memory (MB)","Model Size (MB)"])
    ax2 = dict(gridcolor=_GRID,linecolor=_GRID,tickfont=dict(color=_TICK,size=8))
    for i, r in enumerate(results):
        c = clr(i)
        for ci, key2, fmt in [(1,"runtime",".3f"),(2,"memory_mb",".1f"),(3,"model_size_mb",".3f")]:
            fig_comp.add_trace(go.Bar(x=[r["name"]],y=[r[key2]],marker_color=c,
                                       showlegend=False,text=[f"{r[key2]:{fmt}}"],
                                       textposition="outside"),row=1,col=ci)
    fig_comp.update_layout(height=380,plot_bgcolor=_BG,paper_bgcolor=_BG,font=_FT,
                            margin=dict(l=40,r=20,t=50,b=50),
                            xaxis=ax2,yaxis=ax2,xaxis2=ax2,yaxis2=ax2,xaxis3=ax2,yaxis3=ax2)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(sh("🕸️","Model Radar Profile",""), unsafe_allow_html=True)
    if task == "Regression":
        dims = ["R2","Low RMSE","Low MAE","Speed","Mem Eff","Size Eff"]
        def sc(r):
            return [max(0.,min(1.,r.get("r2",0))),
                    ni(r.get("rmse",0),min(x.get("rmse",0) for x in results),max(x.get("rmse",0) for x in results)),
                    ni(r.get("mae",0), min(x.get("mae",0)  for x in results),max(x.get("mae",0)  for x in results)),
                    ni(r["runtime"],   min(x["runtime"]     for x in results),max(x["runtime"]     for x in results)),
                    ni(r["memory_mb"], min(x["memory_mb"]   for x in results),max(x["memory_mb"]   for x in results)+.001),
                    ni(r["model_size_mb"],min(x["model_size_mb"] for x in results),max(x["model_size_mb"] for x in results)+.001)]
    else:
        dims = ["Accuracy","F1","Precision","Recall","Speed","Size Eff"]
        def sc(r):
            return [r.get("accuracy",0),r.get("f1",0),r.get("precision",0),r.get("recall",0),
                    ni(r["runtime"],      min(x["runtime"]      for x in results),max(x["runtime"]      for x in results)),
                    ni(r["model_size_mb"],min(x["model_size_mb"] for x in results),max(x["model_size_mb"] for x in results)+.001)]

    fig_rad = go.Figure()
    for i, r in enumerate(results):
        s2 = sc(r)
        fig_rad.add_trace(go.Scatterpolar(
            r=s2+[s2[0]], theta=dims+[dims[0]], fill="toself", name=r["name"],
            line=dict(color=clr(i),width=2), fillcolor=RFILL[i%len(RFILL)]))
    fig_rad.update_layout(
        polar=dict(bgcolor=_BG,
                   radialaxis=dict(visible=True,range=[0,1],gridcolor="#21262d",
                                   tickfont=dict(color="#7d8590",size=9),tickvals=[.25,.5,.75,1.]),
                   angularaxis=dict(gridcolor="#21262d",tickfont=dict(color="#c9d1d9",size=11))),
        height=480, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FT,
        legend=dict(font=dict(color="#c9d1d9"),bgcolor=_BG,bordercolor=_GRID,borderwidth=1),
        margin=dict(l=80,r=80,t=60,b=60))
    st.plotly_chart(fig_rad, use_container_width=True)

    st.markdown(sh("☁️","Cloud vs Local Simulation",""), unsafe_allow_html=True)
    cv1, cv2 = st.columns(2)
    cmodel = cv1.selectbox("Model", [r["name"] for r in results], key="cld_m")
    cspec  = cv2.selectbox("AWS Instance",
                            ["t3.micro","t3.medium","t3.large",
                             "c5.xlarge","c5.4xlarge","m5.4xlarge","p3.2xlarge"],
                            index=1)
    r_cld  = next(r for r in results if r["name"]==cmodel)
    cd2    = simulate_cloud_vs_local(r_cld, task, cspec)
    lo     = cd2["local"]; cl = cd2["cloud"]
    cc1,cc2,cc3,cc4 = st.columns(4)
    cc1.markdown(kpi("Local Runtime", f"{lo['runtime_s']:.4f}s",
                      f"{lo['vcpu']} vCPU/{lo['ram_gb']}GB"),              unsafe_allow_html=True)
    cc2.markdown(kpi("Cloud Runtime", f"{cl['runtime_s']:.4f}s",
                      cl["instance"], cls="g"),                             unsafe_allow_html=True)
    cc3.markdown(kpi("Speedup",       f"{cl['speedup']}x",
                      "cloud vs local", cls="y"),                           unsafe_allow_html=True)
    cc4.markdown(kpi("Cost/Run",      f"${cl['cost_cents']/100:.6f}",
                      "USD estimate", cls="p"),                             unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Compare":
    st.markdown("# 📐 Cross-Session Comparison")
    if len(st.session_state.sessions) < 2:
        st.info("Train at least 2 sessions to compare."); st.stop()
    opts = [f"{s['name']} | {s['dataset']} ({s['task']}) · {s['timestamp'][:10]}"
            for s in st.session_state.sessions]
    sel  = st.multiselect("Sessions", opts, default=opts[:min(4,len(opts))])
    if len(sel) < 2:
        st.warning("Select ≥2."); st.stop()
    sel_s = [st.session_state.sessions[opts.index(o)] for o in sel]
    rows  = []
    for s in sel_s:
        for r in s["model_results"]:
            row = {"Session":s["name"],"Task":s["task"],"Model":r["name"],
                   "Runtime(s)":r["runtime"],"Memory(MB)":r["memory_mb"]}
            if s["task"] == "Regression":
                row.update({"R2":r.get("r2",0),"RMSE":r.get("rmse",0),"MAE":r.get("mae",0)})
            else:
                row.update({"Accuracy":r.get("accuracy",0),"F1":r.get("f1",0),
                             "Precision":r.get("precision",0),"Recall":r.get("recall",0)})
            rows.append(row)
    df_cmp = pd.DataFrame(rows)
    df_cmp["Label"] = df_cmp["Session"] + " / " + df_cmp["Model"]
    for metric in [c for c in ["R2","Accuracy","Runtime(s)"] if c in df_cmp.columns]:
        st.markdown(sh("📊", f"{metric} Across Sessions",""), unsafe_allow_html=True)
        fig = px.bar(df_cmp[df_cmp[metric].notna()], x="Label", y=metric, color="Session",
                      text=df_cmp[df_cmp[metric].notna()][metric].apply(lambda v: f"{v:.4f}"),
                      color_discrete_sequence=PAL, title=metric)
        fig.update_traces(textposition="outside"); fig.update_layout(**PT(380))
        st.plotly_chart(fig, use_container_width=True)
    nc = [c for c in ["R2","Accuracy","F1","RMSE","MAE","Runtime(s)","Memory(MB)"]
           if c in df_cmp.columns]
    heat = df_cmp.set_index("Label")[nc].fillna(0)
    fig_heat = px.imshow(heat.T, text_auto=".3f", color_continuous_scale="Viridis",
                          aspect="auto", title="Sessions x Metrics Heatmap")
    fig_heat.update_layout(**{k:v for k,v in PT(max(300,len(heat)*30)).items()
                               if k not in ["xaxis","yaxis"]})
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(df_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — MODEL LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Model Library":
    st.markdown("# 📚 Model Library")
    st.markdown("Complete reference — how each model works, hyperparameter guide, "
                "original papers, and curated learning links.")
    tf = st.radio("Show models for:",
                   ["All","Regression only","Classification only"], horizontal=True)
    sq = st.text_input("Search", "", placeholder="e.g. forest, linear, neural, svm")
    keys = list(MODEL_KNOWLEDGE.keys())
    if tf == "Regression only":
        rk   = set(REGRESSION_MODELS.values()); keys = [k for k in keys if k in rk]
    elif tf == "Classification only":
        ck   = set(CLASSIFICATION_MODELS.values()); keys = [k for k in keys if k in ck]
    if sq:
        q    = sq.lower()
        keys = [k for k in keys
                if q in MODEL_KNOWLEDGE[k]["full"].lower()
                or q in MODEL_KNOWLEDGE[k]["type"].lower()
                or q in MODEL_KNOWLEDGE[k].get("short","").lower()]
    if not keys:
        st.warning("No models match."); st.stop()
    for key in keys:
        info = MODEL_KNOWLEDGE[key]
        with st.expander(f"**{info['full']}** — {info['type']} · "
                         f"{info.get('year','')} · {info.get('author','')}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{info['full']}**")
                st.markdown(f"*{info['type']} · {info.get('year','')} · {info.get('author','')}*")
                st.markdown(info.get("short",""))
                st.markdown("**How it works**")
                st.markdown(f"<p style='font-size:.84rem;color:#c9d1d9;line-height:1.6;'>"
                            f"{info.get('how_it_works','')}</p>", unsafe_allow_html=True)
                st.markdown("**✅ Strengths**")
                for s3 in info.get("strengths",[]): st.markdown(f"• {s3}")
                st.markdown("**⚠️ Weaknesses**")
                for w in info.get("weaknesses",[]): st.markdown(f"• {w}")
                st.markdown(f"**🎯 Best for:** {info.get('best_for','')}")
            with c2:
                hp = info.get("hyperparams",{})
                if hp:
                    st.markdown("**🎛️ Hyperparameters**")
                    for param, desc in hp.items():
                        st.markdown(f"`{param}` — {desc}")
                papers = info.get("papers",[])
                if papers:
                    st.markdown("**📄 Papers**")
                    for title, url in papers:
                        st.markdown(f"[{title}]({url})")
                learn = info.get("learn",[])
                if learn:
                    st.markdown("**📖 Learn More**")
                    links = "".join(
                        f"<a class='mk-link' href='{url}' target='_blank'>{title}</a>"
                        for title, url in learn)
                    st.markdown(links, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 History":
    st.markdown("# 📁 Session History")
    if not st.session_state.sessions:
        st.info("No sessions yet."); st.stop()
    for s in reversed(st.session_state.sessions):
        results = s["model_results"]
        task    = s["task"]
        mk      = "r2" if task=="Regression" else "accuracy"
        ml      = "R2" if task=="Regression" else "Acc"
        ns      = f"{s['n_samples']:,} rows"
        nf      = f"{s['n_features']} feats"
        tg      = f"Target: {s['target_col']}"
        nm      = f"{len(results)} models"
        with st.expander(f"📌 {s['name']}  ·  {s['dataset']}  ·  {s['timestamp'][:19]}"):
            aml_b = "&nbsp;"+badge("AutoML","bt") if s.get("automl_info") else ""
            st.markdown(
                f"{badge(task,'bg' if task=='Regression' else 'bb')}&nbsp;"
                f"{badge(ns,'bo')}&nbsp;{badge(nf,'bb')}&nbsp;"
                f"{badge(tg,'bp')}&nbsp;{badge(nm,'bt')}{aml_b}",
                unsafe_allow_html=True)
            st.markdown("")
            res_cols = st.columns(min(len(results),6))
            for col, r in zip(res_cols, results):
                col.markdown(kpi(r["name"], f"{r.get(mk,0):.4f}", ml),
                             unsafe_allow_html=True)
            if s.get("prep_log"):
                st.caption("Prep: " + " · ".join(s["prep_log"][:3]))
            st.markdown("")
            ca2, cb2 = st.columns([3,1])
            with ca2:
                if st.button("📊 Set as active", key=f"act_{s['id']}"):
                    st.session_state.active_session = s
                    st.success("Active session updated.")
            with cb2:
                if st.button("🗑️ Delete", key=f"del_{s['id']}"):
                    st.session_state.sessions = [x for x in st.session_state.sessions
                                                  if x["id"] != s["id"]]
                    delete_session(s["id"]); st.rerun()
