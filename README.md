# CropML Research Studio v5.0

Research-grade ML platform — Regression & Classification.
AutoML, Multi-Dataset analysis, 23 models, PDF reports, cloud simulation.

---

## HOW TO RUN (3 commands)

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Run the app
streamlit run app.py

# 3 — Open browser
# Navigate to: http://localhost:8501
```

If `streamlit` is not found, use:
```bash
python -m streamlit run app.py
```

---

## Project Structure

```
cropml/
├── app.py                    ← Entry point — run this file
├── requirements.txt          ← All Python dependencies
├── README.md
├── backend/
│   ├── __init__.py
│   ├── models.py             ← 23 models + recommender + trainer + cloud sim
│   ├── automl.py             ← AutoML: benchmark → Optuna tune → fit-quality
│   ├── multi_dataset.py      ← Multi-CSV merge, correlation, enrichment
│   ├── preprocessor.py       ← DataProfiler + PreprocessingPipeline
│   ├── report.py             ← PDF report generator
│   └── session.py            ← Session save/load/delete
├── frontend/
│   ├── __init__.py
│   └── hyperparams.py        ← Per-model hyperparameter slider widgets
└── saved_sessions/           ← Auto-created; one JSON per training session
```

---

## Pages

| Page | Description |
|---|---|
| 🔬 Train | Upload CSV → preprocess → smart model recommendations → tune → train |
| 🤖 AutoML Focus | Benchmark all → Optuna-tune the best → over/underfitting diagnosis |
| 🗂️ Multi-Dataset | Upload N CSVs → merge → cross-correlation → enrichment |
| 📊 Analysis | Full charts, metrics table, residuals, radar, cloud vs local |
| 📐 Compare | Cross-session bar charts + heatmap |
| 📚 Model Library | How each model works, papers, hyperparameter guide, learning links |
| 📁 History | Browse, activate, delete saved sessions |

---

## Models

### Regression (13)
Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost,
Decision Tree, Ridge, Lasso, ElasticNet, Linear Regression,
SVR (SVM), K-Nearest Neighbors, MLP Neural Network

### Classification (10)
Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost,
Decision Tree, Logistic Regression, SVC (SVM),
K-Nearest Neighbors, MLP Neural Network

---

## Model Performance Optimisation Code

The AutoML Focus Engine uses the following optimisation approach:

```python
# 1. Fast benchmark — cross-validate all models with default params
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="r2")

# 2. Optuna TPE search — focus on best model
import optuna
study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=60, timeout=120)

# 3. Fit quality check — train/test gap diagnosis
gap = train_score - test_score
# gap > 0.20  → overfitting  → increase regularisation, reduce depth
# gap > 0.10  → mild overfit → slight regularisation increase
# score < baseline + 0.02 → underfitting → more estimators, complex model
# else        → well fitted
```

Key hyperparameters searched per model:
- **Random Forest / Extra Trees**: n_estimators (100–800), max_depth, min_samples_split, min_samples_leaf, max_features
- **XGBoost**: n_estimators, learning_rate (log scale), max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, subsample
- **SVR/SVC**: C (log scale), kernel, epsilon
- **KNN**: n_neighbors (1–30), weights, metric
- **MLP**: hidden layer architecture (1–3 layers, 32–256 units each), activation, learning rate

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| `kpi() unexpected keyword cls` | Fixed in this version — `cls` is now a proper kwarg |
| PDF export blank charts | Install kaleido: `pip install kaleido` |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Optuna too slow | Reduce Trials slider or increase Timeout on AutoML page |
