import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import re
from sklearn import __version__ as sklearn_version
import warnings
warnings.filterwarnings('ignore')

# ── Exact training loop from onehot_train.py ───────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "Public_services_pressure.csv"
df = pd.read_csv(DATA_PATH)

target = "PRESSURE_SCORE_GAUSSIAN"
df = df.dropna(subset=[target])

df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], errors="coerce")
df["dow"]   = df["OCCUPANCY_DATE"].dt.dayofweek
df["month"] = df["OCCUPANCY_DATE"].dt.month
df["day"]   = df["OCCUPANCY_DATE"].dt.day

cat_features = ["LOCATION_POSTAL_CODE","SECTOR","OVERNIGHT_SERVICE_TYPE",
                "PROGRAM_MODEL","PROGRAM_AREA","CAPACITY_TYPE"]
num_features = ["ACTUAL_CAPACITY","lat","lon","dow","month","day"]

X = df[cat_features + num_features]
eps = 1e-6
y_raw = df[target].clip(eps, 1 - eps)
y = np.log(y_raw / (1 - y_raw))

df_sorted = df.sort_values("OCCUPANCY_DATE")
cut = int(len(df_sorted) * 0.8)
train_idx = df_sorted.index[:cut]
test_idx  = df_sorted.index[cut:]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

version_match = re.match(r"^(\d+)\.(\d+)", sklearn_version)
if version_match:
    sk_major, sk_minor = map(int, version_match.groups())
    ohe_kwargs = {"sparse_output": False} if (sk_major, sk_minor) >= (1, 2) else {"sparse": False}
else:
    ohe_kwargs = {"sparse": False}

preprocess = ColumnTransformer(transformers=[
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
    ]), cat_features),
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median"))
    ]), num_features),
])

model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_depth=8, max_iter=400, random_state=42)

pipe = Pipeline([("prep", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

pred_logit = pipe.predict(X_test)
pred = 1 / (1 + np.exp(-pred_logit))

y_test_raw  = y_raw.loc[test_idx].values
y_train_raw = y_raw.loc[train_idx].values
naive_pred  = np.full(len(y_test_raw), y_train_raw.mean())

baseline_mae = mean_absolute_error(y_test_raw, naive_pred)
mae  = mean_absolute_error(y_test_raw, pred)
rmse = mean_squared_error(y_test_raw, pred) ** 0.5
residuals = y_test_raw - pred

print(f"Baseline MAE : {baseline_mae:.6f}")
print(f"Model MAE    : {mae:.2e}")
print(f"Model RMSE   : {rmse:.2e}")

# ── Plots ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor='#0f1117')
fig.suptitle('HistGradientBoosting — PRESSURE_SCORE_GAUSSIAN',
             fontsize=16, color='white', fontweight='bold', y=0.98)

gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
pkw = dict(facecolor='#1a1d27')
tc  = 'white'
ac1 = '#4fc3f7'; ac2 = '#ef5350'; ac3 = '#66bb6a'

rng = np.random.default_rng(0)
sample = rng.choice(len(y_test_raw), min(4000, len(y_test_raw)), replace=False)

# 1. Predicted vs Actual
ax1 = fig.add_subplot(gs[0, 0], **pkw)
ax1.scatter(y_test_raw[sample], pred[sample], alpha=0.3, s=5, color=ac1)
lims = [y_test_raw.min(), y_test_raw.max()]
ax1.plot(lims, lims, '--', lw=1.5, color=ac2, label='Perfect fit')
ax1.set_xlabel('Actual', color=tc, fontsize=9)
ax1.set_ylabel('Predicted', color=tc, fontsize=9)
ax1.set_title('Predicted vs Actual', color=tc, fontsize=11, fontweight='bold')
ax1.tick_params(colors=tc, labelsize=8)
ax1.legend(fontsize=8, labelcolor=tc, facecolor='#2a2d3a')
for s in ax1.spines.values(): s.set_edgecolor('#444')

# 2. Residuals Distribution
ax2 = fig.add_subplot(gs[0, 1], **pkw)
ax2.hist(residuals, bins=80, color=ac1, edgecolor='none', alpha=0.85)
ax2.axvline(0, color=ac2, lw=1.5, linestyle='--')
ax2.set_xlabel('Residual', color=tc, fontsize=9)
ax2.set_ylabel('Count', color=tc, fontsize=9)
ax2.set_title('Residuals Distribution', color=tc, fontsize=11, fontweight='bold')
ax2.tick_params(colors=tc, labelsize=8)
for s in ax2.spines.values(): s.set_edgecolor('#444')

# 3. Permutation feature importances
# sklearn requires estimator to have fit(); wrap pipe with a no-op fit
ax3 = fig.add_subplot(gs[0, 2], **pkw)
sub = rng.choice(len(y_test_raw), min(3000, len(y_test_raw)), replace=False)
X_test_sub = X_test.iloc[sub]
y_test_sub = y_raw.loc[test_idx].values[sub]

class _PipeWrapper:
    """Thin wrapper so permutation_importance sees a valid estimator."""
    def __init__(self, pipeline):
        self._pipe = pipeline
    def fit(self, X, y):
        return self
    def predict(self, X):
        return 1 / (1 + np.exp(-self._pipe.predict(X)))

result = permutation_importance(
    _PipeWrapper(pipe),
    X_test_sub, y_test_sub,
    n_repeats=5, random_state=0,
    scoring='neg_mean_absolute_error'
)
imp_grouped = pd.Series(result.importances_mean,
                        index=cat_features + num_features).sort_values()
colors = [ac3 if v > imp_grouped.median() else ac1 for v in imp_grouped.values]
ax3.barh(imp_grouped.index, imp_grouped.values, color=colors, edgecolor='none')
ax3.set_xlabel('Permutation Importance\n(mean decrease in MAE)', color=tc, fontsize=8)
ax3.set_title('Feature Importances', color=tc, fontsize=11, fontweight='bold')
ax3.tick_params(colors=tc, labelsize=8)
for s in ax3.spines.values(): s.set_edgecolor('#444')

# 4. MAE vs iterations
ax4 = fig.add_subplot(gs[1, 0], **pkw)
iters = list(range(10, 401, 20))
train_maes, test_maes = [], []
for n_iter in iters:
    m = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=8,
                                      max_iter=n_iter, random_state=42)
    p2 = Pipeline([("prep", preprocess), ("model", m)])
    p2.fit(X_train, y_train)
    tr_pred = 1 / (1 + np.exp(-p2.predict(X_train)))
    te_pred = 1 / (1 + np.exp(-p2.predict(X_test)))
    train_maes.append(mean_absolute_error(y_train_raw, tr_pred))
    test_maes.append(mean_absolute_error(y_test_raw, te_pred))
ax4.plot(iters, train_maes, color=ac3, lw=2, label='Train MAE')
ax4.plot(iters, test_maes,  color=ac1, lw=2, label='Test MAE')
ax4.axhline(baseline_mae, color=ac2, lw=1.5, linestyle='--', label='Baseline MAE')
ax4.set_xlabel('n_estimators', color=tc, fontsize=9)
ax4.set_ylabel('MAE', color=tc, fontsize=9)
ax4.set_title('MAE vs Iterations', color=tc, fontsize=11, fontweight='bold')
ax4.tick_params(colors=tc, labelsize=8)
ax4.legend(fontsize=8, labelcolor=tc, facecolor='#2a2d3a')
for s in ax4.spines.values(): s.set_edgecolor('#444')

# 5. Residuals vs Predicted
ax5 = fig.add_subplot(gs[1, 1], **pkw)
ax5.scatter(pred[sample], residuals[sample], alpha=0.3, s=5, color=ac1)
ax5.axhline(0, color=ac2, lw=1.5, linestyle='--')
ax5.set_xlabel('Predicted', color=tc, fontsize=9)
ax5.set_ylabel('Residual', color=tc, fontsize=9)
ax5.set_title('Residuals vs Predicted', color=tc, fontsize=11, fontweight='bold')
ax5.tick_params(colors=tc, labelsize=8)
for s in ax5.spines.values(): s.set_edgecolor('#444')

# 6. Metrics Summary
ax6 = fig.add_subplot(gs[1, 2], **pkw)
ax6.axis('off')
metrics = [
    ('Baseline MAE',    f'{baseline_mae:.6f}'),
    ('Model MAE',       f'{mae:.2e}'),
    ('Model RMSE',      f'{rmse:.2e}'),
    ('MAE Improvement', f'{baseline_mae/mae:.0f}×'),
    ('N Test Samples',  f'{len(y_test_raw):,}'),
    ('Target Mean',     f'{y_test_raw.mean():.4f}'),
    ('Target Std',      f'{y_test_raw.std():.4f}'),
]
yp = 0.92
for k, v in metrics:
    col = ac3 if k == 'MAE Improvement' else tc
    ax6.text(0.05, yp, k, transform=ax6.transAxes, fontsize=10, color='#aaa', va='top')
    ax6.text(0.95, yp, v, transform=ax6.transAxes, fontsize=10, color=col,
             va='top', ha='right', fontweight='bold')
    yp -= 0.13
ax6.set_title('Model Summary', color=tc, fontsize=11, fontweight='bold')

out = Path(__file__).resolve().parent / "regression_tree_graphs.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved to {out}")
