import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
from sklearn import __version__ as sklearn_version

df = pd.read_csv("Public_services_pressure.csv")

# Target
target = "PRESSURE_SCORE_GAUSSIAN"
df = df.dropna(subset=[target])

# Date features
df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], errors="coerce")
df["dow"] = df["OCCUPANCY_DATE"].dt.dayofweek
df["month"] = df["OCCUPANCY_DATE"].dt.month
df["day"] = df["OCCUPANCY_DATE"].dt.day

# Categorical and numeric features (NO occupancy rate / occupied capacity)
cat_features = [
    "LOCATION_POSTAL_CODE",
    "SECTOR",
    "OVERNIGHT_SERVICE_TYPE",
    "PROGRAM_MODEL",
    "PROGRAM_AREA",
    "CAPACITY_TYPE",
]
num_features = [
    "ACTUAL_CAPACITY",
    "lat", "lon",
    "dow", "month", "day",
]

X = df[cat_features + num_features]
# Logit-transform the target (clip to avoid infinities at 0/1)
eps = 1e-6
y_raw = df[target].clip(eps, 1 - eps)
y = np.log(y_raw / (1 - y_raw))

# Time-aware split (train on earlier, test on later)
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

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
        ]), cat_features),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), num_features),
    ]
)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=400,
    random_state=42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

pipe.fit(X_train, y_train)
pred_logit = pipe.predict(X_test)
pred = 1 / (1 + np.exp(-pred_logit))

mae = mean_absolute_error(y_raw.loc[test_idx], pred)
rmse = mean_squared_error(y_raw.loc[test_idx], pred) ** 0.5
# Check if a trivial baseline is also "good"
naive_pred = np.full(len(y_test), y_raw.loc[train_idx].mean())
print("Baseline MAE:", mean_absolute_error(y_raw.loc[test_idx], naive_pred))

# Check target variance
print(y_raw.describe())

# Check feature correlations
print(df[num_features + [target]].corr()[target].sort_values())

print("MAE:", mae)
print("RMSE:", rmse)


