import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# =======================
# 1. LOAD & MERGE
# =======================
MTPLfreq = pd.read_csv("Dataset/freMTPL2freq.csv")
MTPLsev = pd.read_csv("Dataset/freMTPL2sev.csv")

# Aggregate sev data
MTPLsev_grp = MTPLsev.groupby('IDpol', as_index=False)['ClaimAmount'].sum()

# Merge (inner by default — avoids phantom rows)
df_merged = pd.merge(MTPLfreq, MTPLsev_grp, on='IDpol', how='inner')

# =======================
# 2. CATEGORICAL ENCODING
# =======================
categorical_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
label_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
onehot_cols = [c for c in categorical_cols if c not in label_cols]

# Label encode selected
le = LabelEncoder()
for col in label_cols:
    df_merged[col] = le.fit_transform(df_merged[col].astype(str))

# One-hot encode the rest
if onehot_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe.fit_transform(df_merged[onehot_cols]),
                          columns=ohe.get_feature_names_out(onehot_cols))
    df_merged = pd.concat([df_merged.drop(columns=onehot_cols), ohe_df], axis=1)

# =======================
# 3. SPLIT DATA
# =======================
df_merged = df_merged.dropna(subset=['ClaimAmount'])  # Drop rows with missing target
X = df_merged.drop(columns=['ClaimAmount'])
y = df_merged['ClaimAmount']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================
# 4. IMPUTE & SCALE
# =======================
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_valid_scaled = scaler.transform(imputer.transform(X_valid))

# =======================
# 5. FEATURE SELECTION
# =======================
lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train_scaled, y_train)
mask = lasso.coef_ != 0
X_train_sel = X_train_scaled[:, mask]
X_valid_sel = X_valid_scaled[:, mask]

# =======================
# 6. MODELS & GRIDSEARCH
# =======================
models = {
    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [50, 100, 150]}),
    "Poisson": (PoissonRegressor(max_iter=3000), {"alpha": [0.01, 0.1, 1]}),
    "Tweedie": (TweedieRegressor(max_iter=3000), {"alpha": [0.01, 0.1, 1]}),
    "XGB": (XGBRegressor(random_state=42, verbosity=0), {"n_estimators": [50, 100, 150]})
}

results = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train_sel, y_train)
    best_model = grid.best_estimator_
    mae = mean_absolute_error(y_valid, best_model.predict(X_valid_sel))
    results[name] = {"Best Params": grid.best_params_, "Validation MAE": mae}

# =======================
# 7. RESULTS
# =======================
results_df = pd.DataFrame(results).T.sort_values(by="Validation MAE")
print(results_df)
