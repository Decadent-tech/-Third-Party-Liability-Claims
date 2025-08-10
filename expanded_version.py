import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 0. Settings
# -------------------------
RANDOM_STATE = 42
CV = 3            # keep CV small for faster runs; increase if you want more robust tuning
N_JOBS = -1

# -------------------------
# 1. Load & merge data
# -------------------------
MTPLfreq = pd.read_csv("Dataset/freMTPL2freq.csv")
MTPLsev  = pd.read_csv("Dataset/freMTPL2sev.csv")

MTPLsev_grp = MTPLsev.groupby(['IDpol'])[['ClaimAmount']].sum().reset_index()
df = pd.merge(MTPLfreq, MTPLsev_grp, how='left', on='IDpol').fillna(0)

# Keep original ID as a column for traceability
df = df.reset_index(drop=True)
df['PolicyID'] = df['IDpol']

# -------------------------
# 2. Create frequency & severity targets
# -------------------------
# y_freq = count of claims per policy (ClaimNb exists in freMTPL2freq)
if 'ClaimNb' in df.columns:
    y_freq = df['ClaimNb'].astype(int)
else:
    # fallback: indicator (0/1) if ClaimNb absent
    y_freq = (df['ClaimAmount'] > 0).astype(int)

# severity = average claim size (only for policies with >0 claims)
y_sev = np.where(df['ClaimNb'] > 0, df['ClaimAmount'] / df['ClaimNb'], 0)

# Ground truth for final evaluation
y_total = df['ClaimAmount']

# -------------------------
# 3. Preprocess categorical cols (simple encoding)
# -------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# choose some columns to label encode explicitly if present
label_cols = [c for c in ['Area', 'VehBrand', 'VehGas', 'Region'] if c in df.columns]
onehot_cols = [c for c in categorical_cols if c not in label_cols and c != 'PolicyID']

# Label encode chosen columns
le = LabelEncoder()
for col in label_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

# One-hot encode the rest if any
if onehot_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(
        ohe.fit_transform(df[onehot_cols]),
        columns=ohe.get_feature_names_out(onehot_cols),
        index=df.index
    )
    df = pd.concat([df.drop(columns=onehot_cols), ohe_df], axis=1)

# -------------------------
# 4. Create feature matrix X
# -------------------------
drop_cols = ['ClaimAmount', 'ClaimNb', 'PolicyID', 'IDpol']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# -------------------------
# 5. Position-based split (prevents 'not in index' issues)
# -------------------------
n = len(df)
all_idx = np.arange(n)
train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_STATE)

X_train_df = X.iloc[train_idx].copy()
X_val_df   = X.iloc[val_idx].copy()

y_freq_train = y_freq.iloc[train_idx].reset_index(drop=True)
y_freq_val   = y_freq.iloc[val_idx].reset_index(drop=True)
y_sev_all    = pd.Series(y_sev).reset_index(drop=True)   # will index by position
y_total_val  = y_total.iloc[val_idx].reset_index(drop=True)

# -------------------------
# 6. Impute & scale (fit only on training)
# -------------------------
imputer = SimpleImputer(strategy='mean')
scaler  = MinMaxScaler()

X_train_imp = imputer.fit_transform(X_train_df)
X_val_imp   = imputer.transform(X_val_df)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)

# -------------------------
# 7. Lasso feature selection (fit on frequency target in train)
# -------------------------
lasso = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=100000)
lasso.fit(X_train_scaled, y_freq_train)
selected_mask = lasso.coef_ != 0

# If Lasso dropped everything (rare), fall back to using all features
if selected_mask.sum() == 0:
    selected_mask = np.ones(X_train_scaled.shape[1], dtype=bool)

X_train_sel = X_train_scaled[:, selected_mask]
X_val_sel   = X_val_scaled[:, selected_mask]

# For severity training: only positive claim rows inside TRAIN
train_pos_mask = (y_freq_train > 0).to_numpy()  # boolean mask aligned with X_train_sel rows
if train_pos_mask.sum() == 0:
    raise ValueError("No positive-claim rows in the training set — cannot train severity model.")

X_train_sel_pos = X_train_sel[train_pos_mask]
y_sev_train_pos = y_sev_all.iloc[train_idx].to_numpy()[train_pos_mask]

# -------------------------
# 8. Define models + param grids
# -------------------------
freq_models = {
    "Poisson": (PoissonRegressor(max_iter=3000),
                {"alpha": [0.01, 0.1, 1]}),
    "XGB": (XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {"n_estimators": [50, 100], "max_depth": [3, 6]}),
    "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE),
                     {"n_estimators": [100, 150], "max_depth": [6, 12]})
}

sev_models = {
    "Tweedie_Gamma": (TweedieRegressor(power=2, max_iter=3000),
                      {"alpha": [0.01, 0.1, 1], "power": [1.5, 2.0]}),  # allow power to vary
    "XGB": (XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {"n_estimators": [50, 100], "max_depth": [3, 6]}),
    "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE),
                     {"n_estimators": [100, 150], "max_depth": [6, 12]})
}

# -------------------------
# 9. Tuning + training loop
# -------------------------
results = []

for f_name, (f_model, f_grid) in freq_models.items():
    print(f"\nTuning Frequency model: {f_name}")
    gs_f = GridSearchCV(f_model, f_grid, cv=CV, scoring='neg_mean_absolute_error', n_jobs=N_JOBS)
    gs_f.fit(X_train_sel, y_freq_train)                    # fit on selected features
    best_f = gs_f.best_estimator_
    print(f" -> Best freq params: {gs_f.best_params_}")

    # frequency predictions on validation
    freq_pred_val = best_f.predict(X_val_sel)

    for s_name, (s_model, s_grid) in sev_models.items():
        print(f"   Tuning Severity model: {s_name}")
        # GridSearch for severity must be run only on positive-claim training subset
        gs_s = GridSearchCV(s_model, s_grid, cv=CV, scoring='neg_mean_absolute_error', n_jobs=N_JOBS)
        gs_s.fit(X_train_sel_pos, y_sev_train_pos)
        best_s = gs_s.best_estimator_
        print(f"    -> Best sev params: {gs_s.best_params_}")

        # severity predictions for validation (we predict for all val rows; for zero-claim policies freq_pred will be ~0)
        sev_pred_val = best_s.predict(X_val_sel)

        # final cost prediction = freq * severity
        final_pred_val = freq_pred_val * sev_pred_val

        # compute MAE against true total claim amount for validation rows
        mae = mean_absolute_error(y_total_val.to_numpy(), final_pred_val)
        print(f"    Combined MAE (freq={f_name}, sev={s_name}): {mae:.4f}")

        results.append({
            "freq_model": f_name,
            "freq_params": gs_f.best_params_,
            "sev_model": s_name,
            "sev_params": gs_s.best_params_,
            "MAE": mae
        })

# -------------------------
# 10. Results summary
# -------------------------
results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
print("\n=== Summary (top 10) ===")
print(results_df.head(10))
# Save results to CSV if you want
results_df.to_csv("two_step_model_comparison_results.csv", index=False)
print("\nSaved results to two_step_model_comparison_results.csv")
