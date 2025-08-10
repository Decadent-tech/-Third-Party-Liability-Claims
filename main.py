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
# 0. Load & merge
# -------------------------
MTPLfreq = pd.read_csv("Dataset/freMTPL2freq.csv")
MTPLsev  = pd.read_csv("Dataset/freMTPL2sev.csv")

MTPLsev_grp = MTPLsev.groupby(['IDpol'])[['ClaimAmount']].sum().reset_index()
df_merged = pd.merge(MTPLfreq, MTPLsev_grp, how='left', on='IDpol').fillna(0)

# -------------------------
# 1. Create frequency & severity (ensure float dtype)
# -------------------------
df_merged["ClaimFrequency"] = df_merged["ClaimNb"] / df_merged["Exposure"]
df_merged["AvgSeverity"] = np.where(
    df_merged["ClaimNb"] > 0,
    df_merged["ClaimAmount"] / df_merged["ClaimNb"],
    0.0
)
# create a float 'Severity' column explicitly to avoid dtype warnings
df_merged['Severity'] = 0.0
mask_claims = df_merged['ClaimNb'] > 0
df_merged.loc[mask_claims, 'Severity'] = (
    (df_merged.loc[mask_claims, 'ClaimAmount'] / df_merged.loc[mask_claims, 'ClaimNb']).astype(float)
)

df_merged["PolicyID"] = df_merged["IDpol"]

# -------------------------
# 2. Record raw columns BEFORE encoding (for prediction construction)
# -------------------------
raw_columns_before_encoding = df_merged.columns.tolist()  # we'll use this for default values later

# -------------------------
# 3. Encoding (fit and store encoders)
# -------------------------
categorical_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
label_cols = [c for c in ['Area', 'VehBrand', 'VehGas', 'Region'] if c in df_merged.columns]
onehot_cols = [c for c in categorical_cols if c not in label_cols]

label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_merged[col] = df_merged[col].astype(str)
    df_merged[col] = le.fit_transform(df_merged[col])
    label_encoders[col] = le

# Fit OHE on any remaining object columns (onehot_cols may be empty)
if onehot_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe.fit_transform(df_merged[onehot_cols]),
                          columns=ohe.get_feature_names_out(onehot_cols),
                          index=df_merged.index)
    df_merged = pd.concat([df_merged.drop(columns=onehot_cols), ohe_df], axis=1)
else:
    ohe = None  # handle later in predict

# -------------------------
# 4. Prepare training feature matrix (features used in model)
# -------------------------
drop_cols = ['ClaimAmount', 'Severity', 'ClaimNb', 'IDpol']
features = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns], errors='ignore')
# Save feature column names for later validation
trained_feature_columns = features.columns.tolist()

# -------------------------
# 5. Train/validation split
# -------------------------
X_train, X_valid, y_freq_train, y_freq_valid = train_test_split(
    features, df_merged['ClaimNb'], test_size=0.2, random_state=42
)
_, _, y_sev_train, y_sev_valid = train_test_split(
    features, df_merged['Severity'], test_size=0.2, random_state=42
)

# -------------------------
# 6. Impute -> Scale -> Lasso selection
# -------------------------
imputer = SimpleImputer(strategy='mean')
scaler  = MinMaxScaler()

X_train_imp = imputer.fit_transform(X_train)
X_valid_imp = imputer.transform(X_valid)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_valid_scaled = scaler.transform(X_valid_imp)

lasso = Lasso(alpha=0.001, random_state=42, max_iter=200000)
lasso.fit(X_train_scaled, y_freq_train)
selected_mask = lasso.coef_ != 0
# fall back to all features if none selected
if selected_mask.sum() == 0:
    selected_mask = np.ones(X_train_scaled.shape[1], dtype=bool)

X_train_sel = X_train_scaled[:, selected_mask]
X_valid_sel = X_valid_scaled[:, selected_mask]

# -------------------------
# 7. Prepare severity training subset (positive claims within train)
# -------------------------
train_claim_mask = (y_freq_train > 0).to_numpy()
if train_claim_mask.sum() == 0:
    raise ValueError("No positive claims in training set - cannot train severity.")

X_train_sel_pos = X_train_sel[train_claim_mask]
y_sev_train_pos = y_sev_train.to_numpy()[train_claim_mask]

# -------------------------
# 8. Model grids (tune)
# -------------------------
freq_models = {
    "Poisson": (PoissonRegressor(max_iter=3000), {"alpha": [0.01, 0.1, 1]}),
    "XGB": (XGBRegressor(objective='reg:squarederror', random_state=42),
            {"max_depth": [3, 6], "n_estimators": [50, 100]}),
    "RandomForest": (RandomForestRegressor(random_state=42),
                     {"max_depth": [6, 12], "n_estimators": [100, 150]})
}

sev_models = {
    "Tweedie_Gamma": (TweedieRegressor(power=1.5, max_iter=3000),
                      {"alpha": [0.1, 1], "power": [1.5]}),
    "XGB": (XGBRegressor(objective='reg:squarederror', random_state=42),
            {"max_depth": [3], "n_estimators": [50]}),
    "RandomForest": (RandomForestRegressor(random_state=42),
                     {"max_depth": [6], "n_estimators": [100]})
}

# -------------------------
# 9. Train & evaluate combos
# -------------------------
results = []
best_mae = float('inf')
best_freq_model = None
best_sev_model = None

for freq_name, (freq_estimator, freq_params) in freq_models.items():
    freq_gs = GridSearchCV(freq_estimator, freq_params, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
    freq_gs.fit(X_train_sel, y_freq_train)
    freq_best = freq_gs.best_estimator_
    freq_preds = freq_best.predict(X_valid_sel)

    for sev_name, (sev_estimator, sev_params) in sev_models.items():
        sev_gs = GridSearchCV(sev_estimator, sev_params, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
        sev_gs.fit(X_train_sel_pos, y_sev_train_pos)
        sev_best = sev_gs.best_estimator_
        sev_preds = sev_best.predict(X_valid_sel)

        final_preds = freq_preds * sev_preds
        mae = mean_absolute_error(df_merged['ClaimAmount'].iloc[X_valid.index], final_preds)

        results.append({
            "freq_model": freq_name,
            "freq_params": freq_gs.best_params_,
            "sev_model": sev_name,
            "sev_params": sev_gs.best_params_,
            "MAE": mae
        })

        if mae < best_mae:
            best_mae = mae
            best_freq_model = freq_best
            best_sev_model = sev_best

# -------------------------
# 10. Results
# -------------------------
df_results = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
print("\n=== Summary (top 10) ===")
print(df_results.head(10))
print(f"\nBest Combo -> Freq: {df_results.iloc[0]['freq_model']} {df_results.iloc[0]['freq_params']} | "
      f"Sev: {df_results.iloc[0]['sev_model']} {df_results.iloc[0]['sev_params']} | MAE: {best_mae:.4f}")

# -------------------------
# 11. Helper: safe label transform (vectorized)
# -------------------------


def safe_label_transform(encoder, values):
    # Ensure numpy array of strings
    values = np.array(values, dtype=str)

    # Replace unseen labels with first known class
    mask_known = np.isin(values, encoder.classes_)
    values[~mask_known] = encoder.classes_[0]

    return encoder.transform(values)

# -------------------------
# 12. Prediction function (API-ready)
# -------------------------
def predict_claim_cost(raw_policy_df):
    """
    raw_policy_df: DataFrame with raw columns similar to MTPLfreq rows.
                   Missing columns will be filled with sensible defaults.
    """
    raw = raw_policy_df.copy().reset_index(drop=True)

    # 1) Ensure raw contains necessary pre-encoding columns (label_cols, onehot_cols, numeric columns like Exposure etc.)
    # We'll construct a "base" DataFrame with all expected raw columns (derived from df_merged before encoding)
    base_cols = set()
    # columns that we encoded/used originally before dropping in features:
    # start from raw_columns_before_encoding (captured earlier)
    for c in raw_columns_before_encoding:
        if c not in ['ClaimAmount', 'Severity', 'ClaimNb', 'IDpol']:
            base_cols.add(c)
    base_cols = list(base_cols)

    # create template filled with defaults (0 or empty string)
    template = pd.DataFrame([{c: "" if c in label_cols + onehot_cols else 0 for c in base_cols}])
    # overwrite with provided values
    for c in raw.columns:
        if c in template.columns:
            template.loc[0, c] = raw.loc[0, c]

    # compute engineered features exactly as in training
    # ClaimFrequency needs ClaimNb and Exposure present; if not present, assume 0
    try:
        claimnb = float(template.get('ClaimNb', 0) or 0)
    except Exception:
        claimnb = 0.0
    try:
        exposure = float(template.get('Exposure', 0) or 0)
    except Exception:
        exposure = 0.0
    template['ClaimFrequency'] = claimnb / exposure if exposure > 0 else 0.0

    # AvgSeverity / Severity: if ClaimNb >0
    template['AvgSeverity'] = (template['ClaimAmount'] / template['ClaimNb']) if (template.get('ClaimNb', 0) and template['ClaimNb'] > 0) else 0.0
    template['Severity'] = template['AvgSeverity']  # keep naming consistent if used elsewhere
    template['PolicyID'] = template.get('IDpol', 0)

    # 2) Label-encode safely
    for col in label_cols:
        if col not in template.columns:
            template[col] = ""
        template[col] = template[col].astype(str)
        template[col] = safe_label_transform(label_encoders[col], template[col])

    # 3) One-hot encode using fitted OHE (if present)
    if ohe is not None and onehot_cols:
        ohe_input = template[onehot_cols].astype(str)
        ohe_df = pd.DataFrame(ohe.transform(ohe_input),
                              columns=ohe.get_feature_names_out(onehot_cols))
        # drop original onehot_cols and concat encoded
        template = pd.concat([template.drop(columns=onehot_cols, errors='ignore'), ohe_df], axis=1)

    # 4) Keep only the feature columns in the same order as training features
    # Missing columns are filled with 0
    for col in trained_feature_columns:
        if col not in template.columns:
            template[col] = 0

    template = template[trained_feature_columns]  # enforce order

    # 5) Impute -> scale -> select features
    sample_imp = imputer.transform(template)
    sample_scaled = scaler.transform(sample_imp)
    sample_sel = sample_scaled[:, selected_mask]

    # 6) Predict
    freq_pred = best_freq_model.predict(sample_sel)
    sev_pred = best_sev_model.predict(sample_sel)
    return freq_pred * sev_pred

# -------------------------
# 13. Quick test with an actual row from your dataset (guaranteed to be valid)
# -------------------------
test_row = MTPLfreq.iloc[[0]].copy()  # raw row from original frequency file
predicted_cost = predict_claim_cost(test_row)
print(f"\nTest Row Prediction (from MTPLfreq row 0): {predicted_cost[0]:.2f}")

# -------------------------
# You can also test with a custom raw dict; e.g.
#raw = pd.DataFrame([{'IDpol':999999,'ClaimNb':0,'ClaimAmount':0,'Exposure':1,'Area':'A','VehBrand':'B1','VehGas':'Diesel','Region':'R1', 'VehPower':'7','VehType':'Small','VehColor':'Red','VehAge':5,'DrivAge':45,'BonusMalus':50,'Density':350}])
#print(predict_claim_cost(raw))
# -------------------------
