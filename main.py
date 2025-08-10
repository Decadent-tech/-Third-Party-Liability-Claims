import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# =======================
# 1. Load & Merge Data
# =======================
MTPLfreq = pd.read_csv("Dataset/freMTPL2freq.csv")
MTPLsev = pd.read_csv("Dataset/freMTPL2sev.csv")

MTPLsev_grp = MTPLsev.groupby(['IDpol'])[['ClaimAmount']].agg('sum').reset_index()
df_merged = pd.merge(MTPLfreq, MTPLsev_grp, how='left', on='IDpol').fillna(0)

# =======================
# 2. Create Frequency & Severity
# =======================
df_merged["ClaimFrequency"] = df_merged["ClaimNb"] / df_merged["Exposure"]
df_merged["AvgSeverity"] = np.where(
    df_merged["ClaimNb"] > 0,
    df_merged["ClaimAmount"] / df_merged["ClaimNb"],
    0
)
df_merged["PolicyID"] = df_merged["IDpol"]

# =======================
# 3. Encoding
# =======================
categorical_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
label_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
onehot_cols = [col for col in categorical_cols if col not in label_cols]

le = LabelEncoder()
for col in label_cols:
    df_merged[col] = df_merged[col].astype(str)
    df_merged[col] = le.fit_transform(df_merged[col])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_ohe = pd.DataFrame(ohe.fit_transform(df_merged[onehot_cols]))
encoded_ohe.columns = ohe.get_feature_names_out(onehot_cols)
df_merged = pd.concat([df_merged.drop(columns=onehot_cols), encoded_ohe], axis=1)

# =======================
# 4. Targets
# =======================
df_merged['Severity'] = 0
mask_claims = df_merged['ClaimNb'] > 0
df_merged.loc[mask_claims, 'Severity'] = (
    df_merged.loc[mask_claims, 'ClaimAmount'] / df_merged.loc[mask_claims, 'ClaimNb']
)

features = df_merged.drop(columns=['ClaimAmount', 'Severity', 'ClaimNb', 'IDpol'])
X_train, X_valid, y_freq_train, y_freq_valid = train_test_split(
    features, df_merged['ClaimNb'], test_size=0.2, random_state=42
)
_, _, y_sev_train, y_sev_valid = train_test_split(
    features, df_merged['Severity'], test_size=0.2, random_state=42
)

# =======================
# 5. Impute + Scale + Lasso
# =======================
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()

X_train_imp = imputer.fit_transform(X_train)
X_valid_imp = imputer.transform(X_valid)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_valid_scaled = scaler.transform(X_valid_imp)

lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train_scaled, y_freq_train)
selected_mask = lasso.coef_ != 0

X_train_sel = X_train_scaled[:, selected_mask]
X_valid_sel = X_valid_scaled[:, selected_mask]

# =======================
# 6. Models + Param Grids
# =======================
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

# =======================
# 7. Train & Evaluate Combinations
# =======================
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
        sev_gs.fit(X_train_sel[mask_claims.loc[X_train.index]], y_sev_train[mask_claims.loc[X_train.index]])
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

# =======================
# 8. Output Results
# =======================
df_results = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
print("\n=== Summary (top 10) ===")
print(df_results.head(10))
print(f"\nBest Combo -> Freq: {df_results.iloc[0]['freq_model']} {df_results.iloc[0]['freq_params']} | "
      f"Sev: {df_results.iloc[0]['sev_model']} {df_results.iloc[0]['sev_params']} | MAE: {best_mae:.4f}")

# =======================
# 9. Prediction Function
# =======================
def predict_claim_cost(sample_df):
    # preprocess same way
    for col in label_cols:
        sample_df[col] = sample_df[col].astype(str)
        sample_df[col] = le.transform(sample_df[col])
    encoded_ohe_sample = pd.DataFrame(ohe.transform(sample_df[onehot_cols]))
    encoded_ohe_sample.columns = ohe.get_feature_names_out(onehot_cols)
    sample_df = pd.concat([sample_df.drop(columns=onehot_cols), encoded_ohe_sample], axis=1)

    sample_imp = imputer.transform(sample_df)
    sample_scaled = scaler.transform(sample_imp)
    sample_sel = sample_scaled[:, selected_mask]

    freq_pred = best_freq_model.predict(sample_sel)
    sev_pred = best_sev_model.predict(sample_sel)
    return freq_pred * sev_pred


# Example policy (before encoding)
sample_df = pd.DataFrame([{
    "Area": "C","VehBrand": "B12","VehGas": "Regular","Region": "R24","Exposure": 0.75,"DrivAge": 45,"BonusMalus": 50,"VehAge": 5,
    "Density": 350.5,"ClaimNb": 0,"ClaimAmount": 0,"IDpol": 999999,"VehPower": "7","VehType": "Small","VehColor": "Red"
                        }])
#IDpol,ClaimNb,Exposure,Area,VehPower,VehAge,DrivAge,BonusMalus,VehBrand,VehGas,Density,Region
# Predict claim cost
predicted_cost = predict_claim_cost(sample_df)
print(f"Predicted Claim Cost: {predicted_cost[0]:.2f}")
