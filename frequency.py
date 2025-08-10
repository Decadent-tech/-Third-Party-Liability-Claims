import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error

# =======================
# 1. Load Data
# =======================
MTPLfreq = pd.read_csv("Dataset/freMTPL2freq.csv")
MTPLsev = pd.read_csv("Dataset/freMTPL2sev.csv")

# Aggregate claim amounts by policy
MTPLsev_grp = MTPLsev.groupby(['IDpol'])[['ClaimAmount']].sum().reset_index()

# Merge
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
# 3. Preprocess
# =======================
categorical_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
label_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
onehot_cols = [col for col in categorical_cols if col not in label_cols]

le = LabelEncoder()
for col in label_cols:
    df_merged[col] = df_merged[col].astype(str)
    df_merged[col] = le.fit_transform(df_merged[col])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_ohe = pd.DataFrame(ohe.fit_transform(df_merged[onehot_cols]),
                           columns=ohe.get_feature_names_out(onehot_cols),
                           index=df_merged.index)
df_merged = pd.concat([df_merged.drop(columns=onehot_cols), encoded_ohe], axis=1)

# =======================
# 4. Targets
# =======================
y_freq = (df_merged['ClaimAmount'] > 0).astype(int)   # claim indicator
y_sev = np.where(y_freq > 0, df_merged['ClaimAmount'] / df_merged['ClaimNb'], 0)

X = df_merged.drop(columns=['ClaimAmount', 'ClaimNb', 'PolicyID'])
df_merged.reset_index(drop=True, inplace=True)

# =======================
# 5. Split (position-based)
# =======================
train_idx, val_idx = train_test_split(range(len(df_merged)), test_size=0.2, random_state=42)

X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
y_freq_train, y_freq_valid = y_freq.iloc[train_idx], y_freq.iloc[val_idx]
y_sev_train, y_sev_valid = y_sev[train_idx], y_sev[val_idx]

# =======================
# 6. Impute + Scale
# =======================
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_valid_scaled = scaler.transform(imputer.transform(X_valid))

# =======================
# 7. Feature Selection
# =======================
lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train_scaled, y_freq_train)
selected_mask = lasso.coef_ != 0

X_train_sel = X_train_scaled[:, selected_mask]
X_valid_sel = X_valid_scaled[:, selected_mask]

# =======================
# 8. Frequency Model
# =======================
freq_model = PoissonRegressor(max_iter=3000)
freq_model.fit(X_train_sel, y_freq_train)
freq_preds = freq_model.predict(X_valid_sel)

# =======================
# 9. Severity Model (train only on claim cases in TRAIN set)
# =======================
train_claim_mask = y_freq_train > 0
sev_model = TweedieRegressor(power=2, max_iter=3000)  # Gamma
sev_model.fit(X_train_sel[train_claim_mask], y_sev_train[train_claim_mask])

# Predict severity for validation set
sev_preds = sev_model.predict(X_valid_sel)

# =======================
# 10. Combine & Evaluate
# =======================
final_preds = freq_preds * sev_preds
mae_two_step = mean_absolute_error(df_merged['ClaimAmount'].iloc[val_idx], final_preds)

print(f"Two-step Frequency–Severity MAE: {mae_two_step:.4f}")
