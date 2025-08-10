# Two-Step Frequency–Severity Modeling for Third-Party Liability Claims

This project implements a **two-step frequency–severity modeling** approach to predict insurance claim costs for Motor Third-Party Liability (MTPL) claims.  
It uses a combination of machine learning models for **claim frequency** and **claim severity**, selected via hyperparameter tuning to minimize Mean Absolute Error (MAE).  
The best-performing combination is then used for live predictions, with automatic handling of unseen categorical values.



## Features

- **Two-step modeling**:
  - **Frequency model**: Predicts expected number of claims.
  - **Severity model**: Predicts expected claim cost, given a claim occurred.
- **Model selection**:
  - Tests multiple algorithms (`RandomForest`, `XGB`, `Poisson`, `Tweedie_Gamma`, etc.)
  - Hyperparameter tuning for each model.
- **Automatic categorical encoding**:
  - Uses `LabelEncoder` for training categories.
  - Replaces unseen categories during prediction to avoid runtime errors.
- **Sample prediction**:
  - Supports direct testing with any sample row from the dataset.
  - Returns estimated claim cost.



## Installation

1. **Clone the repository**
   
   git clone https://github.com/yourusername/Third-Party-Liability-Claims.git
   cd Third-Party-Liability-Claims
2. **Install dependencies**

    pip install -r requirements.txt
3. **Prepare dataset**
    * Place your MTPL frequency and severity datasets in the data/ folder.
    * Ensure column names match those expected in the script.

4. **Usage**
    Train Models
    python main.py
    
    This will:

    * Train frequency & severity models with multiple algorithms.
    * Select the best combination based on MAE.
    * Save trained models and encoders for prediction.

5. Make a Prediction
    Inside main.py, after training:

    raw_policy = {
        "Region": "R1",
        "VehicleType": "C",
        "DriverAge": 45,
        "PolicyDuration": 12,
        # Add other required features
    }

    predicted_cost = predict_claim_cost(raw_policy)
    print(f"Predicted Claim Cost: {predicted_cost:.2f}")
    Unseen categories (like a new Region value) will be replaced with a default known value to avoid errors.

6. **Example Output**

=== Summary (top 10) ===
     freq_model                             freq_params      sev_model                             sev_params         MAE
0   RandomForest  {'max_depth': 12, 'n_estimators': 100}   RandomForest  {'max_depth': 6, 'n_estimators': 100}  160.999275

    Best Combo -> Freq: RandomForest {'max_depth': 12, 'n_estimators': 100} | Sev: RandomForest {'max_depth': 6, 'n_estimators': 100} | MAE: 160.9993

    Test Row Prediction (from MTPLfreq row 0): 12.05
7. **API Integration**
    The predict_claim_cost() function is ready for live API usage:
    
    * Takes a dictionary of feature values.
    * Encodes categorical variables with safe handling for unseen values.
    * Returns predicted claim cost in numeric form.

8. **Requirements**
    Python 3

    pandas

    numpy

    scikit-learn

    xgboost
Install with:

    pip install pandas numpy scikit-learn xgboost
9. **License**
    This project is licensed under the MIT License.