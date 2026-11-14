import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

TYPE_MAPPING = {'A': 1, 'B': 2, 'C': 3}

def load_and_preprocess_data(file_path='Data_Set.csv'):
    # Load dataset with fallback for Windows/Linux
    if not os.path.exists(file_path):
        alt_path = 'D:/Data Analyst/Internship_Data/Walmart Sales/Data_Set.csv'
        if os.path.exists(alt_path):
            file_path = alt_path

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Cleaning: Drop rows with missing target or department
    df = df.dropna(subset=['Dept', 'Weekly_Sales']).copy()
    df['Dept'] = df['Dept'].astype(int)

    # 2. Date Parsing & Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # Store Type Encoding
    df['Type_Code'] = df['Type'].map(TYPE_MAPPING).fillna(0).astype(int)

    # MarkDowns: In Walmart data, markdowns are 0 when not active
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        df[f'{col}_Active'] = (~df[col].isna()).astype(int)
        df[col] = df[col].fillna(0)

    # Economic Indicators: CPI & Unemployment median imputation
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    return df

def train_and_evaluate_models():
    df = load_and_preprocess_data()

    # 3. Chronological Train-Test Split (Strictly Time-Series to prevent leakage)
    df = df.sort_values('Date').reset_index(drop=True)
    split_ratio = 0.80
    split_idx = int(len(df) * split_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    split_date = test_df['Date'].min()
    print(f"Data Split: Train records: {len(train_df):,} | Test records: {len(test_df):,}")
    print(f"Chronological Split Date: {split_date.strftime('%Y-%m-%d')}")

    # 4. Target Encoding / Historical Mean Sales (Fit strictly on Train set!)
    store_dept_means = train_df.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().rename('Store_Dept_Avg')
    train_df = train_df.merge(store_dept_means, on=['Store', 'Dept'], how='left')
    test_df = test_df.merge(store_dept_means, on=['Store', 'Dept'], how='left')

    dept_means = train_df.groupby('Dept')['Weekly_Sales'].mean().rename('Dept_Avg')
    train_df = train_df.merge(dept_means, on='Dept', how='left')
    test_df = test_df.merge(dept_means, on='Dept', how='left')

    overall_mean = train_df['Weekly_Sales'].mean()
    train_df['Store_Dept_Avg'] = train_df['Store_Dept_Avg'].fillna(train_df['Dept_Avg']).fillna(overall_mean)
    test_df['Store_Dept_Avg'] = test_df['Store_Dept_Avg'].fillna(test_df['Dept_Avg']).fillna(overall_mean)

    # Feature List
    feature_cols = [
        'Store', 'Dept', 'Size', 'Type_Code', 'IsHoliday',
        'Year', 'Month', 'Week', 'Day', 'Store_Dept_Avg',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'MarkDown1_Active', 'MarkDown2_Active', 'MarkDown3_Active', 'MarkDown4_Active', 'MarkDown5_Active'
    ]

    categorical_features = ['Store', 'Dept', 'Type_Code', 'IsHoliday']

    X_train = train_df[feature_cols]
    y_train = train_df['Weekly_Sales']
    X_test = test_df[feature_cols]
    y_test = test_df['Weekly_Sales']

    # 5. Model 1: Baseline Linear Model (Ridge with Standardization)
    print("\n--- Training Model 1: Ridge Regression (Standardized Baseline) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    # 6. Model 2: Random Forest Regressor
    print("--- Training Model 2: Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 7. Model 3: Best Model - HistGradientBoostingRegressor (with categorical handling)
    print("--- Training Model 3: HistGradientBoostingRegressor (Best Gradient Booster) ---")
    hgb = HistGradientBoostingRegressor(
        categorical_features=categorical_features,
        max_iter=300,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42
    )
    t0 = time.time()
    hgb.fit(X_train, y_train)
    hgb_train_time = time.time() - t0
    y_pred_hgb = hgb.predict(X_test)

    # Weighted Mean Absolute Error (Walmart Official Metric)
    # Holiday weeks have 5x weight
    holiday_weights = np.where(test_df['IsHoliday'] == 1, 5, 1)

    def calc_metrics(y_true, y_pred, name):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        wmae = float(np.sum(holiday_weights * np.abs(y_true - y_pred)) / np.sum(holiday_weights))
        return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'WMAE': wmae}

    results = [
        calc_metrics(y_test, y_pred_ridge, 'Ridge (Standardized Baseline)'),
        calc_metrics(y_test, y_pred_rf, 'Random Forest (50 trees, depth 16)'),
        calc_metrics(y_test, y_pred_hgb, 'HistGradientBoosting (Best Model)')
    ]

    results_df = pd.DataFrame(results)
    print("\n================== Model Evaluation & Comparison ==================")
    print(results_df.to_string(index=False))
    print("===================================================================\n")

    # 8. Save Best Model and Preprocessor Bundle
    model_bundle = {
        'model': hgb,
        'feature_cols': feature_cols,
        'categorical_features': categorical_features,
        'store_dept_means': store_dept_means.to_dict(),
        'dept_means': dept_means.to_dict(),
        'overall_mean': overall_mean,
        'type_mapping': TYPE_MAPPING,
        'metrics': calc_metrics(y_test, y_pred_hgb, 'HistGradientBoosting (Best Model)')
    }

    model_filename = 'best_walmart_model.joblib'
    joblib.dump(model_bundle, model_filename)
    print(f" Best model successfully saved to '{model_filename}'!")

    # 9. Quick Prediction Demo
    print("\n--- Example Prediction Test ---")
    sample_row = X_test.iloc[[0]].copy()
    actual_val = y_test.iloc[0]
    predicted_val = hgb.predict(sample_row)[0]
    print(f"Sample Store: {sample_row['Store'].values[0]}, Dept: {sample_row['Dept'].values[0]}")
    print(f"Actual Weekly Sales:    ${actual_val:,.2f}")
    print(f"Predicted Weekly Sales: ${predicted_val:,.2f}")
    print(f"Prediction Difference:  ${abs(actual_val - predicted_val):,.2f}")

    return results_df, model_bundle

def predict_single(sample_dict, model_path='best_walmart_model.joblib'):
    """
    Convenience function to predict sales for any given store, department, and week.
    """
    bundle = joblib.load(model_path)
    model = bundle['model']
    feature_cols = bundle['feature_cols']

    row = pd.DataFrame([sample_dict])

    # Date breakdown if Date provided
    if 'Date' in row.columns:
        date = pd.to_datetime(row['Date'])
        row['Year'] = date.dt.year
        row['Month'] = date.dt.month
        row['Week'] = date.dt.isocalendar().week.astype(int)
        row['Day'] = date.dt.day

    if 'Type' in row.columns and 'Type_Code' not in row.columns:
        row['Type_Code'] = row['Type'].map(bundle['type_mapping']).fillna(0).astype(int)

    # Store_Dept_Avg lookup
    store = row['Store'].values[0]
    dept = row['Dept'].values[0]
    store_dept_means = bundle['store_dept_means']
    dept_means = bundle['dept_means']
    overall_mean = bundle['overall_mean']

    avg_val = store_dept_means.get((store, dept), dept_means.get(dept, overall_mean))
    row['Store_Dept_Avg'] = avg_val

    # Ensure all feature columns exist with defaults
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    return float(model.predict(row[feature_cols])[0])

if __name__ == '__main__':
    train_and_evaluate_models()
