import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_predictive_modeling():
    print("Initializing Week 4 Predictive Modeling & Optimization Pipeline...")
    
    # 1. Load Simulated Production Logistics Dataset (12,500 records)
    np.random.seed(42)
    n_records = 12500
    df = pd.DataFrame({
        'distance_km': np.random.uniform(2.5, 35.0, n_records),
        'package_weight_kg': np.random.uniform(0.5, 18.5, n_records),
        'traffic_index': np.random.uniform(1.0, 5.0, n_records),
        'driver_experience_yrs': np.random.randint(1, 12, n_records),
        'dispatch_time_hrs': np.random.uniform(0.2, 1.5, n_records),
        'delivery_time_hours': np.random.uniform(0.5, 4.2, n_records)
    })

    # Inject realistic non-linear relationship with traffic and distance
    df['delivery_time_hours'] = (0.08 * df['distance_km']) + (0.45 * df['traffic_index']) + (0.1 * df['dispatch_time_hrs']) + np.random.normal(0, 0.15, n_records)

    # 2. Feature Selection & Train-Test Split
    X = df[['distance_km', 'package_weight_kg', 'traffic_index', 'driver_experience_yrs', 'dispatch_time_hrs']]
    y = df['delivery_time_hours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training with Hyperparameter Tuning (GridSearchCV)
    print("Training Random Forest Regressor with GridSearchCV...")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 12],
        'min_samples_split': [5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # 4. Evaluation & Validation
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"--- Model Evaluation Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.4f} hours")
    print(f"MAE: {mae:.4f} hours")
    print(f"R2 Score: {r2:.4f}")
    print("Predictive Modeling & Optimization Pipeline Completed Successfully!")

if __name__ == '__main__':
    run_predictive_modeling()
  
