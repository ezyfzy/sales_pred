import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_prepared_data():
    """Load the prepared datasets"""
    data_dir = 'data'
    train_df = pd.read_csv(os.path.join(data_dir, 'prepared_train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'prepared_test.csv'))
    return train_df, test_df

def prepare_training_data(df):
    """Prepare training data by splitting features and target"""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Identify categorical columns (object dtype)
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Initialize dictionary to store label encoders
    label_encoders = {}
    
    # Encode categorical variables
    for column in categorical_columns:
        if column not in ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column].astype(str))
    
    # Columns to drop
    columns_to_drop = ['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier']
    
    # Get existing columns to drop
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    # Split features and target
    X = df.drop(existing_columns, axis=1)
    y = df['Item_Outlet_Sales']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function for XGBoost optimization"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBRegressor(**param,early_stopping_rounds=50,
        verbose=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def train_xgboost_model(X_train, X_val, y_train, y_val, n_trials=100):
    """Train XGBoost model with Optuna hyperparameter optimization"""
    print("Optimizing hyperparameters with Optuna...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    print("  RMSE: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = trial.params
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = 42
    
    best_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50,
        verbose=False)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
       
    )
    
    # Make predictions
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print("\nFinal Model Performance:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Validation R2: {val_r2:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Plot optimization history
    try:
        import plotly.express as px
        history = pd.DataFrame({
            'Trial': range(len(study.trials)),
            'RMSE': [t.value for t in study.trials]
        })
        fig = px.line(history, x='Trial', y='RMSE', title='Optimization History')
        fig.write_html('optimization_history.html')
        print("\nOptimization history plot saved as 'optimization_history.html'")
    except ImportError:
        print("\nPlotly not installed. Skipping optimization history plot.")
    
    return best_model, feature_importance, study

def save_model_and_features(model, feature_importance, study, output_dir='models'):
    """Save trained model, feature importance, and study results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Save study results
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"\nModel, feature importance, and study results saved in {output_dir} directory")

def predict_test_data(model, test_df):
    """Make predictions on test data"""
    # Prepare test features
    # Create a copy to avoid modifying the original dataframe
    test_df = test_df.copy()
    
    # Identify categorical columns (object dtype)
    categorical_columns = test_df.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    for column in categorical_columns:
        if column not in ['Item_Identifier', 'Outlet_Identifier']:
            test_df[column] = LabelEncoder().fit_transform(test_df[column].astype(str))
    
    # Columns to drop
    columns_to_drop = ['Item_Identifier', 'Outlet_Identifier']
    
    # Get existing columns to drop
    existing_columns = [col for col in columns_to_drop if col in test_df.columns]
    
    # Prepare test features
    X_test = test_df.drop(existing_columns, axis=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    # Save predictions
    submission.to_csv('data/predictions.csv', index=False)
    print("\nPredictions saved as 'predictions.csv'")

def main():
    # Load prepared data
    print("Loading prepared data...")
    train_df, test_df = load_prepared_data()
    
    # Prepare training data
    print("Preparing training data...")
    X_train, X_val, y_train, y_val = prepare_training_data(train_df)
    
    # Train XGBoost model with Optuna
    print("Training XGBoost model with Optuna optimization...")
    best_model, feature_importance, study = train_xgboost_model(
        X_train, X_val, y_train, y_val, n_trials=100
    )
    
    # Save model and results
    save_model_and_features(best_model, feature_importance, study)
    
    # Make predictions on test data
    print("\nMaking predictions on test data...")
    predict_test_data(best_model, test_df)

if __name__ == "__main__":
    main() 