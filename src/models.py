# src/models.py

import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from src.plot_utils import plot_learning_curve, plot_residuals

class ModelTrainer:
    def __init__(self, model_params, hyper_params, config):
        # Switch to CPU-compatible tree method
        self.model_params = model_params
        self.model_params["tree_method"] = "hist"  # Changed from "gpu_hist"
        self.model_params["enable_categorical"] = True
        self.hyper_params = hyper_params
        self.config = config
        self.model = xgb.XGBRegressor(**self.model_params)

    def train_and_evaluate(self, X_train, X_val, y_train, y_val, X_test, y_test):
        # Hyperparameter tuning
        grid_search = GridSearchCV(self.model, self.hyper_params, cv=3, scoring='neg_mean_squared_error', error_score='raise')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        # Final training with best hyperparameters
        self.model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        metrics = {
            "Validation MSE": mean_squared_error(y_val, y_val_pred),
            "Validation MAE": mean_absolute_error(y_val, y_val_pred),
            "Validation R2": r2_score(y_val, y_val_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
            "Test R2": r2_score(y_test, y_test_pred),
        }

        # Save metrics and model configuration
        with open("model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        with open("best_hyperparameters.json", "w") as f:
            json.dump(grid_search.best_params_, f, indent=4)
        
        print("Metrics:", metrics)
        
        # Plot learning curves and residuals
        plot_learning_curve(self.model, X_train, y_train, X_val, y_val)
        plot_residuals(y_val, y_val_pred, "Validation Residuals")
        plot_residuals(y_test, y_test_pred, "Test Residuals")

