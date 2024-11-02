# launch.py

from src.data_processor import DataProcessor
from src.models import ModelTrainer

# Load data
data_processor = DataProcessor("updated_data_set_ab2_abc_ab.xlsx")
X_train, X_val, X_test, y_train, y_val, y_test = data_processor.prepare_data()

# Model and hyperparameter configurations
model_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",  # Assuming GPU usage
    "predictor": "gpu_predictor"
}

hyper_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

# Initialize and train model
model_trainer = ModelTrainer(model_params, hyper_params, config={})
model_trainer.train_and_evaluate(X_train, X_val, y_train, y_val, X_test, y_test)

