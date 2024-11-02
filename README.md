# Physics ML Project

This repository contains the code and data for a machine learning model designed to predict the G0W0 band gap for material samples based on their properties. The project is built using XGBoost for regression and includes hyperparameter tuning, accuracy assessment, and various performance metrics for scientific evaluation.

## Project Overview

The project includes the following main components:

- **Data Preprocessing**: Handles loading, cleaning, and preparing data for model training.
- **Model Training and Evaluation**: Utilizes XGBoost regression to predict band gaps, with hyperparameter optimization through GridSearchCV.
- **Performance Metrics**: Evaluates the model with metrics like MSE (Mean Squared Error) and visualizes results through learning curves and residuals.

## Project Structure

- **`config.yaml`**: Configuration file containing hyperparameter options and other settings.
- **`launch.py`**: Main file to launch training and evaluation processes.
- **`src/`**: Contains the source code files.
   - **`data_processor.py`**: Data loading and preprocessing.
   - **`distributed_trainer.py`**: Distributes the training across available computing resources.
   - **`models.py`**: Defines the model training and evaluation functions.
   - **`plot_utils.py`**: Helper functions to create plots for visualizing learning curves and residuals.
- **`requirements.txt`**: Lists all required Python packages.
- **`slurm_submit.sh`**: SLURM script for submitting the training job on an HPC environment.

## Setup and Usage

### Prerequisites

- Python 3.8 or later
- XGBoost, scikit-learn, pandas, and matplotlib (installable via `requirements.txt`)

### Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/physics_ml_project.git
   cd physics_ml_project
2.	Install dependencies:
bash
pip install -r requirements.txt
3.	Run the Project:
o	To run the model training and evaluation, execute:
bash
python launch.py
4.	Results and Outputs:
o	After completion, model metrics, optimal hyperparameters, and plots will be saved in the root directory.
Project Details
•	Hyperparameters: Tuned using GridSearchCV to optimize for best model performance.
•	Evaluation Metrics: Mean Squared Error (MSE) is used for validation and test sets.
•	Predictions: Predicts band gaps with the trained model; outputs are available for analysis and interpretation.
Contributing
If you wish to improve this project, please feel free to fork the repository and create a pull request.

