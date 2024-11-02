# src/plot_utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_learning_curve(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(train_errors, "r-+", linewidth=2, label="Train")
    plt.plot(val_errors, "b-", linewidth=3, label="Validation")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.savefig("learning_curve.png")
    plt.close()

def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="r", linestyles="dashed")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

