import tkinter as tk
from tkinter import ttk, messagebox
from model.ridge_regression import RidgeRegression  # Ensure this is the correct import path
import pandas as pd

class RidgeRegressionDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = X
        self.Y = Y
        self.theme = theme
        self.model = RidgeRegression()
        
        self.title('Ridge Regression Training')
        self.configure(bg=theme)
        self.geometry('600x500')  # Adjust size as per requirement
        
        self.create_widgets()

    def create_widgets(self):
        ttk.Button(self, text="Train Model with Default Parameters", command=self.train_default).pack(padx=10, pady=(20, 10), fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)
        
        self.result_text = tk.Text(self, height=10, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=(10, 20), expand=True, fill=tk.BOTH)

    def train_default(self):
        # Train the model with default alpha
        self.model.fit(self.X, self.Y)
        mse, r_squared = self.model.score(self.X, self.Y)
        # Ensure mse and r_squared are scalars for display
        mse = mse if isinstance(mse, (int, float)) else mse.item()
        r_squared = r_squared if isinstance(r_squared, (int, float)) else r_squared.item()
        self.display_results("Default Parameters (alpha=0.1)", mse, r_squared)

    def train_with_tuning(self):
    # Example alpha values for tuning
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        best_alpha, best_score, best_r_squared = self.model.tune_and_fit(self.X, self.Y, alphas)  # Update to capture R-squared
        parameters_used = f"Hyperparameters (Best alpha={best_alpha})"
        self.display_results(parameters_used, best_score, best_r_squared)  # Pass R-squared to display

    def display_results(self, parameters_used, mse, r_squared):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"The Parameters used to train the model to get the below Results:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}"
        # Ensure r_squared is a scalar for display
        if r_squared is not None:
            r_squared_value = r_squared.item() if isinstance(r_squared, pd.Series) else r_squared
            result_msg += f"\nR-squared: {r_squared_value:.4f}"
        self.result_text.insert(tk.END, result_msg)

