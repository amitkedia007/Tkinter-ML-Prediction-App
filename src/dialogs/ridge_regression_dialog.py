import tkinter as tk
from tkinter import ttk, messagebox
from model.ridge_regression import RidgeRegression
import numpy as np

class RidgeRegressionDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        # Ensure X and Y are numpy arrays
        self.X = np.asarray(X.select_dtypes(include=[np.number])) if X is not None else None
        self.Y = np.asarray(Y).flatten() if Y is not None else None  # Ensure Y is 1D
        self.theme = theme
        
        self.title('Ridge Regression Training')
        self.configure(bg=theme)
        self.geometry('600x500')  # Adjust size as per requirement
        
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Enter Alpha Value:").pack(padx=10, pady=(20, 10))
        self.alpha_entry = ttk.Entry(self)
        self.alpha_entry.pack(padx=10, pady=(0, 20))
        
        ttk.Button(self, text="Train Model with Default Parameters", command=self.train_default).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Custom Alpha", command=self.train_custom_alpha).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)
        
        self.result_text = tk.Text(self, height=10, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=(10, 20), expand=True, fill=tk.BOTH)

    def train_default(self):
        default_model = RidgeRegression()
        default_model.fit(self.X, self.Y)
        mse, r_squared = default_model.score(self.X, self.Y)
        self.display_results("Default Parameters (alpha=0.1)", mse, r_squared)
        
    def train_custom_alpha(self):
        try:
            # Retrieve the user input and convert to float
            alpha = float(self.alpha_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for alpha.")
            return
        
        # Train the model with the custom alpha
        custom_model = RidgeRegression(alpha=alpha)
        custom_model.fit(self.X, self.Y)
        mse, r_squared = custom_model.score(self.X, self.Y)
        self.display_results(f"Custom Alpha (alpha={alpha})", mse, r_squared)

    def train_with_tuning(self):
        tuning_model = RidgeRegression()
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        best_alpha, best_score, best_r_squared = tuning_model.k_fold_cross_validation(self.X, self.Y, k=5, alphas=alphas)
        parameters_used = f"Hyperparameters using K-Fold Cross Validation (Best alpha={best_alpha})"
        self.display_results(parameters_used, best_score, best_r_squared)

    def display_results(self, parameters_used, mse, r_squared):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"The Parameters used to train the model to get the below Results:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}\nR-squared: {r_squared:.4f}"
        self.result_text.insert(tk.END, result_msg)
