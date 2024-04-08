import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from model.knn import K_Nearest_Neighbour

class KNNDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = np.asarray(X) if X is not None else None
        self.Y = np.asarray(Y).flatten() if Y is not None else None
        self.theme = theme
        self.model = K_Nearest_Neighbour(k=3)
        
        self.title('KNN Training')
        self.configure(bg=theme)
        self.geometry('600x500')
        
        self.create_widgets()

    def create_widgets(self):
        ttk.Button(self, text="Train Model with Default K", command=self.train_default).pack(padx=10, pady=(20, 10), fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)
        
        self.result_text = tk.Text(self, height=10, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=(10, 20), expand=True, fill=tk.BOTH)

    def train_default(self):
        if self.X is not None and self.Y is not None:
            self.model.fit(self.X, self.Y)
            score = self.model.score(self.X, self.Y)
            # Check if score is a tuple (mse, r_squared) or a single value
            if isinstance(score, tuple):
                mse, r_squared = score
            else:
                mse = score
                r_squared = None  # Or handle as appropriate
            self.display_results("Default K=3", mse, r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def train_with_tuning(self):
        if self.X is not None and self.Y is not None:
            k_values = list(range(1, 21))
            best_k, best_mse, best_r_squared = self.model.cross_validate(self.X, self.Y, k_values, n_splits=5)
            self.display_results(f"K-Fold CV Best K={best_k}", best_mse, best_r_squared) 
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def display_results(self, parameters_used, mse, r_squared=None):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"The Parameters used to train the model to get the below Results:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}"
        if r_squared is not None:
            result_msg += f"\nR-squared: {r_squared:.4f}"
        self.result_text.insert(tk.END, result_msg)
