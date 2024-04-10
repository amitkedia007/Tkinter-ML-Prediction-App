import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from model.knn import K_Nearest_Neighbour
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from matplotlib import pyplot as plt
    
class KNNDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = np.asarray(X) if X is not None else None
        self.Y = np.asarray(Y).flatten() if Y is not None else None
        self.theme = theme
        self.model = K_Nearest_Neighbour(k=3)
        
        self.title('KNN Training')
        self.configure(bg=theme)
        self.geometry('600x600')  # Adjusted for additional visualization space
        
        self.create_widgets()

    def create_widgets(self):
        ttk.Button(self, text="Train Model with Default K", command=lambda: self.train_model(3, "Default K=3")).pack(padx=10, pady=(20, 10), fill=tk.X)
        
        self.k_entry = ttk.Entry(self)
        self.k_entry.pack(padx=10, pady=(10, 20))
        ttk.Button(self, text="Train Model with Custom K", command=self.train_custom_k).pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)

        # Add a button for visualization
        ttk.Button(self, text="Visualize Results", command=self.visualize_results).pack(padx=10, pady=10, fill=tk.X)
        
        self.result_text = tk.Text(self, height=10, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=(10, 20), expand=True, fill=tk.BOTH)

    def train_custom_k(self):
        try:
            custom_k = int(self.k_entry.get())
            self.train_model(custom_k, f"Custom K={custom_k}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for K.")

    def train_model(self, k, description):
        if self.X is not None and self.Y is not None:
            self.model.k = k
            self.model.fit(self.X, self.Y)
            mse, r_squared = self.model.score(self.X, self.Y)
            # Use a unified way to display results.
            results_msg = f"{description}\n\nResults:\nMSE: {mse:.4f}\nR-squared: {r_squared:.4f}"
            self.display_results(results_msg)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def train_with_tuning(self):
        if self.X is not None and self.Y is not None:
            k_values = list(range(1, 21))
            best_k, best_score_mse, best_score_r2, detailed_results = self.model.cross_validate(self.X, self.Y, k_values, n_splits=5)
            final_results = detailed_results + f"\nBest K: {best_k}, Best MSE: {best_score_mse:.4f}, Best R-squared: {best_score_r2:.4f}"
            self.display_results(final_results)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def display_results(self, results):
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, results)
    
    def visualize_results(self):
        # Implement visualization of MSE and R² for varying k values
        k_values = range(1, 21)
        mses = []
        r2s = []

        for k in k_values:
            self.model.k = k
            mse_scores = cross_val_score(self.model, self.X, self.Y, cv=5, scoring=make_scorer(mean_squared_error))
            r2_scores = cross_val_score(self.model, self.X, self.Y, cv=5, scoring='r2')
            mses.append(np.mean(mse_scores))
            r2s.append(np.mean(r2_scores))

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(k_values, mses, marker='o', linestyle='-', color='r')
        plt.title('K vs MSE')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('MSE')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(k_values, r2s, marker='o', linestyle='-', color='b')
        plt.title('K vs R²')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('R²')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

