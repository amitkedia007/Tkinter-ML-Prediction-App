import tkinter as tk
from tkinter import ttk, messagebox
from model.decision_tree import DecisionTreeRegressor

class DecisionTreeDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        # Convert X and Y to NumPy arrays if they are Pandas structures
        self.X = X.values if hasattr(X, 'values') else X
        self.Y = Y.values.ravel() if hasattr(Y, 'values') else Y.ravel() if Y is not None else None
        self.theme = theme
        self.model = DecisionTreeRegressor()
        
        self.title('Decision Tree Training')
        self.configure(bg=theme)
        self.geometry('600x500')
        
        self.create_widgets()


    def create_widgets(self):
        ttk.Button(self, text="Train Model with Default Parameters", command=self.train_default).pack(padx=10, pady=20, fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)
        
        self.result_text = tk.Text(self, height=10, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=20, expand=True, fill=tk.BOTH)

    def train_default(self):
        if self.X is not None and self.Y is not None:
            self.model.fit(self.X, self.Y)
            mse, r_squared = self.model.score(self.X, self.Y)
            self.display_results("Default Parameters", mse, r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def train_with_tuning(self):
        if self.X is not None and self.Y is not None:
            max_depth_values = [None] + list(range(1, 11))  # None for unlimited depth and then 1 to 10
            best_max_depth, best_mse, best_r_squared = self.model.cross_validate(self.X, self.Y, max_depth_values, n_splits=5)
            self.display_results(f"K-Fold CV Best max_depth={best_max_depth}", best_mse, best_r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def display_results(self, parameters_used, mse, r_squared=None):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"The Parameters used to train the model to get the below Results:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}"
        if r_squared is not None:
            result_msg += f"\nR-squared: {r_squared:.4f}"
        self.result_text.insert(tk.END, result_msg)
