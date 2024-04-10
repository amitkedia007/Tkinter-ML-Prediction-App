import tkinter as tk
from tkinter import ttk, messagebox
from model.ridge_regression import RidgeRegression
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RidgeRegressionDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = np.asarray(X.select_dtypes(include=[np.number])) if X is not None else None
        self.Y = np.asarray(Y).flatten() if Y is not None else None
        self.theme = theme
        self.model = RidgeRegression()  # Initialize the model
        
        # Initialization for visualization data
        self.alphas = []
        self.mse_scores = []

        self.title('Ridge Regression Training')
        self.configure(bg=theme)
        self.geometry('600x600')  # Adjusted for the visualization button
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Enter Alpha Value:").pack(padx=10, pady=(20, 10))
        self.alpha_entry = ttk.Entry(self)
        self.alpha_entry.pack(padx=10, pady=(0, 20))

        ttk.Button(self, text="Train Model with Default Parameters", command=self.train_default).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Custom Alpha", command=self.train_custom_alpha).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)

        ttk.Button(self, text="Visualize Results", command=self.visualize_results).pack(padx=10, pady=10, fill=tk.X)

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
        if self.X is not None and self.Y is not None:
            alphas = [0.001, 0.01, 0.1, 1, 10, 100]
            best_alpha, best_score, best_r_squared, alphas_tested, mse_scores_for_alphas = self.model.k_fold_cross_validation(self.X, self.Y, k=5, alphas=alphas)

            self.alphas = alphas_tested
            self.mse_scores = mse_scores_for_alphas

            detailed_results = "\n".join(f"Alpha: {alpha}, Avg MSE: {mse:.4f}" for alpha, mse in zip(alphas_tested, mse_scores_for_alphas))
            self.display_results(f"Hyperparameters Tuning Results:\n{detailed_results}\nBest Alpha: {best_alpha}, Best MSE: {best_score:.4f}, Best R-squared: {best_r_squared:.4f}", best_score, best_r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def visualize_results(self):
        if not self.alphas or not self.mse_scores:
            messagebox.showerror("Visualization Error", "No tuning data available for visualization.")
            return
        visualization_window = tk.Toplevel(self)
        visualization_window.title("Alpha vs. MSE Score")

        fig = Figure(figsize=(6, 4), dpi=100)
        plot = fig.add_subplot(1, 1, 1)

        plot.plot(self.alphas, self.mse_scores, marker='o', linestyle='-')
        plot.set_title("Alpha vs. MSE Score")
        plot.set_xlabel("Alpha")
        plot.set_ylabel("MSE Score")

        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def display_results(self, parameters_used, mse, r_squared):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"The Parameters used to train the model to get the below Results:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}\nR-squared: {r_squared:.4f}"
        self.result_text.insert(tk.END, result_msg)
