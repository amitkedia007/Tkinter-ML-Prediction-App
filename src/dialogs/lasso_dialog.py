import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from model.lasso import LassoRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LassoDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = X
        self.Y = Y.ravel() if Y is not None else None
        self.theme = theme
        self.model = LassoRegression()  # Default initialization
        self.lambda_values = []
        self.mse_scores = []

        self.title('Lasso Regression Training')
        self.configure(bg=theme)
        self.geometry('600x700')  # Adjusted for additional button
        self.create_widgets()


    def create_widgets(self):
        ttk.Label(self, text="Enter Learning Rate:").pack(padx=10, pady=(20, 5))
        self.learning_rate_entry = ttk.Entry(self)
        self.learning_rate_entry.pack(padx=10, pady=(0, 10))

        ttk.Label(self, text="Enter No. of Iterations:").pack(padx=10, pady=(5, 5))
        self.no_of_iterations_entry = ttk.Entry(self)
        self.no_of_iterations_entry.pack(padx=10, pady=(0, 10))

        ttk.Label(self, text="Enter Lambda Value:").pack(padx=10, pady=(5, 5))
        self.lambda_entry = ttk.Entry(self)
        self.lambda_entry.pack(padx=10, pady=(0, 20))

        ttk.Button(self, text="Train Model with Custom Hyperparameters", command=self.train_custom).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Default Parameters", command=self.train_default).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Train Model with Hyperparameter Tuning", command=self.train_with_tuning).pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(self, text="Visualize Results", command=self.visualize_results).pack(padx=10, pady=10, fill=tk.X)

        self.result_text = tk.Text(self, height=15, width=50, wrap=tk.WORD)
        self.result_text.pack(padx=10, pady=20, expand=True, fill=tk.BOTH)

    def train_custom(self):
        try:
            learning_rate = float(self.learning_rate_entry.get())
            no_of_iterations = int(self.no_of_iterations_entry.get())
            lambda_value = float(self.lambda_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for learning rate, number of iterations, and lambda.")
            return

        self.model = LassoRegression(learning_rate=learning_rate, no_of_iterations=no_of_iterations, lambda_parameter=lambda_value)
        self.train_model(f"Custom Hyperparameters:\nLearning Rate: {learning_rate}, No. of Iterations: {no_of_iterations}, Lambda: {lambda_value}")

    def train_default(self):
        default_params = {"learning_rate": 0.01, "no_of_iterations": 1000, "lambda_parameter": 0.1}
        self.model = LassoRegression(**default_params)
        self.train_model(f"Default Parameters:\nLearning Rate: {default_params['learning_rate']}, No. of Iterations: {default_params['no_of_iterations']}, Lambda: {default_params['lambda_parameter']}")

    def train_model(self, hyperparameters_text):
        if self.X is not None and self.Y is not None:
            self.model.fit(self.X, self.Y)
            mse, r_squared = self.model.evaluate(self.X, self.Y)
            self.display_results(hyperparameters_text, mse, r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def train_with_tuning(self):
        if self.X is not None and self.Y is not None:
            lambda_values = [0.001, 0.01, 0.1, 1, 10]
            best_lambda, best_mse, best_r_squared, tuning_details, lambdas, mses = self.model.tune_and_fit(self.X, self.Y, lambda_values, k_folds=5)
            self.lambda_values = lambdas
            self.mse_scores = mses
            final_message = f"{tuning_details}\nBest Lambda: {best_lambda}, Best MSE: {best_mse:.4f}, Best R-squared: {best_r_squared:.4f}"
            self.display_results(final_message, best_mse, best_r_squared)
        else:
            messagebox.showerror("Error", "Data not available or not properly formatted.")

    def display_results(self, parameters_used, mse, r_squared=None):
        self.result_text.delete('1.0', tk.END)
        result_msg = f"Parameters used to train the model:\n{parameters_used}\n\nResults:\nMSE: {mse:.4f}"
        if r_squared is not None:
            result_msg += f"\nR-squared: {r_squared:.4f}"
        self.result_text.insert(tk.END, result_msg)
    def visualize_results(self):
        if not self.lambda_values or not self.mse_scores:
            messagebox.showerror("Visualization Error", "No tuning data available for visualization.")
            return
        
        visualization_window = tk.Toplevel(self)
        visualization_window.title("Lambda vs. MSE Score")
        
        fig = Figure(figsize=(6, 4), dpi=100)
        plot = fig.add_subplot(1, 1, 1)
        
        plot.plot(self.lambda_values, self.mse_scores, marker='o', linestyle='-')
        plot.set_title("Lambda vs. MSE Score")
        plot.set_xlabel("Lambda")
        plot.set_ylabel("MSE Score")
        
        canvas = FigureCanvasTkAgg(fig, master=visualization_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
