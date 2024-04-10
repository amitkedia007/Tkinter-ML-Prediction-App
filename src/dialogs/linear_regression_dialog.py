import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
from model.multiple_linear_regression import MultipleLinearRegression
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class LinearRegressionDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = X
        self.Y = Y
        self.theme = theme
        self.model = MultipleLinearRegression()  # Initialize the model here for later use
        self.title('Linear Regression Training')
        self.configure(bg=theme)
        self.geometry('700x500')  # Adjust size as per requirement

        # Fonts
        self.fontStyleButton = tkFont.Font(family="Lucida Grande", size=12)
        self.fontStyleLabel = tkFont.Font(family="Lucida Grande", size=10)

        self.create_widgets()

    def create_widgets(self):
        # Train Button
        train_button = tk.Button(self, text="Train Model", command=self.train_model, bg=self.theme, fg='white', font=self.fontStyleButton)
        train_button.pack(pady=(20, 10))  # Increased padding for visual separation

        # Visualize Button
        visualize_button = tk.Button(self, text="Visualize the Results", command=self.visualize_results, bg=self.theme, fg='white', font=self.fontStyleButton)
        visualize_button.pack(pady=(10, 20))  # Padding for visual separation

        # Results Frame
        self.results_frame = tk.Frame(self, bg=self.theme)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)  # Added padding for the frame

        # Results Labels (initially empty)
        self.mse_label = tk.Label(self.results_frame, text="", bg=self.theme, fg='white', font=self.fontStyleLabel)
        self.mse_label.pack(pady=(10, 5))  # Added padding for aesthetic spacing
        
        self.r_squared_label = tk.Label(self.results_frame, text="", bg=self.theme, fg='white', font=self.fontStyleLabel)
        self.r_squared_label.pack(pady=(5, 10))  # Added padding for aesthetic spacing

    def train_model(self):
        self.model.fit(self.X, self.Y)
        mse, r_squared = self.model.score(self.X, self.Y)
        self.y_pred = self.model.predict(self.X)  # Store predictions for visualization

        # Update labels with results
        self.mse_label.config(text=f"MSE: {mse:.4f}")
        self.r_squared_label.config(text=f"R-squared: {r_squared:.4f}")

    def visualize_results(self):
        # Convert self.Y and self.y_pred to numpy arrays if they aren't already
        Y_actual = np.asarray(self.Y).flatten()
        Y_predicted = np.asarray(self.y_pred).flatten()

        # Ensure both are 1D arrays of the same length
        if Y_actual.shape != Y_predicted.shape:
            messagebox.showerror("Visualization Error", "The shapes of actual and predicted values do not match.")
            return

        # Now proceed with your visualization code
        plt.figure(figsize=(10, 5))

        # Actual vs. Predicted Plot
        plt.subplot(1, 2, 1)
        plt.scatter(Y_actual, Y_predicted, alpha=0.5)
        plt.plot([Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')

        # QQ Plot of Residuals
        residuals = Y_actual - Y_predicted
        plt.subplot(1, 2, 2)
        sm.qqplot(residuals, line='45', fit=True)
        plt.title('QQ Plot of Residuals')

        plt.tight_layout()
        plt.show()
