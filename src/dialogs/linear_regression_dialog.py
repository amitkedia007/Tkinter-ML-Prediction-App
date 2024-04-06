import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
from model.multiple_linear_regression import MultipleLinearRegression

class LinearRegressionDialog(tk.Toplevel):
    def __init__(self, master, X, Y, theme='#345'):
        super().__init__(master)
        self.X = X
        self.Y = Y
        self.theme = theme
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

        # Results Frame
        self.results_frame = tk.Frame(self, bg=self.theme)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)  # Added padding for the frame

        # Results Labels (initially empty)
        self.mse_label = tk.Label(self.results_frame, text="", bg=self.theme, fg='white', font=self.fontStyleLabel)
        self.mse_label.pack(pady=(10, 5))  # Added padding for aesthetic spacing
        
        self.r_squared_label = tk.Label(self.results_frame, text="", bg=self.theme, fg='white', font=self.fontStyleLabel)
        self.r_squared_label.pack(pady=(5, 10))  # Added padding for aesthetic spacing

    def train_model(self):
        model = MultipleLinearRegression()
        model.fit(self.X, self.Y)
        mse, r_squared = model.score(self.X, self.Y)

        # Update labels with results, now using custom fonts
        self.mse_label.config(text=f"MSE: {mse:.4f}")
        self.r_squared_label.config(text=f"R-squared: {r_squared:.4f}")
