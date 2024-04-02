import tkinter as tk
from tkinter import messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class DataVisualizer:
    def __init__(self, dataframe):
        """
        Initialize the DataVisualizer with a pandas DataFrame.
        :param dataframe: pandas DataFrame to visualize.
        """
        self.dataframe = dataframe

    def show_box_plot(self):
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please upload a dataset first.")
            return

        numeric_columns = self.dataframe.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            box_plot_window = tk.Toplevel()
            box_plot_window.title("Box Plot")

            fig = plt.Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            sns.boxplot(data=self.dataframe[numeric_columns], ax=ax)

            canvas = FigureCanvasTkAgg(fig, master=box_plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            messagebox.showwarning("Warning", "No numeric columns available for a box plot.")

    def show_histogram_plot(self):
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please upload a dataset first.")
            return

        numeric_columns = self.dataframe.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            num_features = len(numeric_columns)
            grid_size = int(np.ceil(np.sqrt(num_features)))

            histogram_window = tk.Toplevel()
            histogram_window.title("Histogram Plots")
            plt.rcParams.update({'font.size': 8})
            fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(max(12, grid_size * 4), max(12, grid_size * 4)))
            plt.subplots_adjust(wspace=0.4, hspace=0.6)

            axes = axes.flatten()
            for ax, column in zip(axes, numeric_columns):
                sns.histplot(self.dataframe[column].dropna(), bins=30, kde=False, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Histogram of {column}', fontsize=10)
                ax.set_xlabel(column, fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)

            for ax in axes[num_features:]:
                ax.set_visible(False)

            canvas = FigureCanvasTkAgg(fig, master=histogram_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            messagebox.showwarning("Warning", "No numeric columns available for histogram plots.")

    def show_heat_map(self):
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please upload a dataset first.")
            return

        corr = self.dataframe.corr()
        if corr is not None and not corr.empty:
            heatmap_window = tk.Toplevel()
            heatmap_window.title("Heatmap of Correlation Matrix")

            fig_size = max(8, len(corr.columns) * 0.8)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 8}, square=True)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.subplots_adjust(bottom=0.3, top=0.95)

            canvas = FigureCanvasTkAgg(fig, master=heatmap_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            messagebox.showwarning("Warning", "Cannot display a heatmap with no or non-numeric data.")
