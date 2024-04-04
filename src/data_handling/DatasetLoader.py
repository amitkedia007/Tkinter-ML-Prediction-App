import pandas as pd
from tkinter import filedialog, messagebox, ttk
import traceback
class DatasetLoader:
    def __init__(self, treeview, data_description_text, callback=None):
        """
        Initialize the DatasetLoader with the Treeview and data description widgets.
        :param treeview: The Treeview widget to display the dataset
        :param data_description_text: The Text widget to display dataset details
        """
        self.treeview = treeview
        self.data_description_text = data_description_text
        self.dataframe = None
        self.callback = callback
        


    def load_dataset(self):
        """
        Load a dataset from a file using a dialog.
        """
        filename = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("Text Files", "*.txt")]
        )
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.dataframe = pd.read_csv(filename)
                elif filename.endswith('.xlsx'):
                    self.dataframe = pd.read_excel(filename)
                elif filename.endswith('.txt'):
                    self.dataframe = pd.read_csv(filename, delimiter='\t')
                
                # After loading the dataframe successfully, update the UI components.
                self.update_treeview()
                self.update_data_description()

                # Debug print to check the loaded dataframe
                print("Dataset loaded successfully.")
                print(self.dataframe.head())  # This prints the first few lines of the dataframe for verification.

                # If there is a callback, it should be executed here after successful load.
                if self.callback:
                    self.callback(self.dataframe)

            except Exception as e:
                # If there's an error, show an error message and print the traceback.
                messagebox.showerror("Error", f"Failed to load the dataset: {e}")
                traceback.print_exc()  # Prints the full traceback to help diagnose the issue.
                self.dataframe = None  # Ensure that dataframe is set to None if loading failed.
        else:
            print("No file was selected.")
            self.dataframe = None   
        
    def update_treeview(self):
        """
        Update the Treeview widget with the loaded dataset.
        """
        # Clear the current Treeview content
        self.treeview.delete(*self.treeview.get_children())

        # Set the Treeview columns
        self.treeview['columns'] = list(self.dataframe.columns)
        self.treeview['show'] = 'headings'
        for col in self.dataframe.columns:
            self.treeview.heading(col, text=col)
            self.treeview.column(col, width=100, anchor='center')

        # Insert the data rows into the Treeview
        for index, row in self.dataframe.iterrows():
            self.treeview.insert("", 'end', values=list(row))

    def update_data_description(self):
        """
        Update the Text widget with a description of the dataset.
        """
        self.data_description_text.delete('1.0', 'end')
        description_text = f"Dataset Description:\n\nNumber of Rows: {self.dataframe.shape[0]}\n"
        description_text += f"Number of Columns: {self.dataframe.shape[1]}\n\nColumn Details:\n"
        for column in self.dataframe.columns:
            description_text += f"{column} ({self.dataframe[column].dtype})\n"
        self.data_description_text.insert('end', description_text)

