# File: gui/UIManager.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from data_handling.DatasetLoader import DatasetLoader
from visualization.DataVisualizer import DataVisualizer
from data_handling.DataPreprocessor import DataPreprocessor

class UIManager:
    def __init__(self, master):
        self.master = master
        self.dataframe = None  # This will hold the dataframe once loaded
        self.data_preprocessor = None
        self.data_description_text = None
        self.configure_gui()
        self.create_widgets()
        self.create_data_description_frame()  
        self.create_view_dataset_frame()      
        self.dataset_loader = DatasetLoader(self.treeview, self.data_description_text, self.update_data_visualizer)  
        self.data_visualizer = DataVisualizer(self.dataset_loader.dataframe)
        
          # This will set up the UI


    def configure_gui(self):
        self.master.title('Boston House Price Prediction App')
        self.master.geometry('1300x800')
        self.master.configure(bg='#345')

    def create_widgets(self):
        # Create and place all widgets, frames, labels, etc.
        self.create_data_description_frame()
        self.create_view_dataset_frame()
        self.create_visualization_frame()
        self.create_data_prep_frame()
        self.create_model_train_frame()
    
        # Upload Dataset Button
        upload_button = tk.Button(self.master, text="Upload Dataset", command=self.upload_dataset)
        upload_button.place(relx=0.05, rely=0.42, relwidth=0.4, height=30)

    def create_data_description_frame(self):
        self.data_description_frame = tk.Frame(self.master, bg='#345')
        self.data_description_frame.place(relwidth=0.5, relheight=0.4)  # Adjust size and placement as needed

        self.data_description_text = tk.Text(self.data_description_frame, wrap="word", bg='#345', fg='white')
        self.data_description_text.pack(expand=True, fill='both')


    def create_view_dataset_frame(self):
        self.view_dataset_frame = tk.Frame(self.master, bg='#345')
        self.view_dataset_frame.place(relx=0.5, relwidth=0.5, relheight=0.4)  # Adjust size and placement as needed

        self.treeview = ttk.Treeview(self.view_dataset_frame)
        self.treeview.pack(expand=True, fill='both')

    def upload_dataset(self):
        # Load the dataset here with a file dialog or similar method
        self.dataframe = self.dataset_loader.load_dataset()

        # Check if the DataFrame has been loaded correctly
        if self.dataframe is not None:
            self.data_preprocessor = DataPreprocessor(self.dataframe)
            self.update_comboboxes()


    def dataset_loaded(self, dataframe):
        # Now that the dataset is loaded, initialize the DataPreprocessor
        self.data_preprocessor = DataPreprocessor(dataframe)
        # Update the comboboxes with the column names
        self.update_comboboxes()
        # Refresh the data view part of the UI
        self.refresh_data_viewer()

    def update_comboboxes(self):
        # This method updates all comboboxes with the new columns from the loaded dataframe
        if self.data_preprocessor:
            column_names = self.data_preprocessor.dataframe.columns.tolist()
            for combobox in self.preprocess_comboboxes.values():
                combobox['values'] = column_names



    def get_column_names(self):
        # This method should return the list of column names from the dataframe
        if self.data_preprocessor and hasattr(self.data_preprocessor, 'dataframe'):
            return self.data_preprocessor.dataframe.columns.tolist()
        else:
            return []


    def create_visualization_frame(self):
        self.visualization_frame = tk.Frame(self.master, bg='#345')
        self.visualization_frame.place(relwidth=0.5, relheight=0.35, rely=0.48)

        # Buttons for visualization
        visualization_buttons = [
            ("Box Plot", lambda: self.data_visualizer.show_box_plot()),
            ("Histogram Plot", lambda: self.data_visualizer.show_histogram_plot()),
            ("Heat Map", lambda: self.data_visualizer.show_heat_map())
        ]

        for i, (button_text, button_command) in enumerate(visualization_buttons):
            button = tk.Button(self.visualization_frame, text=button_text, command=button_command)
            button.grid(row=i, column=1, padx=10, pady=5, sticky="ew")

    def apply_preprocessing(self, action):
        column_name = self.preprocess_entries[action].get().strip()  # Get the entered column name
        if not column_name:
            messagebox.showwarning("Warning", f"Please enter a column name for {action}.")
            return
        if column_name not in self.dataframe.columns:
            messagebox.showerror("Error", f"Column '{column_name}' not found in the dataset.")
            return
        
        # Execute the corresponding preprocessing method
        try:
            if action == "Fill Mean":
                self.data_preprocessor.fill_mean(column_name)
            elif action == "Fill Unknown":
                self.data_preprocessor.fill_unknown(column_name)
            elif action == "Drop Columns":
                self.data_preprocessor.drop_columns(column_name)
            elif action == "Convert Categorical":
                self.data_preprocessor.convert_categorical(column_name)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        else:
            # Refresh the data viewer to reflect changes
            self.refresh_data_viewer()

    def scale_all_data(self):
        # Call the scale_data method on all numeric columns
        self.data_preprocessor.scale_data()
        self.refresh_data_viewer()  # Refresh only the data viewer part of the UI

    def refresh_data_viewer(self):
        if self.data_preprocessor and hasattr(self.data_preprocessor, 'dataframe'):
            self.treeview.delete(*self.treeview.get_children())
            for index, row in self.data_preprocessor.dataframe.iterrows():
                self.treeview.insert("", 'end', values=row.tolist())
        else:
            messagebox.showinfo("Info", "No data available to display.")


    def create_data_prep_frame(self):
        # Logic for creating data preparation frame
        self.data_prep_frame = tk.Frame(self.master, bg='#345')
        self.data_prep_frame.place(relx=0.5, relwidth=0.5, relheight=0.25, rely=0.48)

        self.create_data_preprocessing_section(self.data_prep_frame)

    def create_data_preprocessing_section(self, frame):
    # Data Preprocessing Section
        self.preprocess_entries = {}  # Dictionary to hold Entry widgets for each preprocessing action

        # Define preprocessing actions
        actions = ["Fill Mean", "Fill Unknown", "Drop Columns", "Convert Categorical"]
        for i, action in enumerate(actions):
            # Label for the action
            tk.Label(frame, text=action, bg='#345', fg='white').grid(row=i, column=0, padx=10, pady=5, sticky="w")
            
            # Entry for typing in the column name
            action_entry = tk.Entry(frame)
            action_entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            self.preprocess_entries[action] = action_entry

            # Button to apply the preprocessing action
            action_button = tk.Button(frame, text=action, command=lambda act=action: self.apply_preprocessing(act))
            action_button.grid(row=i, column=2, padx=10, pady=5, sticky="ew")

    # Update to ensure data_preprocessor is initialized with the dataframe when loaded
    def update_data_visualizer_and_preprocessor(self, dataframe):
        self.data_visualizer = DataVisualizer(dataframe)
        self.data_preprocessor = DataPreprocessor(dataframe)
        # Update combobox choices with new columns
        self.preprocess_comboboxes['column']['values'] = dataframe.columns.tolist()
    
    def refresh_data_viewer(self):
        # Clear the current contents of the Treeview
        self.treeview.delete(*self.treeview.get_children())

        # Assuming 'self.treeview' is already created and 
        # 'self.dataset_loader.dataframe' holds the updated data
        for index, row in self.dataset_loader.dataframe.iterrows():
            self.treeview.insert("", 'end', values=row.to_list())


    def create_model_train_frame(self):
        # Logic for creating model train frame
        self.model_train_frame = tk.Frame(self.master, bg='#345')
        self.model_train_frame.place(relx=0.5, rely=0.73, relwidth=0.5, relheight=0.1)

        model_label = tk.Label(self.model_train_frame, text="Select Model:", bg='#345', fg='white')
        model_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        model_options = ['Linear Regression', 'SVR', 'Decision Tree', 'Ridge Regression']
        self.model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(self.model_train_frame, textvariable=self.model_var, values=model_options)
        model_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        train_button = tk.Button(self.model_train_frame, text="Train Model", command=self.open_training_window)
        train_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

    def update_data_visualizer(self, dataframe):
        # This method will be called by DatasetLoader after the dataset is loaded
        self.data_visualizer = DataVisualizer(dataframe)  # Update or create the DataVisualizer instance
        # self.refresh_visualization_options()  # Update the visualization options if needed
    
    # def refresh_visualization_options(self):
        # Enable visualization buttons or other elements as needed now that data is available
        # pass


    def open_training_window(self):
        # Placeholder for opening the training window logic
        pass

    def run(self):
        self.master.mainloop()

# This should be in main.py
if __name__ == "__main__":
    root = tk.Tk()
    app = UIManager(root)
    app.run()
