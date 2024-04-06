# File: gui/UIManager.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from data_handling.DatasetLoader import DatasetLoader
from visualization.DataVisualizer import DataVisualizer
from data_handling.DataPreprocessor import DataPreprocessor
import pandas as pd
from dialogs.linear_regression_dialog import LinearRegressionDialog
from dialogs.ridge_regression_dialog import RidgeRegressionDialog
from dialogs.knn_dialog import KNNDialog
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
        self.dataset_loader = DatasetLoader(self.treeview, self.data_description_text, self.dataset_loaded)
        self.data_visualizer = DataVisualizer(None)
        self.create_preprocess_data_button() 
        self.create_target_variable_selector()
        self.X_preprocessed = pd.DataFrame()
        self.Y_preprocessed = pd.DataFrame()

        
    def configure_gui(self):
        self.master.title('Boston House Price Prediction App')
        self.master.geometry('1300x800')
        self.master.configure(bg='#345')

    def create_widgets(self):
        # Create and place all widgets, frames, labels, etc.
        self.create_data_description_frame()
        self.create_view_dataset_frame()
        self.create_visualization_frame()
        # self.create_data_prep_frame()
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
        loaded_dataframe = self.dataset_loader.load_dataset()
        if loaded_dataframe is not None:
            self.update_data_and_refresh_gui(loaded_dataframe)


    def dataset_loaded(self, dataframe):
        self.dataframe = dataframe  # Assuming this is where you store the loaded dataframe
        self.data_visualizer.update_dataframe(dataframe)  # Update DataVisualizer with the loaded dataframe
        self.refresh_data_viewer()  # Optionally refresh the UI, if applicable
        self.target_dropdown['values'] = self.dataframe.columns.tolist()  # Update target variable selector, if you have one

    def prepare_and_visualize_data(self):
        if self.X_preprocessed is not None and self.Y_preprocessed is not None:
            # Merge the preprocessed features and target
            full_dataframe = pd.concat([self.X_preprocessed, self.Y_preprocessed], axis=1)
            
            # Update the visualizer's dataframe
            self.data_visualizer.update_dataframe(full_dataframe)
            
            self.data_visualizer.show_box_plot()  
        else:
            messagebox.showinfo("Info", "Preprocessed data not available for visualization.")


    def update_data_and_refresh_gui(self, new_dataframe):
        """Updates the DataFrame in DataPreprocessor and refreshes the TreeView and other GUI components."""
        self.dataframe = new_dataframe  
        self.refresh_data_viewer()  
        if self.data_visualizer:  
            self.data_visualizer.update_dataframe(new_dataframe)

    def create_target_variable_selector(self):
        label = tk.Label(self.master, text="Select Target Variable:", bg='#345', fg='white')
        label.place(relx=0.55, rely=0.48)

        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(self.master, textvariable=self.target_var)
        self.target_dropdown.place(relx=0.55, rely=0.51, relwidth=0.4)

        # Button to confirm selection
        confirm_button = tk.Button(self.master, text="Confirm Target", command=self.confirm_target_selection)
        confirm_button.place(relx=0.55, rely=0.55, relwidth=0.4, height=30)

    def confirm_target_selection(self):
        target_var = self.target_var.get()
        if target_var:
            # Split dataset into X and Y based on selected target variable
            self.Y = self.dataframe[[target_var]]
            self.X = self.dataframe.drop(columns=[target_var])
            
            # Here, call your preprocessing functions for X and Y separately
            self.preprocess_features_and_target()
            messagebox.showinfo("Success", f"Target variable '{target_var}' selected successfully.")
        else:
            messagebox.showinfo("Error", "Please select a target variable.")

    def create_preprocess_data_button(self):
        """Create the 'Preprocess the Data' button."""
        preprocess_button = tk.Button(self.master, text="Preprocess the Data", command=self.initiate_preprocessing)
        preprocess_button.place(relx=0.55, rely=0.42, relwidth=0.4, height=30) 
    
    def initiate_preprocessing(self):
        """Handles the preprocessing initiation."""
        if self.dataframe is not None and self.target_var.get():
            self.preprocess_features_and_target()
            self.refresh_tree_view_with_preprocessed_data()
            messagebox.showinfo("Success", "Data preprocessing completed successfully.")
        else:
            messagebox.showinfo("Error", "Please load a dataset and select a target variable.")

    def refresh_tree_view_with_preprocessed_data(self):
        # Clear the treeview
        self.treeview.delete(*self.treeview.get_children())

        # Assuming X_preprocessed and Y_preprocessed are correctly set after preprocessing
        # Merge them for display
        full_df = pd.concat([self.X_preprocessed, self.Y_preprocessed], axis=1)

        # Insert new rows into the treeview
        for index, row in full_df.iterrows():
            self.treeview.insert("", 'end', values=row.tolist())

    def preprocess_features_and_target(self):
        """Separates and preprocesses features and target."""
        target_var = self.target_var.get()
        if target_var:
            self.Y = self.dataframe[[target_var]]
            self.X = self.dataframe.drop(columns=[target_var])
            
            # Preprocess features
            feature_preprocessor = DataPreprocessor(self.X)
            self.X_preprocessed = feature_preprocessor.preprocess_data()
            
            # Preprocess target, adjust according to your DataPreprocessor capabilities
            target_preprocessor = DataPreprocessor(self.Y)
            self.Y_preprocessed = target_preprocessor.preprocess_data()  # Adjust if you have a specific method for target
            
            # Now you can update the UI or proceed to model training with preprocessed data
            # e.g., self.refresh_data_viewer() with preprocessed data
        else:
            messagebox.showinfo("Error", "Target variable not set.")

    # def get_column_names(self):
    #     # This method should return the list of column names from the dataframe
    #     if self.data_preprocessor and hasattr(self.data_preprocessor, 'dataframe'):
    #         return self.data_preprocessor.dataframe.columns.tolist()
    #     else:
    #         return []

    # Visualization Related Functions
    
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

    
    def refresh_data_viewer(self):
        if self.data_preprocessor and hasattr(self.data_preprocessor, 'dataframe'):
            self.treeview.delete(*self.treeview.get_children())
            for index, row in self.data_preprocessor.dataframe.iterrows():
                self.treeview.insert("", 'end', values=row.tolist())
        else:
            messagebox.showinfo("Info", "No data available to display.")


    # def create_data_prep_frame(self):
    #     # Logic for creating data preparation frame
    #     self.data_prep_frame = tk.Frame(self.master, bg='#345')
    #     self.data_prep_frame.place(relx=0.5, relwidth=0.5, relheight=0.25, rely=0.48)

    #     self.create_data_preprocessing_section(self.data_prep_frame)

    # def create_data_preprocessing_section(self, frame):
    # # Data Preprocessing Section
    #     self.preprocess_entries = {}  # Dictionary to hold Entry widgets for each preprocessing action

    #     # Define preprocessing actions
    #     actions = ["Fill Mean", "Fill Unknown", "Drop Columns", "Convert Categorical"]
    #     for i, action in enumerate(actions):
    #         # Label for the action
    #         tk.Label(frame, text=action, bg='#345', fg='white').grid(row=i, column=0, padx=10, pady=5, sticky="w")
            
    #         # Entry for typing in the column name
    #         action_entry = tk.Entry(frame)
    #         action_entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    #         self.preprocess_entries[action] = action_entry

    #         # Button to apply the preprocessing action
    #         action_button = tk.Button(frame, text=action, command=lambda act=action: self.apply_preprocessing(act))
    #         action_button.grid(row=i, column=2, padx=10, pady=5, sticky="ew")

    # Update to ensure data_preprocessor is initialized with the dataframe when loaded
    def update_data_visualizer_and_preprocessor(self, dataframe):
        self.data_visualizer = DataVisualizer(dataframe)
        self.data_preprocessor = DataPreprocessor(dataframe)
        
    
    def refresh_data_viewer(self):
        # Clear the current contents of the Treeview
        self.treeview.delete(*self.treeview.get_children())

        # Assuming 'self.treeview' is already created and 
        # 'self.dataset_loader.dataframe' holds the updated data
        for index, row in self.dataset_loader.dataframe.iterrows():
            self.treeview.insert("", 'end', values=row.to_list())

    def preprocess_data_and_update_ui(self):
        if self.dataframe is not None:
            preprocessor = DataPreprocessor(self.dataframe)
            preprocessed_df = preprocessor.preprocess_data()  # Optionally, set drop_correlated, scale_data
            self.preprocessed_dataframe = preprocessed_df  # Store the preprocessed dataframe for model training
            self.refresh_data_viewer(preprocessed_df)  # Update UI, such as a TreeView, to reflect preprocessed data
            messagebox.showinfo("Success", "Data preprocessing completed successfully.")
        else:
            messagebox.showinfo("Error", "No data loaded for preprocessing.")

# All the Model training related functions 

    def create_model_train_frame(self):
        # Logic for creating model train frame
        self.model_train_frame = tk.Frame(self.master, bg='#345')
        self.model_train_frame.place(relx=0.5, rely=0.73, relwidth=0.5, relheight=0.1)

        model_label = tk.Label(self.model_train_frame, text="Select Model:", bg='#345', fg='white')
        model_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        model_options = ['Linear Regression', 'KNN', 'Decision Tree', 'Ridge Regression']
        self.model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(self.model_train_frame, textvariable=self.model_var, values=model_options)
        model_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        train_button = tk.Button(self.model_train_frame, text="Train Model", command=self.open_training_window)
        train_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

    def update_data_visualizer(self, dataframe):
        self.data_visualizer.dataframe = dataframe
    
    

    def open_linear_regression_dialog(self):
        if hasattr(self, 'X') and hasattr(self, 'Y'):
            # Check if X and Y have been set
            if self.X is not None and self.Y is not None:
                dialog = LinearRegressionDialog(self.master, self.X, self.Y)
                dialog.grab_set()  # Make the dialog modal, if necessary
            else:
                messagebox.showinfo("Error", "Please preprocess the data and set the target variable first.")
        else:
            messagebox.showinfo("Error", "Data not preprocessed or target variable not selected.")

    def open_ridge_regression_dialog(self):
        if hasattr(self, 'X_preprocessed') and hasattr(self, 'Y_preprocessed'):
            # Check if data is preprocessed
            if not self.X_preprocessed.empty and not self.Y_preprocessed.empty:
                dialog = RidgeRegressionDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
                dialog.grab_set() 
            else:
                messagebox.showinfo("Info", "Please preprocess the data first.")
        else:
            messagebox.showinfo("Info", "Data not preprocessed or target variable not selected.")
    
    def open_knn_dialog(self):
        if hasattr(self, 'X_preprocessed') and hasattr(self, 'Y_preprocessed'):
            if not self.X_preprocessed.empty and not self.Y_preprocessed.empty:
                dialog = KNNDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
                dialog.grab_set()  # Make the dialog modal, if necessary
            else:
                messagebox.showinfo("Error", "Please preprocess the data and set the target variable first.")
        else:
            messagebox.showinfo("Error", "Data not preprocessed or target variable not selected.")

    
    def open_training_window(self):
        selected_model = self.model_var.get()

        # Ensure the preprocessed data is available and not empty
        if hasattr(self, 'X_preprocessed') and hasattr(self, 'Y_preprocessed'):
            if not self.X_preprocessed.empty and not self.Y_preprocessed.empty:
                if selected_model == "Linear Regression":
                    # Open Linear Regression Dialog
                    dialog = LinearRegressionDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
                    dialog.grab_set()
                elif selected_model == "Ridge Regression":
                    # Open Ridge Regression Dialog
                    dialog = RidgeRegressionDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
                    dialog.grab_set()
                elif selected_model == "KNN":
                    dialog = KNNDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
                    dialog.grab_set()

                # Add additional elif conditions here for other models like SVR, Decision Tree, etc.
                else:
                    messagebox.showinfo("Error", f"Model {selected_model} not supported yet.")
            else:
                messagebox.showinfo("Error", "Please preprocess the data and set the target variable first.")
        else:
            messagebox.showinfo("Error", "Data not preprocessed or target variable not selected.")


    # def open_training_window(self):
    #     selected_model = self.model_var.get()

    #     # Ensure the data has been preprocessed or loaded
    #     if not self.X_preprocessed or not self.Y_preprocessed:
    #         messagebox.showinfo("Error", "Please preprocess the data first.")
    #         return

    #     # Open the corresponding dialog based on the selected model
    #     if selected_model == "Linear Regression":
    #         dialog = LinearRegressionDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
    #     # elif selected_model == "SVR":
    #     #     # Assuming SVRDialog is defined
    #     #     dialog = SVRDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
    #     # elif selected_model == "Decision Tree":
    #     #     # Assuming DecisionTreeDialog is defined
    #     #     dialog = DecisionTreeDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
    #     # elif selected_model == "Ridge Regression":
    #     #     # Assuming RidgeRegressionDialog is defined
    #     #     dialog = RidgeRegressionDialog(self.master, self.X_preprocessed, self.Y_preprocessed)
    #     else:
    #         messagebox.showinfo("Error", "Model not supported.")
    #         return

    #     dialog.grab_set()  # Make the dialog modal if needed

    
    def run(self):
        self.master.mainloop()

# This should be in main.py
if __name__ == "__main__":
    root = tk.Tk()
    app = UIManager(root)
    app.run()
