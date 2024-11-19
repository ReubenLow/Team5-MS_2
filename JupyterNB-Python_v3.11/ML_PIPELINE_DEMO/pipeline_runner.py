import papermill as pm

# Define the notebooks
notebooks = {
    "clean_data": "clean_data.ipynb",
    "model_training": "model_training.ipynb",
    "perform_prediction": "perform_prediction.ipynb",
}

# Define the execution function
def run_pipeline():
    print("Running pipeline...")

    # Execute clean_data notebook
    print("Step 1: Cleaning data...")
    pm.execute_notebook(
        notebooks["clean_data"],
        notebooks["clean_data"].replace(".ipynb", "_output.ipynb")
    )
    print("Data cleaning completed.")

    # Execute model_training notebook
    print("Step 2: Training models...")
    pm.execute_notebook(
        notebooks["model_training"],
        notebooks["model_training"].replace(".ipynb", "_output.ipynb")
    )
    print("Model training completed.")

    # Execute perform_prediction notebook
    print("Step 3: Making predictions...")
    pm.execute_notebook(
        notebooks["perform_prediction"],
        notebooks["perform_prediction"].replace(".ipynb", "_output.ipynb")
    )
    print("Predictions completed.")

# Main entry point
if __name__ == "__main__":
    run_pipeline()
