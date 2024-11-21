import os
import sys

def display_menu():
    print("\nPipeline CLI Menu:")
    print("c. Clean data")
    print("e. Perform EDA on training data")
    print("m. Train model")
    print("p. Predict model")
    print("4. Exit")

def main():
    script_folder = os.path.join("..", "PURE_SRC_CODE")
    config_file = os.path.join("..", "SCRIPTS_CFG", "config.txt")

    while True:
        display_menu()
        choice = input("Enter your choice (c/e/m/p/4): ").strip()

        if choice == "4":
            print("Exiting the pipeline...")
            sys.exit(0)
        elif choice == "c":
            print("Listing files in 'training_data' folder, please select one to clean:")
            # Clean data command
            os.system(f"python {os.path.join(script_folder, 'clean_data.py')} --config_file {config_file}")
        elif choice == "e":
            print("Performing EDA on training data...")
            # Perform EDA command
            os.system(f"python {os.path.join(script_folder, 'perform_eda.py')} --config_file {config_file}")
        elif choice == "m":
            print("Training model...")
            # Train model command
            os.system(f"python {os.path.join(script_folder, 'model_training.py')} --config_file {config_file}")
        elif choice == "p":
            print("Predicting model...")
            # Perform prediction command
            os.system(f"python {os.path.join(script_folder, 'perform_prediction.py')} --config_file {config_file}")
        else:
            print("Invalid choice. Please enter c, e, m, p, or 4.")

if __name__ == "__main__":
    main()
