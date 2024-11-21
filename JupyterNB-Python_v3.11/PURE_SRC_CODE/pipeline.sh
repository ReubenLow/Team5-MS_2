#!/bin/bash

# Display the CLI menu
while true; do
    echo -e "\nPipeline CLI Menu:"
    echo "c. Clean data"
    echo "e. Perform EDA on training data"
    echo "m. Train model"
    echo "p. Predict model"
    echo "4. Exit"
    read -p "Enter your choice (t/c/e/m/p/4): " choice

    if [[ "$choice" == "4" ]]; then
        echo "Exiting the pipeline..."
        exit 0
    elif [[ "$choice" == "c" ]]; then
        echo "Listing files in 'training_data' folder, please select one to clean:"
        # python3 clean_data.py --input_folder ../OTH_DATA/training_data --output_folder ../OTH_DATA/cleaned_data
        python3 clean_data.py --config_file ../SCRIPTS_CFG/config.txt
    elif [[ "$choice" == "e" ]]; then
        echo "Performing EDA on training data..."
        # python3 perform_eda.py --input_folder ../OTH_DATA/cleaned_data --output ../EDA_DATA
        python3 perform_eda.py --config_file ../SCRIPTS_CFG/config.txt
    elif [[ "$choice" == "m" ]]; then
        echo "Training model..."
        # python3 model_training.py --input_folder ../OTH_DATA/cleaned_data --model_output ../ML_DATA/model_outputs
        python3 model_training.py --config_file ../SCRIPTS_CFG/config.txt
    elif [[ "$choice" == "p" ]]; then
        echo "Predicting model..."
        # python3 perform_prediction.py --input_folder ../OTH_DATA/cleaned_data --model_folder ../ML_DATA/model_outputs --output_folder ../ML_DATA/predict_outputs
        python3 perform_prediction.py --config_file ../SCRIPTS_CFG/config.txt
    else
        echo "Invalid choice. Please enter 1, 2, 3, or 4."
    fi
done

#!/bin/bash

# # For exe
# CONFIG_PATH="../SCRIPTS_CFG/config.txt"
# EXECUTABLES_PATH="../EXECUTABLES"
# # Pipeline CLI Menu
# while true; do
#     echo "Pipeline CLI Menu:"
#     echo "c. Clean data"
#     echo "e. Perform EDA on training data"
#     echo "m. Train model"
#     echo "p. Predict model"
#     echo "4. Exit"
#     read -p "Enter your choice (t/c/e/m/p/4): " choice

#     case $choice in
#         c) 
#             echo "Cleaning data..."
#             "$EXECUTABLES_PATH/clean_data.exe" --config_file "$CONFIG_PATH"
#             ;;
#         e)
#             echo "Performing EDA on data..."
#             "$EXECUTABLES_PATH/perform_eda.exe" --config_file "$CONFIG_PATH"
#             ;;
#         m)
#             echo "Training model..."
#             "$EXECUTABLES_PATH/model_training.exe" --config_file "$CONFIG_PATH"
#             ;;
#         p)
#             echo "Predicting model..."
#             "$EXECUTABLES_PATH/perform_prediction.exe" --config_file "$CONFIG_PATH"
#             ;;
#         4)
#             echo "Exiting..."
#             break
#             ;;
#         *)
#             echo "Invalid choice. Please enter a valid option."
#             ;;
#     esac
# done

