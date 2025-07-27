# Load libraries
import os
import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ensure logs directory exists
os.makedirs("../logs", exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate logging by disabling propagation
logger.propagate = False

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("logs/preprocessing.log")

# Set formatter for handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


TARGET_COLUMN = "target"


# Preprocess data
def preprocess_data():
    logger.info("Starting data preprocessing")

    try:
        # Load data
        logger.info("Loading data")
        df = pd.read_csv("data/raw/iris.csv")

        # Split data into features and target
        logger.info("Splitting data into features and target")
        features = df.drop(columns=[TARGET_COLUMN])
        target = df[TARGET_COLUMN]

        # Scale features
        logger.info("Scaling features")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Split data into training and testing sets
        logger.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=0.2, random_state=42
        )

        # Save data
        logger.info("Checking if data directories exist")
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Save data
        logger.info("Saving data")
        # Convert numpy arrays to pandas DataFrames for saving
        import numpy as np
        X_train_df = pd.DataFrame(X_train, columns=features.columns)
        X_test_df = pd.DataFrame(X_test, columns=features.columns)
        y_train_df = pd.DataFrame(y_train, columns=[TARGET_COLUMN])
        y_test_df = pd.DataFrame(y_test, columns=[TARGET_COLUMN])
        
        X_train_df.to_csv("data/processed/X_train.csv", index=False)
        X_test_df.to_csv("data/processed/X_test.csv", index=False)
        y_train_df.to_csv("data/processed/y_train.csv", index=False)
        y_test_df.to_csv("data/processed/y_test.csv", index=False)

        # Save scaler
        logger.info("Saving scaler")
        joblib.dump(scaler, "models/scaler.joblib")

        # Log success
        logger.info("Data preprocessing completed successfully")
    
    except Exception as e:
        # Log error
        logger.error(f"Error preprocessing data: {e}")
        # Raise error
        raise e
    
    logger.info("Data preprocessing has ended")    


if __name__ == "__main__":
    preprocess_data()
