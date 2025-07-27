# Load libraries
import os
import pandas as pd
from sklearn.datasets import load_iris

def load_iris_data():
    # Create iris instance
    iris = load_iris()

    # Create a dataframe with the iris data
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    os.makedirs("data", exist_ok=True)

    csv_path = "data/iris.csv"
    df.to_csv(csv_path, index=False)

    print(f"Data saved to {csv_path}")


if __name__ == "__main__":
    load_iris_data()
