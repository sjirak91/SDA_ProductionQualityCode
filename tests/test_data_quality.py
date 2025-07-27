import os
import pandas as pd

def test_file_exists(csv_path = "data/raw/iris.csv"):
    assert os.path.exists(csv_path), "File does not exist"

def test_correct_shape(csv_path = "data/raw/iris.csv"):
    df = pd.read_csv(csv_path)
    assert df.shape == (150, 5)

def test_file_has_correct_columns(csv_path = "data/raw/iris.csv"):
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"}

def test_file_has_correct_data_types(csv_path = "data/raw/iris.csv"):
    df = pd.read_csv(csv_path)
    assert all(df.dtypes[:-1] == "float64")
    assert df["target"].dtype == "int64"

def test_no_missing_values(csv_path = "data/raw/iris.csv"):
    df = pd.read_csv(csv_path)
    assert df.isnull().sum().sum() == 0
