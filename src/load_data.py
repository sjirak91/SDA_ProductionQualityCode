import os
import csv
from importlib import resources

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import Bunch, _convert_data_dataframe

DATA_MODULE = "sklearn.datasets.data"
RANDOM_SEED = 0
TEST_SPLIT = 0.20

from matplotlib_venn import venn2
from matplotlib import pyplot as plt

# import matplotlib.pyplot as plt

# venn2(subsets=(3, 2, 1))
# plt.show()

# plus zmiana środowiska
# venn2([{1, 2, 3}, {2, 3, 6, 7, 8}])
# plt.show()


# to optimize imports
# pip install isort
# isort src/load_data.py

# Issue with the following code
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_base.py#L331


def load_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    # descr_module=DESCR_MODULE,
    encoding="utf-8",
):
    """Loads `data_file_name` from `data_module with `importlib.resources`."""
    data_path = resources.files(data_module) / data_file_name
    assert os.path.exists(data_path), f"The file {data_file_name} does not exist"

    with data_path.open("r", encoding=encoding) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    if descr_file_name is None:
        return data, target, target_names
    # else:
    #     assert descr_module is not None
    #     descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
    #     return data, target, target_names, descr


import json


def get_column_names(file="iris"):
    json_path = f"../data/{file}_data_columns.json"
    # json_path = f"data/{file}_data_columns_INCORRECT.json"
    # assert os.path.exists(json_path)

    try:
        with open(json_path, "r") as file:
            data = json.load(file)
            columns = data["feature_names"], data["target_columns"]
    except FileNotFoundError:
        columns = None, None

    return columns


def load_iris(*, return_X_y=False, as_frame=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    [np.str_('setosa'), np.str_('versicolor'), np.str_('virginica')]
    """
    data_file_name = "iris.csv"
    data, target, target_names = load_csv_data(data_file_name=data_file_name)

    # feature_names = [
    #     "sepal length (cm)",
    #     "sepal width (cm)",
    #     "petal length (cm)",
    #     # "petal width (cm)",
    # ]

    frame = None
    # target_columns = [
    #     "target",
    # ]
    feature_names, target_columns = get_column_names()

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_iris", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        # DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )


def get_train_test_data(test_size=TEST_SPLIT):
    iris = load_iris()

    try:
        X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    except ValueError:
        X = pd.DataFrame(iris["data"])
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


X_train, _, y_train, _ = get_train_test_data()
print(X_train)
