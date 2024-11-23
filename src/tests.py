import unittest
from load_data import get_column_names


IRIS_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

IRIS_TARGET = ["target"]


class MyTestCase(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1 + 1, 2)

    def test_example1(self):
        self.assertTrue("hello".isalpha())

    def test_get_column_names_incorrect_name(self):
        output = get_column_names(file="iris159")
        self.assertIsNone(output[0])
        self.assertIsNone(output[1])

    def test_get_column_names_iris(self):
        output = get_column_names(file="iris")
        self.assertEqual(output[0], IRIS_COLUMNS)
        self.assertEqual(output[1], IRIS_TARGET)


if __name__ == "__main__":
    unittest.main()
