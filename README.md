# AI Engineer Production Quality Code

A comprehensive machine learning project demonstrating production-quality code practices using the classic Iris dataset. This project showcases best practices for writing maintainable, testable, and efficient machine learning code.

## 🎯 Project Overview

This project demonstrates how to convert experimental machine learning code into production-ready, maintainable code with a focus on:

- **Modular Architecture**: Well-organized code structure with clear separation of concerns
- **Exception Handling**: Robust error handling and logging
- **Unit Testing**: Comprehensive test coverage for data quality and functionality
- **Performance Optimization**: Efficient data processing and model training
- **Documentation**: Clear code documentation and project structure

## 📁 Project Structure

```
AIEngineerProductionQualityCode/
├── data/                   # Data storage
│   └── iris.csv           # Iris dataset
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb         # Exploratory Data Analysis
│   └── ProductionQualityCode.ipynb  # Production code examples
├── src/                   # Source code
│   ├── load_data.py      # Data loading utilities
│   ├── train.py          # Model training with logging
│   ├── utils.py          # Utility functions
│   └── tests.py          # Unit tests
├── tests/                 # Test files
│   └── test_data_quality.py  # Data quality tests
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIEngineerProductionQualityCode
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Load and prepare the Iris dataset**
   ```bash
   python src/load_data.py
   ```

2. **Run the training script with logging**
   ```bash
   python src/train.py
   ```

3. **Run unit tests**
   ```bash
   python -m pytest tests/
   ```

4. **Run data quality tests**
   ```bash
   python tests/test_data_quality.py
   ```

## 📊 Dataset

The project uses the classic **Iris dataset** which contains:
- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width (all in cm)
- **3 classes**: Setosa, Versicolor, Virginica

## 🔧 Key Features

### Data Loading (`src/load_data.py`)
- Automated dataset loading from scikit-learn
- Data preprocessing and CSV export
- Column name validation

### Training Pipeline (`src/train.py`)
- Comprehensive logging setup (file + console)
- Exception handling with custom error classes
- Train-test data splitting

### Utility Functions (`src/utils.py`)
- Data conversion utilities
- Pandas support checking
- Bunch class for attribute access

### Testing (`src/tests.py` & `tests/test_data_quality.py`)
- Unit tests for data loading functions
- Data quality validation tests
- Shape, column, and data type verification

## 🧪 Testing

The project includes comprehensive testing:

### Unit Tests
```bash
python src/tests.py
```

### Data Quality Tests
```bash
python tests/test_data_quality.py
```

### All Tests with pytest
```bash
python -m pytest tests/ -v
```

## 📈 Performance Monitoring

The training script includes:
- **Logging**: Both file and console output
- **Error Tracking**: Custom exception handling
- **Performance Metrics**: Execution time monitoring

## 🛠️ Dependencies

Key dependencies include:
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **pytest**: Testing framework

See `requirements.txt` for the complete list.

## 📚 Learning Resources

The project includes Jupyter notebooks demonstrating:
- **EDA.ipynb**: Exploratory Data Analysis
- **ProductionQualityCode.ipynb**: Production code examples and best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn team for the Iris dataset
- The machine learning community for best practices
- SDA (School of Data Analysis) for educational content

---

**Note**: This project is designed for educational purposes and demonstrates production-quality code practices in machine learning projects.