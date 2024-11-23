from tests import MyTestCase
from load_data import get_train_test_data


X_train, _, y_train, _ = get_train_test_data()
print(X_train.head())


import logging


class InvalidConfigurationError(Exception):
    pass


logging.basicConfig(
    # filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)

try:
    raise InvalidConfigurationError("Missing configuration file!")
except InvalidConfigurationError as e:
    logging.error(f"Error occurred: {e}")
    print("An error occurred. Check the log for details.")

logging.warning("Script was finished")
logging.info("Script was finished")
