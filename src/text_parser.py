import os
import csv
import re
from typing import List

import jaconv

# Logging
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter(
    "%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s"
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)


class TrainValidateTestDataParser:
    def __init__(
        self,
        train_data_file_paths: List[str] = [
            f"{os.path.dirname(__file__)}/train_data_file.csv"
        ],
        validate_data_file_paths: List[str] = [
            f"{os.path.dirname(__file__)}/validate_data_file.csv"
        ],
        test_data_file_paths: List[str] = [
            f"{os.path.dirname(__file__)}/test_data_file.csv"
        ],
    ):
        self.train_data_file_paths = train_data_file_paths
        self.validate_data_file_paths = validate_data_file_paths
        self.test_data_file_paths = test_data_file_paths

        self.label_2_class = {}
        self.class_2_label = {}
        self.number_of_classes = 0

    def get_class_by_label(self, label: str):
        if not label in self.label_2_class:
            _class = len(self.label_2_class)
            self.label_2_class[label] = _class
        return self.label_2_class[label]

    def get_label_by_class(self, _class: int):
        if not _class in self.class_2_label:
            self.class_2_label = {
                value: key for key, value in self.label_2_class.item()
            }
        try:
            label = self.class_2_label[_class]
            return label
        except Exception as e:
            logger.error(f"{_class} was not found. {e}")
            raise

    def _find_column_names(self, row, possible_column_names: List[str]):
        """
        Check if one of the possible column names can be found in column names.
        Add proper column name candidate into the possible_column_names argument.
        """
        for name in possible_column_names:
            if name in row:
                return name
        logger.error(
            f"No available column name was found. [{', '.join(possible_column_names)}]"
        )
        raise

    def _text_cosmetics(self, text: str) -> str:
        text = jaconv.h2z(text)
        text = re.sub(r"\s+", "", text)
        return text

    def _load_data(self, file_paths: List[str]):
        texts_already_taken_into_account = []
        texts, classes = [], []

        text_column_name = None
        label_column_name = None
        text_column_name_candidates = ["question", "\ufeffquestion"]
        label_column_name_candidates = ["correct_answer", "\ufeffcorrect_answer"]
        for i_file_path, file_path in enumerate(file_paths):
            logger.info(
                f"[ {i_file_path + 1} / {len(file_paths)} ] load data from {file_path}"
            )
            with open(file_path, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if text_column_name is None:
                        text_column_name = self._find_column_names(
                            row=row, possible_column_names=text_column_name_candidates
                        )
                    if label_column_name is None:
                        label_column_name = self._find_column_names(
                            row=row, possible_column_names=label_column_name_candidates
                        )
                    if not row[text_column_name] in texts_already_taken_into_account:
                        texts_already_taken_into_account.append(row[text_column_name])
                        texts.append(self._text_cosmetics(text=row[text_column_name]))
                        classes.append(
                            self.get_class_by_label(label=row[label_column_name])
                        )

        logger.info(f"{len(texts)} data was loader.")
        return texts, classes

    def load_train_data(self):
        logger.info("Load train data")
        return self._load_data(file_paths=self.train_data_file_paths)

    def load_validate_data(self):
        logger.info("Load validate data")
        return self._load_data(file_paths=self.validate_data_file_paths)

    def load_test_data(self):
        logger.info("Load test data")
        return self._load_data(file_paths=self.test_data_file_paths)

    def get_number_of_unique_classes(self):
        return len(self.label_2_class)


if __name__ == "__main__":
    data_parser = TrainValidateTestDataParser(
        train_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-008/gen_gpt4_upto25.csv"
        ],
        validate_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto25.csv"
        ],
        test_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/train/train.csv"
        ],
    )
    train_texts, train_classes = data_parser.load_train_data()
    validate_texts, validate_classes = data_parser.load_validate_data()
    test_texts, test_classes = data_parser.load_test_data()
    number_of_unique_classes = data_parser.get_number_of_unique_classes()
    logger.info(f"The number of unique classes: {number_of_unique_classes}")
