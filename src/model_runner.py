import os
import csv
from typing import List
import torch
import torch.nn.functional as F

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


class ModelRunner:
    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")

    def train(self, data_loader):
        number_of_corrects = 0
        total_loss = 0

        self.model.train()
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            predict_probabilities = self.model(data)
            _, predict_y = predict_probabilities.max(dim=1)
            number_of_corrects += int((predict_y == data.y).sum().item())
            loss = self.criterion(predict_probabilities, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return number_of_corrects / len(data_loader.dataset), total_loss

    def validate(self, data_loader):
        number_of_corrects = 0
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                predict_probabilities = self.model(data)
                _, predict_y = predict_probabilities.max(dim=1)
                number_of_corrects += int((predict_y == data.y).sum().item())
                loss = self.criterion(predict_probabilities, data.y)
                total_loss += loss.item()

        return number_of_corrects / len(data_loader.dataset), total_loss

    def test(self, data_loader):
        number_of_corrects = 0

        self.model.eval()
        for data in data_loader:
            data = data.to(self.device)
            predict_probabilities = self.model(data)
            _, predict_y = predict_probabilities.max(dim=1)
            number_of_corrects += int((predict_y == data.y).sum().item())

        return number_of_corrects / len(data_loader.dataset)

    def inference(self, data):
        self.model.eval()
        data = data.to(self.device)
        predicted_probabilities = self.model(data)
        predicted_probabilities = F.softmax(predicted_probabilities, dim=1)
        return predicted_probabilities

    def save_history_as_csv(
        self,
        train_accuracies: List[float],
        train_losses: List[float],
        validate_accuracies: List[float],
        validate_losses: List[float],
        csv_file_name: str = f"{os.path.dirname(__file__)}/../histories/histories.csv",
    ):
        with open(csv_file_name, mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_accuracy",
                    "train_loss",
                    "validate_accuracy",
                    "validate_loss",
                ]
            )

            for epoch, (
                train_accuracy,
                train_loss,
                validate_accuracy,
                validate_loss,
            ) in enumerate(
                zip(
                    train_accuracies,
                    train_losses,
                    validate_accuracies,
                    validate_losses,
                )
            ):
                writer.writerow(
                    [
                        epoch + 1,
                        train_accuracy,
                        train_loss,
                        validate_accuracy,
                        validate_loss,
                    ]
                )

    def save_model_weights(
        self,
        weight_file_name: str = f"{os.path.dirname(__file__)}/../weight/model_weight.pth",
    ):
        logger.info(f"Model weights are saved into {weight_file_name}")
        torch.save(self.model.state_dict(), weight_file_name)

    def load_model_weights(
        self,
        weight_file_name: str = f"{os.path.dirname(__file__)}/../weight/model_weight.pth",
    ):
        logger.info(f"Model weights are loaded from {weight_file_name}")
        self.model.load_state_dict(torch.load(weight_file_name))

    def execute(
        self,
        epochs,
        train_data_loader,
        validate_data_loader,
        test_data_loader,
        do_train: bool = True,
        do_validate: bool = True,
        do_test: bool = True,
        history_file_name: str = f"{os.path.dirname(__file__)}/../histories/histories.csv",
        weight_file_name: str = f"{os.path.dirname(__file__)}/../weight/model_weight.pth",
    ):
        train_accuracies = []
        train_losses = []
        validate_accuracies = []
        validate_losses = []

        for epoch in range(epochs):
            if do_train:
                train_accuracy, train_loss = self.train(data_loader=train_data_loader)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)
            if do_validate:
                validate_accuracy, validate_loss = self.validate(
                    data_loader=validate_data_loader
                )
                validate_accuracies.append(validate_accuracy)
                validate_losses.append(validate_loss)
            if do_train and do_validate:
                logger.info(
                    f"Epoch: {epoch} --> Train Acc: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {validate_accuracy:.4f}, Val Loss: {validate_loss:.4f}"
                )

                self.save_history_as_csv(
                    train_accuracies=train_accuracies,
                    train_losses=train_losses,
                    validate_accuracies=validate_accuracies,
                    validate_losses=validate_losses,
                    csv_file_name=history_file_name,
                )

        if do_train:
            self.save_model_weights(weight_file_name=weight_file_name)

        if do_test:
            if not do_train:
                self.load_model_weights(weight_file_name=weight_file_name)
            test_accuracy = self.test(data_loader=test_data_loader)
            logger.info(f"Accuracy for the test data: {test_accuracy:.4f}")


if __name__ == "__main__":
    runner = ModelRunner()
    runner.execute()
