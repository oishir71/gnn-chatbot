import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

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


class GraphConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(
        self, number_of_node_features: int, hidden_channels: int, number_of_classes: int
    ) -> None:
        super().__init__()
        logger.info(f"Number of node features: {number_of_node_features}")
        logger.info(f"Number of hidden channels: {hidden_channels}")
        logger.info(f"Number of classes: {number_of_classes}")

        self.gcn_conv1 = GCNConv(number_of_node_features, hidden_channels)
        self.gcn_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, number_of_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn_conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from text_parser import TrainValidateTestDataParser
    from t2g_converter import Text2GraphConverter

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

    converter = Text2GraphConverter()
    train_graph_file_name = f"{os.path.dirname(__file__)}/../graphs/train_graphs.pth"
    train_graph_loader = converter.bulk_text_2_graph(
        texts=train_texts,
        classes=train_classes,
        desc="[ Train ] Text --> Graph",
        graph_file_name=train_graph_file_name,
        forcibly_generate_graphs=True,
        as_data_loader=True,
        batch_size=16,
    )

    model = GraphConvolutionalNeuralNetwork(
        number_of_node_features=train_graph_loader.dataset[0].num_node_features,
        hidden_channels=16,
        number_of_classes=data_parser.get_number_of_unique_classes(),
    )
