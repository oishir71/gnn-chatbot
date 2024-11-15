import os

from text_parser import TrainValidateTestDataParser
from t2g_converter import Text2GraphConverter
from graph_convolutional_neural_network import GraphConvolutionalNeuralNetwork
from model_runner import ModelRunner

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

import argparse

parser = argparse.ArgumentParser(
    description="Graph neural network for training a chatbot"
)
parser.add_argument(
    "--nlp_model",
    dest="nlp_model",
    help="Model name for nlp",
    default="ja_ginza_electra",
    type=str,
)
parser.add_argument(
    "--embedder_model",
    dest="embedder_model",
    help="Model name for embedding",
    default="cl-tohoku/bert-base-japanese-whole-word-masking",
    type=str,
)
parser.add_argument(
    "--forcibly_generate_graphs",
    dest="forcibly_generate_graphs",
    help="Forcibly generate graphs",
    action="store_true",
)
parser.add_argument(
    "--train",
    dest="train",
    help="Run training for graph neural network",
    action="store_true",
)
parser.add_argument(
    "--validate",
    dest="validate",
    help="Run validate for graph neural network",
    action="store_true",
)
parser.add_argument(
    "--test",
    dest="test",
    help="Run testing for graph neural network",
    action="store_true",
)
parser.add_argument(
    "--hidden_channels",
    dest="hidden_channels",
    action="store",
    default=16,
    type=int,
)
parser.add_argument(
    "--epochs",
    dest="epochs",
    help="The number of epochs",
    action="store",
    default=2000,
    type=int,
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    help="Batch size for whole data",
    action="store",
    default=8,
    type=int,
)

args = parser.parse_args()

if __name__ == "__main__":
    ###############################################
    # データのロード
    ###############################################
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

    ###############################################
    # データのロード
    ###############################################
    converter = Text2GraphConverter(
        nlp_model=args.nlp_model,
        embedder_model=args.embedder_model,
    )

    train_graph_file_name = f"{os.path.dirname(__file__)}/../graphs/train_graphs.pth"
    train_graph_loader = converter.bulk_text_2_graph(
        texts=train_texts,
        classes=train_classes,
        desc="[ Training ] Text -> Graph",
        graph_file_name=train_graph_file_name,
        forcibly_generate_graphs=args.forcibly_generate_graphs,
        as_data_loader=True,
        batch_size=args.batch_size,
    )

    validate_graph_file_name = (
        f"{os.path.dirname(__file__)}/../graphs/validate_graphs.pth"
    )
    validate_graph_loader = converter.bulk_text_2_graph(
        texts=validate_texts,
        classes=validate_classes,
        desc="[ validate ] Text -> Graph",
        graph_file_name=validate_graph_file_name,
        forcibly_generate_graphs=args.forcibly_generate_graphs,
        as_data_loader=True,
        batch_size=args.batch_size,
    )

    test_graph_file_name = f"{os.path.dirname(__file__)}/../graphs/test_graphs.pth"
    test_graph_loader = converter.bulk_text_2_graph(
        texts=test_texts,
        classes=test_classes,
        desc="[ Test ] Text -> Graph",
        graph_file_name=test_graph_file_name,
        forcibly_generate_graphs=args.forcibly_generate_graphs,
        as_data_loader=True,
        batch_size=args.batch_size,
    )

    ###############################################
    # GNNの定義
    ###############################################
    gcn = GraphConvolutionalNeuralNetwork(
        number_of_node_features=train_graph_loader.dataset[0].num_node_features,
        hidden_channels=args.hidden_channels,
        number_of_classes=number_of_unique_classes,
    )

    ###############################################
    # GNNのトレーニング
    ###############################################
    history_file_name = f"{os.path.dirname(__file__)}/../histories/histories.csv"
    weight_file_name = f"{os.path.dirname(__file__)}/../weights/model_weight.pth"
    runner = ModelRunner(model=gcn, learning_rate=1e-3, weight_decay=1e-2)
    runner.execute(
        epochs=args.epochs,
        train_data_loader=train_graph_loader,
        validate_data_loader=validate_graph_loader,
        test_data_loader=test_graph_loader,
        do_train=args.train,
        do_validate=args.validate,
        do_test=args.test,
        history_file_name=history_file_name,
        weight_file_name=weight_file_name,
    )
