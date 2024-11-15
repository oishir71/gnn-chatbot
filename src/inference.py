import os
import pprint
import torch

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

from text_parser import TrainValidateTestDataParser
from t2g_converter import Text2GraphConverter
from graph_convolutional_neural_network import GraphConvolutionalNeuralNetwork
from model_runner import ModelRunner

nlp_model = "ja_ginza_electra"
embedder_model = "cl-tohoku/bert-base-japanese-whole-word-masking"

if __name__ == "__main__":
    test_texts = ["衛星干渉事前計算の担当者を変更するにはどうすればいいですか？"]
    test_texts = ["OPEN-UI 衛星干渉事前計算"]
    dummy_classes = [0 for _ in range(len(test_texts))]

    ###############################################
    # データのロード
    ###############################################
    data_parser = TrainValidateTestDataParser(
        train_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-008/gen_gpt4_upto25.csv"
        ],
    )
    train_texts, train_classes = data_parser.load_train_data()
    number_of_unique_classes = data_parser.get_number_of_unique_classes()

    ###############################################
    # データの変換
    ###############################################
    converter = Text2GraphConverter(nlp_model=nlp_model, embedder_model=embedder_model)

    train_graph_file_name = f"{os.path.dirname(__file__)}/../graphs/train_graphs.pth"
    train_graph_loader = converter.bulk_text_2_graph(
        texts=train_texts,
        classes=train_classes,
        desc="[ Training ] Text -> Graph",
        graph_file_name=train_graph_file_name,
        forcibly_generate_graphs=False,
        as_data_loader=True,
        batch_size=64,
    )

    test_graph_loader = converter.bulk_text_2_graph(
        texts=test_texts,
        classes=dummy_classes,
        forcibly_generate_graphs=True,
        as_data_loader=True,
    )

    ###############################################
    # GNNの定義
    ###############################################
    gcn = GraphConvolutionalNeuralNetwork(
        number_of_node_features=train_graph_loader.dataset[0].num_node_features,
        hidden_channels=16,
        number_of_classes=number_of_unique_classes,
    )

    ###############################################
    # GNNのテスト
    ###############################################
    weight_file_name = f"{os.path.dirname(__file__)}/../weights/model_weight.pth"
    runner = ModelRunner(model=gcn)
    runner.load_model_weights(weight_file_name=weight_file_name)

    logger.info(test_texts)
    for test_graph in test_graph_loader:
        predicted_probabilities = runner.inference(data=test_graph)

        topk_values, topk_indices = torch.topk(predicted_probabilities, 5)
        for values, indices in zip(
            topk_values.cpu().tolist(), topk_indices.cpu().tolist()
        ):
            for i_index, (value, index) in enumerate(zip(values, indices)):
                logger.info(
                    f"Top: {i_index} - Class: {index}, Probability: {value * 100:.2f} %"
                )
                texts = data_parser.get_texts_by_class(_class=index)
                for line in pprint.pformat(texts, width=150).split("\n"):
                    logger.info(line)
