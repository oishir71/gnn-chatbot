import os
import sys
from tqdm import tqdm
from typing import Tuple, List, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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

from text_embedder import TextEmbedder

sys.path.append(f"{os.path.dirname(__file__)}/../utils")
from ginza_nlp import GiNZANaturalLanguageProcessing


class Text2GraphConverter:
    def __init__(
        self,
        nlp_model: str = "ja_ginza_electra",
        embedder_model: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
    ) -> None:
        self.nlp = GiNZANaturalLanguageProcessing(model=nlp_model, split_mode="C")
        self.embedder = TextEmbedder(model_name=embedder_model)

    def _preprocess_for_bert_tokens(self, bert_tokens: List[str]):
        new_bert_tokens = []
        for bert_token in bert_tokens:
            if bert_token in ["[CLS]", "[SEP]", "[MASK]"]:
                continue
            new_bert_token = bert_token.replace("##", "")
            new_bert_tokens.append(new_bert_token)
        return new_bert_tokens

    def get_nlp_embedding(
        self, bert_tokens: List[str], bert_embeddings: Tensor, ginza_tokens: List[str]
    ):
        # 使用するembeddingモデルによっては、前処理で文字の変換が行われている場合があるので、逆変換する。
        bert_tokens = self._preprocess_for_bert_tokens(bert_tokens=bert_tokens)

        # bertとginzaのtokenizationの方法が異なるため、可能な限りマッチさせる必要がある。
        # トークンの開始と終了いずれもずれている場合は考慮されていない。
        ginza_embedding = {}
        i_b_start = 0
        ginza_indices = []
        for i_g in range(0, len(ginza_tokens), 1):
            ginza_text = (
                "".join(ginza_tokens[ginza_index].text for ginza_index in ginza_indices)
                + ginza_tokens[i_g].text
            )
            bert_indices = []
            for i_b in range(i_b_start, len(bert_tokens), 1):
                bert_text = (
                    "".join(bert_tokens[bert_index] for bert_index in bert_indices)
                    + bert_tokens[i_b]
                )
                # ginza_textがbert_textと一致した場合
                if len(ginza_text) == len(bert_text):
                    bert_indices.append(i_b)
                    ginza_indices = []
                    i_b_start = i_b + 1
                    break
                # ginza_textがbert_textよりも長い場合
                if len(ginza_text) > len(bert_text):
                    bert_indices.append(i_b)
                # ginza_textがbert_textよりも短い場合
                if len(ginza_text) < len(bert_text):
                    ginza_indices.append(i_g)
                    bert_indices.append(i_b)
                    break

            try:
                ginza_embedding[ginza_tokens[i_g]] = torch.mean(
                    torch.stack(
                        [bert_embeddings[bert_index] for bert_index in bert_indices]
                    ),
                    dim=0,
                )
            except:
                logger.warning(f"Could not get embedding for: {ginza_tokens[i_g]}")
                logger.warning(bert_tokens)
                logger.warning([ginza_token.text for ginza_token in ginza_tokens])
                ginza_embedding[ginza_tokens[i_g]] = (
                    self.embedder.get_default_embedding()
                )

        return ginza_embedding

    def text_2_graph(self, text: str, _class: int):
        nodes = []
        edges = []

        # ginzaのtokenizationとbertのtokenizationを一致させる必要がある
        # - ginzaの係り受け情報を利用
        # - bertの最終層の出力をembedding vectorとして利用
        bert_tokens, bert_embeddings = self.embedder.embed(text=text)
        ginza_tokens = self.nlp.get_tokens(text=text)

        ginza_embeddings = self.get_nlp_embedding(
            bert_tokens=bert_tokens,
            bert_embeddings=bert_embeddings,
            ginza_tokens=ginza_tokens,
        )

        # 文脈的に重要な意味を持つtokenのみを取得し、その他の要素に影響されないgraphを作成する。
        ginza_meaningful_token_2_order = {}
        symbols = ["NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV", "NUM"]
        for ginza_token in ginza_tokens:
            if ginza_token.pos_ in symbols:
                ginza_meaningful_token_2_order[ginza_token] = len(
                    ginza_meaningful_token_2_order
                )

        if len(ginza_meaningful_token_2_order) == 0:
            logger.warning(f"Given text has no meaningful tokens. Text: {text}")
            for ginza_token in ginza_tokens:
                ginza_meaningful_token_2_order[ginza_token] = len(
                    ginza_meaningful_token_2_order
                )

        for ginza_meaningful_token in ginza_meaningful_token_2_order:
            nodes.append(ginza_embeddings[ginza_meaningful_token])

            for child in ginza_meaningful_token.children:
                if (
                    ginza_meaningful_token in ginza_meaningful_token_2_order
                    and child in ginza_meaningful_token_2_order
                ):
                    edges.append(
                        [
                            ginza_meaningful_token_2_order[ginza_meaningful_token],
                            ginza_meaningful_token_2_order[child],
                        ]
                    )
                    edges.append(
                        [
                            ginza_meaningful_token_2_order[child],
                            ginza_meaningful_token_2_order[ginza_meaningful_token],
                        ]
                    )

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        try:
            x = torch.stack(nodes)
        except Exception as e:
            logger.info(ginza_meaningful_token_2_order)
        # y = torch.tensor([_class], dtype=torch.long)

        # self.nlp.display_dependencies(text=text)
        # self.nlp.display_meaningful_token_dependencies(text=text)

        return Data(x=x, edge_index=edge_index)

    def bulk_text_2_graph(
        self,
        texts: List[str],
        classes: List[int],
        desc: str = "Text --> Graph",
        graph_file_name: Union[
            str, None
        ] = f"{os.path.dirname(__file__)}/../graphs/graph.pth",
        forcibly_generate_graphs: bool = False,
        as_data_loader: bool = True,
        batch_size: int = 8,
    ):
        graphs = []
        if os.path.exists(graph_file_name) and not forcibly_generate_graphs:
            graphs = self.load_graphs(graph_file_name=graph_file_name)
            logger.info(f"{desc}: {len(graphs)} were loader.")
        else:
            for text, _class in tqdm(zip(texts, classes), total=len(texts), desc=desc):
                graph = self.text_2_graph(text=text, _class=_class)
                graphs.append(graph)

            self.save_graphs(graphs=graphs, graph_file_name=graph_file_name)

        return (
            DataLoader(graphs, batch_size=batch_size, shuffle=True)
            if as_data_loader
            else graphs
        )

    def load_graphs(
        self, graph_file_name: str = f"{os.path.dirname(__file__)}/../graphs/graph.pth"
    ):
        logger.info(f"Graphs are loaded from {graph_file_name}")
        graphs = torch.load(graph_file_name)
        return graphs

    def save_graphs(
        self,
        graphs: List[Data],
        graph_file_name: str = f"{os.path.dirname(__file__)}/../graphs/graph.pth",
    ):
        logger.info(f"Graphs are saved into {graph_file_name}")
        torch.save(graphs, graph_file_name)


if __name__ == "__main__":
    from text_parser import TrainValidateTestDataParser

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
    _ = converter.bulk_text_2_graph(
        texts=train_texts,
        classes=train_classes,
        desc="[ Train ] Text --> Graph",
        graph_file_name=train_graph_file_name,
        forcibly_generate_graphs=True,
        as_data_loader=True,
        batch_size=16,
    )
