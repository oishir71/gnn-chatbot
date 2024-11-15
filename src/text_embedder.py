import os
from typing import Tuple, List

import torch
from torch import Tensor
from transformers import BertJapaneseTokenizer, BertModel

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


class TextEmbedder:
    def __init__(
        self, model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking"
    ) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)

    def get_default_embedding(self) -> Tensor:
        return torch.zero(self.model.config.hidden_size).to(self.device)

    def embed(self, text: str) -> Tuple[List[str], Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )
        embeddings = last_hidden_states.squeeze()

        return tokens, embeddings


if __name__ == "__main__":
    text = "衛星干渉計算のマスタ情報の設定担当者を変えたいのですが、どこを参照すれば良いですか？"
    embedder = TextEmbedder(
        model_name="cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    tokens, embeddings = embedder.embed(text=text)
    print(tokens)
    print(tokens[0])
    print(embeddings[0])
