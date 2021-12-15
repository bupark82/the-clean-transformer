from typing import Tuple, Dict

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        # TODO: implement transformer
        # raise NotImplementedError
        # 학습을 해야하는 레이어? 임베딩 테이블, 인코더, 디코더, 이 3가지를 학습해야한다.
        self.torken_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.classifier = torch.nn.Linear()

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # --- 임베딩 벡터 불러오기 --- #
        src = self.token_embeddings(src_ids)   # (N, L) -> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)   # (N, L) -> (N, L, H)
        # --- positional encoding --- #
        # TODO : 나중에
        memory = self.encoder.forward(src)
        hidden = self.decoder.forward(tgt, memory)
        return hidden

    #학습을 진행하기 위해서는 입력 & 레이블을 인자로 받는 함수를 정의해야한다.
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> dict:
        X, Y = batch # (N, 2, 2, L), (N, L)
        # X = 입력
        # encoder 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # Y = 레이블
        # decoder 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        hidden = self.forward(src_ids, tgt_ids,
                              src_key_padding_mask, tgt_key_padding_mask)   # (N, L, H)
        cls = self.token_embeddings.weight  # (|V|, H)
        logits = torch.einsum("nlh,vh->nvl", hidden, cls)  # (N, L, H) * (V, H) -> (N, L, V=클래스) X (N, V, L)
        loss = F.cross_entropy(logits, Y)   # (N, V, d1=L), (N, d1=L)
        loss = loss.sum()   # (N,) -> (,)
        return {
            "loss": loss
        }

class Encoder(torch.nn.Module):
    # raise NotImplementedError
    pass

class Decoder(torch.nn.Module):
    # raise NotImplementedError
    pass
