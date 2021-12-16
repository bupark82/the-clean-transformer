from typing import Tuple
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.nn import functional as F


class Transformer(LightningModule):
    # --- ignored --- #
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
    # --- ignored --- #

    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        # self.hparams 이라고하는 딕셔너리에 모든 하이퍼파라미터를 저장
        self.save_hyperparameters()
        # 학습을 해야하는 해야히는 레이어?: 임베딩 테이블, 인코더, 디코더, 이 3가지를 학습해야한다.
        # (|V|, H)
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, heads)
        self.decoder = Decoder()

    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        src_ids: (N, L)
        tgt_ids: (N, L)
        return hidden (N, L, H)
        """
        # --- 임베딩 벡터 불러오기 --- #
        src = self.token_embeddings(src_ids)  # (N, L) -> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)  # (N, L) -> (N, L, H)
        # --- positional encoding --- #
        # TODO: 나중에
        memory = self.encoder.forward(src)  # (N, L, H) -> (N, L, H)
        hidden = self.decoder.forward(tgt, memory)  # (N, L, H) -> (N, L, H)
        return hidden

    # 학습을 진행하기 위해선 입력 & 레이블을 인자로 받는 함수를 정의해야한다.
    # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=training_step#training-step
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> dict:
        # batch 속에 무엇이 들어있을까?
        # A: 사용자 맘입니다. 즉 제가 정의를 해야합니다.
        X, Y = batch  # (N, 2, 2, L), (N, L)
        # X = 입력
        # encoder 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        hidden = self.forward(src_ids, tgt_ids,
                              src_key_padding_mask, tgt_key_padding_mask)  # (N, L, H)
        cls = self.token_embeddings.weight  # (|V|, H)
        # 행렬 곱을 해야한다.
        logits = torch.einsum("nlh,vh->nvl", hidden, cls)  # (N, L, H) * (V, H) ->  (N, L, V=클래스) X (N, V, L)
        loss = F.cross_entropy(logits, Y)  # (N, V, d1=L), (N, d1=L) -> (N,)
        loss = loss.sum()  # (N,) -> (,)
        return {
            "loss": loss
        }

#  learn ->  issue_2
#  issue_2 -> merge -> learn
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        param X (N, 2, 2, L)
        return label_ids (N, L)
        """
        # encoder 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        for time in range(0, self.hparams['max_length'] - 1):  # 0 -> L - 2
            # ------ #
            #  ...  (N, L, H)
            hidden = self.forward(src_ids, tgt_ids,
                                  src_key_padding_mask, tgt_key_padding_mask)

            cls = self.token_embeddings.weight  # (|V|, H)
            # 행렬 곱을 해야한다.
            logits = torch.einsum("nlh,vh->nlv", hidden, cls)   # ... -> (N 0th, L 1th, V 2nd)
            # 가장 로짓값이 높은 인덱스를 바로 다음 토큰을 지정 -> greedy decoding.
            # I walked the ...(dog) (cat)
            # e.g. beam search
            ids = torch.argmax(logits, dim=2)  # (N, L, V) -> (N, L)
            # [BOS] 다음에 와야하는 단어의 아이디
            next_ids = ids[:, time]  # (N, L) -> (N,)
            # 다음 시간대의 토큰을 예측된 토큰으로 갈음
            # time = L - 1
            # time + 1 = L (존재하지 않음). 그래서 0 -> L- 1로 두면 index out of range
            tgt_ids[:, time + 1] = next_ids
            # 다음 시간대의 토큰은 더 이상 패딩 토큰이 아니므로, 마스크를 열어주기.
            tgt_key_padding_mask[:, time + 1] = 0
            # ----- #
        label_ids = tgt_ids
        return label_ids


class Encoder(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        # 최적화 해야할 가중치를 정의
        self.self_attention_layer = MultiHeadAttentionLayer(hidden_size,  heads)
        # TODO -  ffn

    def forward(self, x: torch.Tensor):
        """
        x: (N, L, H)
        return contexts (맥락이 반영된 벡터)
        """
        # 단어가 쓰인 문장에서 단어가 가지는 맥락을 임베딩 벡터에 인코딩 해준다
        contexts = self.self_attention_layer.forward(q=x, k=x, v=x)
        return contexts


class Decoder(torch.nn.Module):
    pass


class MultiHeadAttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads  # 머리가 몇개?  symbol: h
        # e.g. hidden_size = 24 head = 3
        assert self.hidden_size % self.heads == 0  # 조건을 걸자
        self.head_size = self.hidden_size // self.heads  # symbol: s
        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (N, L, H)
        k: (N, L, H)
        v: (N, L, H)
        return contexts (N, L, H)
        """
        N, L, _ = q.size()
        q = self.linear_q(q)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H) -> (N, L, H)

        # ---  머리를 쪼개는 방식으로 멀티 헤드를 만들어야 한다 --- #
        q = q.reshape(N, L, self.heads, self.head_size)   # (N, L, H) -> (N, L, heads, head_size)
        k = k.reshape(N, L, self.heads, self.head_size)   # (N, L, H) -> (N, L, heads, head_size)
        v = v.reshape(N, L, self.heads, self.head_size)   # (N, L, H) -> (N, L, heads, head_size)

        # TODO - "scaled"

        # "h" 차원에 대해서 벡터의 내적이 계산, 그렇게 h 차원은 reduce.
        # (N, L, heads, head_size) *  (N, L, heads, head_size) -> (N, heads, L, L)
        sims = torch.einsum("nqhs,nkhs->nhqk", q, k)

        # TODO - masking (auto-regressive)
        # key 차원에 대해서 정규화를 했기
        #  (N, heads, L, L)
        attentions = torch.softmax(sims, dim=3)  # (N, q의 길이 L, k의 갈이 L <- 마지막 차원을 정규화)
        # "k"차원에 있는 가중치 (확률분포로) 가중평균을 구해야한다
        contexts = torch.einsum("nhqk,nkhs->nqhs", attentions, v)  # (N, L, L) * (N, L, H) -> (N, L, heads, head_size)
        # concat
        # (N, L, heads, head_size) -> (N, L, H = heads *head_size = heads * (H / heads))
        contexts = contexts.reshape(N, L, self.hidden_size)
        # 단순히 이어붙인 여러 의존관계를 join
        contexts = self.linear_o(contexts)  # (N, L, H) -> (N, L, H)
        return contexts


