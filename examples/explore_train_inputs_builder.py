

from cleanformer.builders import TrainInputsBuilder, LabelsBuilder
from cleanformer.fetchers import fetch_tokenizer

# 난 널 사랑해 -> 난, 널, 사랑해
tokenizer = fetch_tokenizer("eubinecto", ver="wp")   # WordPiece
# 길이가 서로 다른 문장의 길이를 통일하기 위해서, 필요.
max_length = 10

inputs_builder = TrainInputsBuilder(tokenizer, max_length)
labels_builder = LabelsBuilder(tokenizer, max_length)

# 트랜스포머 - 번역을 하기 위해서 만들어진 모델
kors = ["난 널 사랑해"]
engs = ["I love you"]

X = inputs_builder(srcs=kors, tgts=engs)    # (N=1, 2=(src/tgt), 2(ids/mask), L)
Y = labels_builder(tgts=engs)   # (N=1, L)

src_ids = X[:, 0, 0]    # (N, 2, 2, L) -> (N, L)
tgt_ids = X[:, 1, 0]    # (N, 2, 2, L) -> (N, L)
src_key_padding_mask = X[:, 0, 1]
tgt_key_padding_mask = X[:, 1, 1]
print(X)
print(Y)
print(src_ids.shape)
print(tgt_ids.shape)
print(src_ids)
print(tgt_ids)
# singleton dimension 제거
src_ids = src_ids.squeeze()   # (1, L) -> (L)
src_key_padding_mask = src_key_padding_mask.squeeze()
tgt_ids = tgt_ids.squeeze()   # (1, L) -> (L)
tgt_key_padding_mask= tgt_key_padding_mask.squeeze()
Y = Y.squeeze()   # (1, L) -> (L)
print(src_ids)
print(tgt_ids)
# encoder의 입력
print([tokenizer.id_to_token(src_id) for src_id in src_ids.tolist()])
print(src_key_padding_mask)
# decoder의 입력
print([tokenizer.id_to_token(tgt_id) for tgt_id in tgt_ids.tolist()])
print(tgt_key_padding_mask)
print([tokenizer.id_to_token(y_id) for y_id in Y.tolist()])

