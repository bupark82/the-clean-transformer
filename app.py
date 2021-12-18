#
# Streamlit의 번역 app.
#
import argparse
from cleanformer.builders import InferInputsBuilder
from cleanformer.fetchers import fetch_config, fetch_tokenizer, fetch_transformer

import streamlit as st

def krtous(kors):
    parser = argparse.ArgumentParser()
    # you must provide this

    parser.add_argument("entity", type=str, help="a wandb entity to download artifacts from")
    parser.add_argument("--ver", type=str, default="overfit_small")
    parser.add_argument("--kor", type=str, default="카페인은 원래 커피에 들어있는 물질이다.")
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    # fetch a pre-trained transformer & and a pre-trained tokenizer
    transformer = fetch_transformer(config['entity'], config['ver'])
    tokenizer = fetch_tokenizer(config['entity'], config['tokenizer'])
    transformer.eval()  # otherwise, the result will be different on every run
    X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=[kors]) #[config['kor']])
    src_ids = X[0, 0, 0].tolist()  # (1, 2, 2, L) -> (L) -> list
    pred_ids = transformer.predict(X).squeeze().tolist()  # (1, L) -> (L) -> list
    pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_token_id)]  # noqa
    print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))

    return tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids)

st.title("the clean transformer app")
st.write("---")

my_text = st.text_input("문자열 입력:")
st.write("입력된 문자열  : ", my_text)
st.write("---")

my_click = st.button('Click ME!')        # Reset이 않됨???
st.write("버튼 클릭 여부: ", my_click)   
st.write("---") 

st.header("번역 결과:")
if my_click == True:
    st.write("Kor->Eng: ", krtous(my_text))




