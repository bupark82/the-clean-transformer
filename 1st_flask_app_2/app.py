from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import argparse
from cleanformer.builders import InferInputsBuilder
from cleanformer.fetchers import fetch_config, fetch_tokenizer, fetch_transformer

app = Flask(__name__)

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
    # print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))
    kr = tokenizer.decode(ids=src_ids)
    us = tokenizer.decode(ids=pred_ids)

    return kr, us

class TranslateForm(Form):
    translate = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = TranslateForm(request.form)
    return render_template('first_app.html', form=form)

@app.route('/translate', methods=['POST'])
def translate():
    form = TranslateForm(request.form)
    if request.method == 'POST' and form.validate():
        kors = request.form['translate']
        kr, us = krtous(kors)
        return render_template('translate.html', kor=kr, eng=us)
    return render_template('first_app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)