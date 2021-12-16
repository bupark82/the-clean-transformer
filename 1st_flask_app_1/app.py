from flask import Flask, render_template
from cleanformer.datamodules import Kor2EngSmallDataModule
from cleanformer.fetchers import fetch_config

app = Flask(__name__)

@app.route('/')
def index():
    ver = "overfit_small"
    config = fetch_config()['train'][ver]
    config.update({'num_workers': 2})
    datamodule = Kor2EngSmallDataModule(config, None)  # noqa
    datamodule.prepare_data()
    # --- explore some data --- #
    for pair in datamodule.kor2eng_train:
        #print(pair)
        pair

    return render_template('first_app.html', pair=pair)

if __name__ == '__main__':
    app.run(debug=True)