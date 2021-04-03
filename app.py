import pandas as pd
import pickle
from flask import Flask, jsonify, request

model = None

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    data.update((x, [y]) for x, y in data.items())
    print(data)

    data_df = pd.DataFrame.from_dict(data)

    result = model.predict(data_df)

    output = {'results':int(result[0])}

    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port = 5000, debug=True)
