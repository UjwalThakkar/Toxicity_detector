from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys

from test import toxicity_classifier

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        model = toxicity_classifier()
        results = model.predict(text)
        df = pd.DataFrame.from_dict(results, orient='index')
        return df.to_json()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/save', methods=['POST'])
def  save():
    return jsonify({'message': 'called the save function'}), 400

if __name__ == '__main__':
    app.run(debug=True)