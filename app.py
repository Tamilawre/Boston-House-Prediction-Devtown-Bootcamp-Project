from flask import Flask, render_template, request
import os
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['CRIM']),
        float(request.form['ZN']),
        float(request.form['INDUS']),
        float(request.form['CHAS']),
        float(request.form['NOX']),
        float(request.form['RM']),
        float(request.form['AGE']),
        float(request.form['DIS']),
        float(request.form['RAD']),
        float(request.form['TAX']),
        float(request.form['PTRATIO']),
        float(request.form['B']),
        float(request.form['LSTAT']),
    ]
    np_features = np.array([features])
    predicted = model.predict(np_features)
    output = round(predicted[0], 2)

    return render_template('index.html', prediction=f'The predicted price is {output}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))


