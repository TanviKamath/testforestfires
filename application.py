from flask import Flask, request, jsonify,render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

#import the model
ridge_model = pickle.load(open('models/ridge_model.pkl', 'rb'))
# Load the scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():  
    if request.method == 'POST':
       Temperature = float(request.form.get('Temperature'))
       RH = float(request.form.get('RH'))
       Ws = float(request.form.get('Ws'))
       Rain = float(request.form.get('Rain'))
       FFMC = float(request.form.get('FFMC'))
       DMC = float(request.form.get('DMC'))
       ISI = float(request.form.get('ISI'))
       Classes = float(request.form.get('Classes'))
       Region = float(request.form.get('Region'))

       new_data_scaled=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes]])
       prediction = ridge_model.predict(new_data_scaled)
       return render_template('home.html', result=prediction)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)