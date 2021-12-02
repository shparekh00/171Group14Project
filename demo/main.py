from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    if request.method == 'GET':
        return render_template('index.html', is_delayed=-1)
    elif request.method == 'POST':

        # Load trained models / scalers/ encoders

        with open('./models/encoder_carrier.pkl', 'rb') as f:
            encoder_carrier = pickle.load(f)

        with open('./models/encoder_origin.pkl', 'rb') as f:
            encoder_origin = pickle.load(f)

        with open('./models/encoder_dest.pkl', 'rb') as f:
            encoder_dest = pickle.load(f)

        with open('./models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('./models/model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Get input from UI

        month = request.form['month']
        day = request.form['day']
        dep_time = request.form['dep_time']
        dep_delay = request.form['dep_delay']
        arr_time = request.form['arr_time']
        carrier = encoder_carrier.transform([[request.form['carrier']]])[0][0]
        flight = request.form['flight']
        origin = encoder_origin.transform([[request.form['origin']]])[0][0]
        dest = encoder_dest.transform([[request.form['dest']]])[0][0]
        air_time = request.form['air_time']
        distance = request.form['distance']
        hour = request.form['hour']
        minute = request.form['minute']

        # Create, scale, and predict

        input_x = [month, day, dep_time, dep_delay, arr_time, carrier, flight, origin, dest, air_time, distance, hour, minute]
        scaled_x = scaler.transform([input_x])
        result = model.predict(scaled_x)[0]

        return render_template('index.html', is_delayed=result)
