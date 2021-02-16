from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('Pickle_XGB_Model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


normalize_to = MinMaxScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        engine_size = float(request.form['engine_size'])
        curb_weight = float(request.form['curb_weight'])
        horse_power = float(request.form['horse_power'])
        city_mpg = float(request.form['city_mpg'])
        highway_mpg = float(request.form['highway_mpg'])
        car_width = float(request.form['car_width'])
        car_length = float(request.form['car_length'])
        fuel_system_ = ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi']
        fuel_system = request.form['fuel_system']
        fuel_system = fuel_system_.index(fuel_system) +1
        drivewheel_ = ['rwd', 'fwd', '4wd']
        drivewheel = request.form['drive_wheel']
        drivewheel = drivewheel_.index(drivewheel) +1
        bore_ratio = float(request.form['bore_ratio'])
        engine_location_ = ['front', 'rear']
        engine_location = request.form['engine_location']
        engine_location = engine_location_.index(engine_location) +1
        wheel_base = float(request.form['wheel_base'])
        aspiration_ = ['std', 'turbo']
        aspiration = request.form['aspiration']
        aspiration = aspiration_.index(aspiration) +1
        
        data = [[engine_size ,curb_weight,horse_power,city_mpg,highway_mpg,car_width,car_length,fuel_system,
                 drivewheel,bore_ratio,engine_location,wheel_base,aspiration]]
        #column = ['enginesize', 'curbweight', 'horsepower', 'citympg', 'highwaympg', 'carwidth', 'carlength', 'fuelsystem', 'drivewheel', 'boreratio', 'enginelocation', 'wheelbase', 'aspiration']
        #df = pd.DataFrame(data,columns=column)
        prediction=model.predict(data)
        
        return render_template('index.html',prediction_text="Price will be ${}".format(prediction[0]))
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
