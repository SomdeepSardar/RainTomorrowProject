from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import logging

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            Location = int(request.form.get("Location")),
            MinTemp = float(request.form.get('MinTemp')),
            MaxTemp = float(request.form.get('MaxTemp')),
            Rainfall = float(request.form.get("Rainfall")), 
            Evaporation= float(request.form.get("Evaporation")), 
            Sunshine = float(request.form.get("Sunshine")),
            WindGustSpeed = float(request.form.get("WindGustSpeed")),
            WindSpeed9am = float(request.form.get("WindSpeed9am")),
            WindSpeed3pm = float(request.form.get("WindSpeed3pm")),
            Humidity9am = float(request.form.get("Humidity9am")), 
            Humidity3pm = float(request.form.get("Humidity3pm")),
            Pressure9am = float(request.form.get("Pressure9am")),
            Cloud9am = float(request.form.get("Cloud9am")),
            Cloud3pm = float(request.form.get("Cloud3pm")),
            WindGustDir = int(request.form.get("WindGustDir")), 
            WindDir9am = int(request.form.get("WindDir9am")),
            WindDir3pm = int(request.form.get("WindDir3pm"))
        )
    new_data = data.get_data_as_dataframe()
    logging.info("New Data Point inside app.py\n")
    logging.info(new_data)
    print('app.py new data\n')
    print(new_data.info())
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    # print('type of the pred variable, dlt later')
    # print(type(pred[0]))

    # results = round(pred[0],2)
    results = pred[0]
    if results == 0.0:
        true_result = "No"
    else:
        true_result = 'Yes'


    return render_template("results.html", final_result = true_result)

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser