from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

task1_model = pickle.load(open('./ml-models/model_1/deases_forest_model.pkl', 'rb'))

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def get_health_values(req):
    df_input = pd.DataFrame([[
       req["age"] ,
       req["sex"],
       req["cp"],
       req["trestbps"],
       req["chol"],
       req["fbs"],
       req["restecg"],
       req["thalach"],
       req["exang"],
       req["oldpeak"],
       req["slope"],
       req["ca"],
       req["thal"]
    ]], columns=feature_names)
    return df_input


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.post("/task_1" )
def task_1():
    req = request.get_json()
    prediction = task1_model.predict(get_health_values(req))
    prediction_int = int(prediction[0])
    return jsonify({"prediction": prediction_int})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



