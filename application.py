from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd

application = Flask(__name__)
CORS(application, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True

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
print("Hello, Server !!!!!")

@application.route("/")
def hello_world():
    print("Hello, World!")
    return "<p>Hello, World!</p>"


@application.route("/task_1", methods=['POST'])
def task_1():
    print("Task 1")
    req = request.get_json()
    prediction = task1_model.predict(get_health_values(req))
    prediction_int = int(prediction[0])
    return jsonify({"prediction": prediction_int})






