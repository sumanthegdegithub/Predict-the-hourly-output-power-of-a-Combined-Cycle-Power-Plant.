import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('ML_project_RF.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    X_train = pd.DataFrame(int_features)
    X_train = X_train/pd.DataFrame([37.11,81.56,1033.30,100.16])
    X_train = (X_train - pd.DataFrame([0.548370,0.638548,0.980277,0.748802]))/pd.DataFrame([0.200604,0.155546,0.005749,0.145842])
    prediction = model.predict(X_train.values.reshape(1,4))

    output = prediction[0]

    return render_template('index.html', prediction_text='Hourly output expected {} MW'.format(output))



if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 8080)