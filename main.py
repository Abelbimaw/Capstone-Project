import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd


app = Flask("__name__")
# Load TFLite model and allocate tensors.
#model = tf.lite.Interpreter(model_path="model_akurasi_overfitting_kecil.tflite")
#model.allocate_tensors()
model = load_model('model_akurasi_overfitting_kecil.h5')

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route("/heart")
def heartPage():
    return render_template('heart.html')

@app.route('/predict',methods = ['POST'])
def predict():
    model = load_model('model_akurasi_overfitting_kecil.h5')
    
    ages = float(request.form['Age'])
    genders = float(request.form['Gender'])
    chestpain = float(request.form['Chest_Pain'])
    resting_blood = float(request.form['Resting_Blood_Plessure'])
    cholesterol = float(request.form['Cholesterol'])
    fastingblood = float(request.form['fbs'])
    restecg = float(request.form['resting_electrocardiographic'])
    maximum_heart = float(request.form['Maximum_Heart_Rate'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['ST_depression_induced_by_exercise'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    
    data = [[ages, genders, chestpain, resting_blood, cholesterol, fastingblood, restecg, maximum_heart, exang, oldpeak, slope, ca, thal]]
    
    new_df = pd.DataFrame(data, columns = ['ages', 'genders', 'chestpain', 'resting_blood', 'cholesterol', 'fastingblood', 'restecg', 'maximum_heart', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    samples_to_predict = np.array(new_df).astype('float32')
    single = model.predict(samples_to_predict)
    classes = np.argmax(single, axis = 1)
    if(classes == [0]):
        prediction = 'Not Heart Disease'
    else:
        prediction = 'Heart Disease'
    
    predict_classes = model.predict_classes(new_df)
    if(predict_classes == [0]):
        print("Not Heart Disease")
    else:
        print("Heart Disease")
    
    return render_template('heart.html', output1= "Patient has predicted = " +prediction, ages = float(request.form['Age']), genders = float(request.form['Gender']), chestpain = float(request.form['Chest_Pain']), resting_blood = float(request.form['Resting_Blood_Plessure']), cholesterol = float(request.form['Cholesterol']), fastingblood = float(request.form['fbs']), restecg = float(request.form['resting_electrocardiographic']),maximum_heart= float(request.form['Maximum_Heart_Rate']),exang = float(request.form['exang']), oldpeak = float(request.form['ST_depression_induced_by_exercise']),slope = float(request.form['slope']), ca = float(request.form['ca']), thal = float(request.form['thal']))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
#    '''
#    For direct API calls trought request
#    '''
#    data = request.get_json(force=True)
#    prediction = model.predict(np.array(list(data.values())))
#
#    output = prediction[0]
#    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)