from flask import Flask, request, render_template
from src.exception import CustomException
import sys
import os
from src.utils import load_obj
import pandas as pd



app = Flask(__name__)

transformer = load_obj("artifacts/preprocessor.pkl")
model = load_obj("artifacts/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        race_ethnicity = request.form['race_ethnicity']
        parental_level_of_education = request.form['parental_level_of_education']
        lunch = request.form['lunch']
        test_preparation_course = request.form['test_preparation_course']
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])
        
        input_data = pd.DataFrame({
            'gender':[gender],
            'race_ethnicity':[race_ethnicity],
            'parental_level_of_education':[parental_level_of_education],
            'lunch':[lunch],
            'test_preparation_course':[test_preparation_course],
            'reading_score':[reading_score],
            'writing_score':[writing_score]
        })
        
        new_transformer = transformer.transform(input_data)
        prediction = model.predict(new_transformer)
        
        return render_template("index.html", prediction_text=f"Predicted math score:{prediction[0]:.2f}")    
        
    except Exception as e:
        raise CustomException(e,sys)



if __name__=="__main__":
    app.run(debug=True)