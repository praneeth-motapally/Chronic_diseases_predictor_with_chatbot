from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



app = Flask(__name__)

# Load all models at startup
models = {
    'diabetes': load_model('models/diabetes_Deep_model.h5'),
    'breast_cancer': pickle.load(open('models/breast_cancer.pkl', 'rb')),
     'heart': load_model('models/heart_Deep_model.h5'),
     'kidney': load_model('models/kidney_deep_model.h5'),
    'liver': pickle.load(open('models/liver.pkl', 'rb')),
}

# Load scaler(s)
scalers = {
    'diabetes': pickle.load(open('models/scaler.pkl', 'rb')),
     'heart': pickle.load(open('models/heart_scaler.pkl', 'rb')),
     'kidney': pickle.load(open('models/kidney_scaler.pkl', 'rb')),
    # Add scalers for other diseases if needed
}

# Map input lengths to disease keys
input_length_to_disease = {
    8: 'diabetes',  
    26: 'breast_cancer',
    13: 'heart',
    18: 'kidney',
    10: 'liver',
}
# Preprocessing dictionary for kidney disease categorical values
kidney_dict = {
    "rbc": {"abnormal": 1, "normal": 0},
    "pc": {"abnormal": 1, "normal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
}

def preprocess_kidney_input(form_data):
    processed = []
    for key, value in form_data.items():
        if key == "id":
            continue
        value = value.lower()
        if key in kidney_dict:
            processed.append(kidney_dict[key].get(value, 0))
        else:
            processed.append(float(value))
    return processed

def predict(values, original_form_data=None):
    values = np.asarray(values).reshape(1, -1)
    disease_key = input_length_to_disease.get(values.shape[1])

    if disease_key is None:
        raise ValueError("Invalid input size for any disease model")

    model = models[disease_key]

    # Apply scaler if exists
    if disease_key in scalers:
        values = scalers[disease_key].transform(values)

    # Deep learning models
    if disease_key in ['diabetes', 'heart', 'kidney']:
        prob = model.predict(values)
        prob_value = prob[0][0] if prob.ndim == 2 else prob[0]
        return prob_value * 100

    # Pickle models
    elif hasattr(model, 'predict_proba'):
        prob = model.predict_proba(values)
        return prob[0][1] * 100 if prob.shape[1] > 1 else prob[0][0] * 100

    else:
        return model.predict(values)[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/diabetes_info", methods=['GET', 'POST'])
def diabetesInfo():
    return render_template('diabetesInfo.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/heart_info", methods=['GET', 'POST'])
def heartInfo():
    return render_template('heartInfo.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/kidney_info", methods=['GET', 'POST'])
def kidneyInfo():
    return render_template('kidneyInfo.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/liver_info", methods=['GET', 'POST'])
def liverInfo():
    return render_template('liverInfo.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/about_us", methods=['GET', 'POST'])
def aboutUs():
    return render_template('about_us.html')

@app.route("/contact_us", methods=['GET', 'POST'])
def contactUs():
    return render_template('contact_us.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['GET', 'POST'])
def predictPage():
    try:
        if request.method == 'POST':
            form_data = request.form.to_dict()
            print("Form Input:", form_data)

            input_size = len(form_data) - (1 if 'id' in form_data else 0)
            disease_key = input_length_to_disease.get(input_size)

            # Kidney-specific preprocessing
            if disease_key == 'kidney':
                values = preprocess_kidney_input(form_data)
            else:
                # Generic preprocessing
                for key in form_data:
                    value = form_data[key].lower()
                    if value in ['yes', 'no']:
                        form_data[key] = 1 if value == 'yes' else 0
                    elif value in ['male', 'female']:
                        form_data[key] = 1 if value == 'male' else 0
                    elif value in ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']:
                        mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}
                        form_data[key] = mapping[value]
                    elif value in ['normal', 'st-t wave abnormality', 'left ventricular hypertrophy']:
                        mapping = {'normal': 0, 'st-t wave abnormality': 1, 'left ventricular hypertrophy': 2}
                        form_data[key] = mapping[value]
                    elif value in ['upsloping', 'flat', 'downsloping']:
                        form_data[key] = {'upsloping': 0, 'flat': 1, 'downsloping': 2}[value]
                    elif value in ['fixed defect', 'normal', 'reversible defect']:
                        form_data[key] = {'normal': 1, 'fixed defect': 2, 'reversible defect': 3}[value]
                    else:
                        try:
                            form_data[key] = float(value)
                        except:
                            raise ValueError(f"Invalid value: {value}")

                values = list(map(float, form_data.values()))

            disease_probability = predict(values, form_data)
            return render_template("predict.html", prob=round(disease_probability, 2), disease=disease_key)

    except Exception as e:
        return render_template("home.html", message=f"Error: {str(e)}")


@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)