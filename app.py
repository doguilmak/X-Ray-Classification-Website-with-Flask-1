import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, redirect, url_for, request
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

classifier = load_model('model.h5')

class_map = {0: "Normal", 
            1: "Pneumonia"}

def predict_class(filename):    
    img = image.load_img(filename, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    xrayClass = classifier.predict(images, batch_size=1)[0]
    return class_map[int(xrayClass[0])]

# PATH="D:/New_folder"
PATH = "C:/Users/Kullanıcı/Desktop/machine_learning_classify_xray_webpage/uploads"
app.config["IMAGE_UPLOADS"] = PATH

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/index.html')
def home():
   return render_template('index.html')

@app.route('/predict.html', methods=["GET", "POST"])
def upload_image():
    xrayClass = ''
    if request.method == "POST":
        if request.files:
            image=request.files["image"]
            imagename = secure_filename(image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))         
            xrayClass=predict_class(PATH + "/" + image.filename)
            return render_template("predict.html", xrayClass=xrayClass)
        
    return render_template("predict.html", xrayClass=xrayClass)

if __name__ == "__main__":
    app.run(debug=True)