import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
import keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

flower_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

model = load_model('Flower_classifier_model.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result)*100 )
    return outcome

@app.route('/')
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def display():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)
      
	predictions = classify_images(img_path)
    
	return render_template("acknowledgement.html", img_path = img_path, prediction= predictions, name = img.filename)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)