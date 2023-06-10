from flask import  Flask, render_template, request
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = tf.keras.models.load_model("digitreg.h5")

@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/"+imagefile.filename
    imagefile.save(image_path)


    img = cv.imread(image_path)[:, :, 0]
    img = np.invert(np.array([img]))
    pred = model.predict(img)
    result = np.argmax(pred)
    # print(f"Prediction : {result}")
    # plt.imshow(img[0], cmap=plt.cm.binary)
    # plt.show()

    classification ='%s' %(result)
    return render_template('index.html',prediction=classification)

if __name__ == "__main__":
    app.run(port=1000, debug=True)
