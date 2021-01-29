import cv2
import os
from os import path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.recognition import recognize
from utils.train import train_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/image', methods=['GET', 'POST'])
def image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    # file.save(filename)

    recognition_result = recognize(filename)
    names, recognized_list = recognition_result.split(" ")

    print(names)

    return jsonify(success=True, recognition=names)


@app.route('/add/<name>', methods=['POST'])
def add(name):
    pic = str(name)
    pic = pic.title
    file = request.files['added_image']
    save_path = "/src/utils/images"
    os.rename(file, pic + ".png")
    if path.exist(save_path + "/" + pic):
        cv2.imwrite(save_path + "/" + pic + "/" + file.filename, file)


@app.route('/train', methods=['POST'])
def train():
    train_model()


app.run(port=5000)
