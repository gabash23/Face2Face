from typing import List, Tuple

import cv2
import os
import requests
from os import path
from flask import Flask, flash, request, jsonify, render_template
from flask_cors import CORS
from utils.recognition import recognize
from utils.train import train_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)


def legal_file(filename, extensions) -> bool:
    if filename == ' ':
        flash('Please select a file')
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/image', methods=['POST'])
def image() -> str:
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    conf: int = -1
    names: str = ""

    recognition_result = recognize(filename)

    if recognition_result != "Unknown":
        conf = recognition_result[1]
        names = recognition_result[0]
        image_name = recognition_result[2]
        r = requests.get("http://127.0.0.1:5000/static/" + image_name)

    else:
        names = recognition_result
        image_name = None

    os.remove(file.filename)

    return jsonify(success=True, recognition=names, image_name=image_name)


@app.route('/add', methods=['POST'])
def add() -> None:
    file = request.files['file']
    file_name = secure_filename(file.filename)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}
    if legal_file(file_name, ALLOWED_EXTENSIONS):
        name = request.args.get('name')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(base_dir, "utils")
        image_dir = os.path.join(utils_dir, "images")
        final_dir = os.path.join(image_dir, name)
        print(final_dir)
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        file.save(os.path.join(final_dir, file_name))
    else:
        flash("Please select an image")
        return jsonify(success=False)

    return jsonify(success=True)


@app.route('/train', methods=['GET'])
def train() -> None:
    train_model()
    return jsonify(success=True)


app.run(port=5000, debug=True)
