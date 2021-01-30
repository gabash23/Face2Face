from typing import List, Tuple

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
    file.save(filename)

    recognition_result = recognize(filename)
    names: List[str] or str

    names = recognition_result

    return jsonify(success=True, recognition=names)


@app.route('/add', methods=['POST'])
def add():
    file = request.files['file']
    filename = secure_filename(file.filename)
    UPLOAD_FOLDER = '/src/utils/images/' + filename
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    return jsonify(success=True)


@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify(success=True)


app.run(port=5000)
