from typing import List, Tuple

import cv2
import os
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


@app.route('/image', methods=['GET', 'POST'])
def image() -> str:
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    recognition_result = recognize(filename)
    names: List[str] or str

    names = recognition_result

    return jsonify(success=True, recognition=names)


@app.route('/add', methods=['GET', 'POST'])
def add() -> None:
    print('hello')
    file = request.files['file']
    file_name = secure_filename(file.filename)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    print("______________________________")
    if legal_file(file.filename, ALLOWED_EXTENSIONS):
        UPLOAD_FOLDER = '/src/utils/images/' + file_name
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        print(app.config['UPLOAD_FOLDER'])
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

    else:
        flash("Please select an image")
        return jsonify(success=False)

    return jsonify(success=True)


@app.route('/train', methods=['POST'])
def train() -> None:
    train_model()
    return jsonify(success=True)


app.run(port=5000)
