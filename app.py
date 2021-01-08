from flask import Flask, Response, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/image', methods=['POST'])
def image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    return jsonify(success = True)

app.run(port=5000)