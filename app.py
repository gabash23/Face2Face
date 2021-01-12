from flask import Flask, Response, request, jsonify
from recognition import recognize
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/image', methods=['POST'])
def image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    recognition_result = recognize(filename)
    print(recognition_result)

    return jsonify(success = True, recognition = recognition_result)

app.run(port=5000)