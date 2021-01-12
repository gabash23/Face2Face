from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
from recognition import recognize
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(filename)

        recognition_result = recognize(filename)
        print(recognition_result)

        return jsonify(success = True, recognition = recognition_result)
    elif request.method == 'GET':
        return render_template("home.html")

app.run(port=5000)