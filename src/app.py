from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.recognition import recognize
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
    print(recognition_result)

    return jsonify(success = True, recognition = recognition_result)

app.run(port=5000)