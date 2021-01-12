from flask import Flask as fl, render_template as rt, url_for

app = fl(__name__)

@app.route("/")
@app.route("/home")
def home():
    return rt('home.html')