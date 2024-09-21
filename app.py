from flask import Flask,request,jsonify
import joblib

app = Flask(__name__)

@app.route("/")
def home_page():
    return "Welcome to Home Page"



if __name__ == '__main__':
    app.run(debug=True)