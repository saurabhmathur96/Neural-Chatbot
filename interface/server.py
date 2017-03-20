from flask import Flask, render_template, jsonify
from time import sleep

app = Flask(__name__)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/respond', methods=['POST'])
def respond():
    sleep(2)
    return jsonify({ 'response': 'i don\'t know' })


if __name__ == '__main__':
    app.run(port=8000)