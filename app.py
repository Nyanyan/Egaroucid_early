# -*- coding: utf-8 -*-
from flask import *
import json
import subprocess
from time import sleep

hw = 8

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route("/move", methods=["POST"])
def get_move():
    req = dict(request.form)
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]

if __name__ == '__main__':
    app.run()