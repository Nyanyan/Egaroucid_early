# -*- coding: utf-8 -*-
from flask import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

if __name__ == '__main__':
    app.run()