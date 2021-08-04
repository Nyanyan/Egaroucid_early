# -*- coding: utf-8 -*-
from flask import *

hw = 8

app = Flask(__name__)

@app.route('/')
def index():
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    return render_template('base.html', grid=grid)

if __name__ == '__main__':
    app.run()