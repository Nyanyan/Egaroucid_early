# -*- coding: utf-8 -*-
from flask import *
import json
import subprocess
from time import sleep

hw = 8

app = Flask(__name__)
ai = subprocess.Popen('./a.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

@app.route('/')
def index():
    return render_template('base.html')

@app.route("/ai", methods=["POST"])
def call_ai():
    req = dict(request.form)
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]
    for y in range(hw):
        for x in range(hw):
            grid[y][x] = int(req[str(y * hw + x)])
    ai_player = int(req["ai_player"])
    tl = int(req["tl"])
    stdin = str(ai_player) + '\n' + str(tl) + '\n'
    for y in range(hw):
        for x in range(hw):
            stdin += '0' if grid[y][x] == 0 else '1' if grid[y][x] == 1 else '.'
    ai.stdin.write(stdin.encode('utf-8'))
    ai.stdin.flush()
    print('sent')
    r, c = [int(i) for i in ai.stdout.readline().decode().strip().split()]
    print(r, c)
    res = {"r": r, "c": c}
    return jsonify(values=json.dumps(res))

if __name__ == '__main__':
    app.run()