# -*- coding: utf-8 -*-
from flask import *
import json
import subprocess
from time import sleep, time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

hw = 8
reload_limit = 1.0

application = Flask(__name__)
limiter = Limiter(application, key_func=get_remote_address, default_limits=["500 per day", "50 per hour"])
ai = subprocess.Popen('./ai.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

@application.route('/')
@limiter.limit("1/second", override_defaults=False)
def index():
    return render_template('base.html')

@application.route("/ai", methods=["POST"])
@limiter.limit("5/second", override_defaults=False)
def call_ai():
    req = dict(request.form)
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]
    for y in range(hw):
        for x in range(hw):
            tmp = req[str(y * hw + x)]
            try:
                tmp = int(tmp)
                if -1 <= tmp <= 2:
                    grid[y][x] = tmp
                else:
                    return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
            except:
                return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
    try:
        ai_player = int(req["ai_player"])
        if ai_player != 0 and ai_player != 1:
            return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
    except:
        return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
    try:
        tl = int(req["tl"])
        if tl < 2 or 200 < tl:
            return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
    except:
        return jsonify(values=json.dumps({"r": -1, "c": -1, "s": 0.0}))
    stdin = str(ai_player) + '\n' + str(tl) + '\n'
    for y in range(hw):
        for x in range(hw):
            stdin += '0' if grid[y][x] == 0 else '1' if grid[y][x] == 1 else '.'
    ai.stdin.write(stdin.encode('utf-8'))
    ai.stdin.flush()
    r, c, s = [float(i) for i in ai.stdout.readline().decode().strip().split()]
    r = int(r)
    c = int(c)
    print(r, c, s)
    res = {"r": r, "c": c, "s": s}
    return jsonify(values=json.dumps(res))

if __name__ == '__main__':
    print('start python')
    application.run()