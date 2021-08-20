# -*- coding: utf-8 -*-
from flask import *
import json
import subprocess
from time import sleep

hw = 8

application = Flask(__name__)
ai = subprocess.Popen('./ai.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

ip_dict = {}

def ip2num(ip):
    nums = [int(i) for i in ip.split('.')]
    res = 0
    for num in nums:
        res *= 256
        res += num
    return res

@application.route('/')
def index():
    ip_num = ip2num(request.remote_addr)
    
    return render_template('base.html')

@application.route("/ai", methods=["POST"])
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
                    return jsonify(values=json.dumps({"r": -1, "c": -1}))
            except:
                return jsonify(values=json.dumps({"r": -1, "c": -1}))
    print('grid done')
    try:
        ai_player = int(req["ai_player"])
        if ai_player != 0 and ai_player != 1:
            return jsonify(values=json.dumps({"r": -1, "c": -1}))
    except:
        return jsonify(values=json.dumps({"r": -1, "c": -1}))
    try:
        tl = int(req["tl"])
        if tl < 10 or 200 < tl:
            return jsonify(values=json.dumps({"r": -1, "c": -1}))
    except:
        return jsonify(values=json.dumps({"r": -1, "c": -1}))
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
    print('start python')
    application.run()