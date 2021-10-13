from random import random, randrange
import trueskill
import subprocess
from tqdm import trange

hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def empty(grid, y, x):
    return grid[y][x] == -1 or grid[y][x] == 2

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

def check(grid, player, y, x):
    res_grid = [[False for _ in range(hw)] for _ in range(hw)]
    res = 0
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if empty(grid, ny, nx):
            continue
        if grid[ny][nx] == player:
            continue
        #print(y, x, dr, ny, nx)
        plus = 0
        flag = False
        for d in range(hw):
            nny = ny + d * dy[dr]
            nnx = nx + d * dx[dr]
            if not inside(nny, nnx):
                break
            if empty(grid, nny, nnx):
                break
            if grid[nny][nnx] == player:
                flag = True
                break
            #print(y, x, dr, nny, nnx)
            plus += 1
        if flag:
            res += plus
            for d in range(plus):
                nny = ny + d * dy[dr]
                nnx = nx + d * dx[dr]
                res_grid[nny][nnx] = True
    return res, res_grid

def pot_canput_line(arr):
    res_p = 0
    res_o = 0
    for i in range(len(arr) - 1):
        if arr[i] == -1 or arr[i] == 2:
            if arr[i + 1] == 0:
                res_o += 1
            elif arr[i + 1] == 1:
                res_p += 1
    for i in range(1, len(arr)):
        if arr[i] == -1 or arr[i] == 2:
            if arr[i - 1] == 0:
                res_o += 1
            elif arr[i - 1] == 1:
                res_p += 1
    return res_p, res_o

class reversi:
    def __init__(self):
        self.grid = [[-1 for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = 1
        self.grid[3][4] = 0
        self.grid[4][3] = 0
        self.grid[4][4] = 1
        self.player = 0 # 0: 黒 1: 白
        self.nums = [2, 2]

    def move(self, y, x):
        plus, plus_grid = check(self.grid, self.player, y, x)
        if (not empty(self.grid, y, x)) or (not inside(y, x)) or not plus:
            print('Please input a correct move')
            return 1
        self.grid[y][x] = self.player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    self.grid[ny][nx] = self.player
        self.nums[self.player] += 1 + plus
        self.nums[1 - self.player] -= plus
        self.player = 1 - self.player
        return 0
    
    def check_pass(self):
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == 2:
                    self.grid[y][x] = -1
        res = True
        for y in range(hw):
            for x in range(hw):
                if not empty(self.grid, y, x):
                    continue
                plus, _ = check(self.grid, self.player, y, x)
                if plus:
                    res = False
                    self.grid[y][x] = 2
        if res:
            #print('Pass!')
            self.player = 1 - self.player
        return res

    def output(self):
        print('  ', end='')
        for i in range(hw):
            print(chr(ord('a') + i), end=' ')
        print('')
        for y in range(hw):
            print(str(y + 1) + ' ', end='')
            for x in range(hw):
                print('○' if self.grid[y][x] == 0 else '●' if self.grid[y][x] == 1 else '* ' if self.grid[y][x] == 2 else '. ', end='')
            print('')
    
    def output_file(self):
        res = ''
        for y in range(hw):
            for x in range(hw):
                res += '*' if self.grid[y][x] == 0 else 'O' if self.grid[y][x] == 1 else '-'
        res += ' *'
        return res

    def end(self):
        if min(self.nums) == 0:
            return True
        res = True
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == -1 or self.grid[y][x] == 2:
                    res = False
        return res
    
    def judge(self):
        if self.nums[0] > self.nums[1]:
            #print('Black won!', self.nums[0], '-', self.nums[1])
            return 0
        elif self.nums[1] > self.nums[0]:
            #print('White won!', self.nums[0], '-', self.nums[1])
            return 1
        else:
            #print('Draw!', self.nums[0], '-', self.nums[1])
            return -1

def self_play():
    ais = []
    for i in range(2):
        ais.append(subprocess.Popen('./self_play.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE))
        ais[i].stdin.write((str(i) + '\n' + str(i) + '\n').encode('utf-8'))
        ais[i].stdin.flush()
    rv = reversi()
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        s = ''
        for y in range(hw):
            for x in range(hw):
                s += '0' if rv.grid[y][x] == 0 else '1' if rv.grid[y][x] == 1 else '.'
        s += '\n'
        ais[rv.player].stdin.write(s.encode('utf-8'))
        ais[rv.player].stdin.flush()
        move = ais[rv.player].stdout.readline().decode().rstrip()
        x = ord(move[0]) - 97
        y = int(move[1]) - 1
        rv.move(y, x)
    stones = [0, 0]
    for y in range(hw):
        for x in range(hw):
            for p in range(2):
                stones[p] += rv.grid[y][x] == p
    ranks = [0, 1] if stones[0] > stones[1] else [1, 0] if stones[0] < stones[1] else [0, 0]
    for i in range(2):
        ais[i].kill()
    return ranks

n_param = 74
population = 10
times1 = 5
times2 = 10

params = [[random() * 2.0 - 1.0 for _ in range(n_param)] for _ in range(population)]
rates = [None for _ in range(population)]

mu = 25.0
sigma = mu / 3.0
beta = sigma / 2.0
tau = sigma / 100.0
draw_probability = 0.1
backend = None

env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability, backend=backend)

for i in range(population):
    rates[i] = env.create_rating()


for _ in trange(population * times1):
    p0 = randrange(0, population)
    p1 = p0
    while p0 == p1:
        p1 = randrange(0, population)
    with open('param/param_eval0.txt', 'w') as f:
        for elem in params[p0]:
            f.write(str(elem) + '\n')
    with open('param/param_eval1.txt', 'w') as f:
        for elem in params[p1]:
            f.write(str(elem) + '\n')
    ranks = self_play()
    (rates[p0],), (rates[p1],) = env.rate(((rates[p0],), (rates[p1],),), ranks=ranks)

for i in range(population):
    print(env.expose(rates[i]), end=' ')
print('')


n = 0
while True:
    n += 1
    print(n)
    parent0 = randrange(0, population)
    parent1 = parent0
    while parent0 == parent1:
        parent1 = randrange(0, population)
    children = [[-10.0 for _ in range(n_param)] for _ in range(2)]
    dv = randrange(1, n_param - 1)
    for i in range(dv):
        children[0][i] = params[parent0][i]
        children[1][i] = params[parent1][i]
    for i in range(dv, n_param):
        children[0][i] = params[parent1][i]
        children[1][i] = params[parent0][i]
    rates_children = [env.create_rating() for _ in range(2)]
    for child in range(2):
        with open('param/param_eval0.txt', 'w') as f:
            for elem in children[child]:
                f.write(str(elem) + '\n')
        for _ in range(times2):
            p = randrange(0, population)
            with open('param/param_eval1.txt', 'w') as f:
                for elem in params[p]:
                    f.write(str(elem) + '\n')
            ranks = self_play()
            (rates_children[child],), (rates[p],) = env.rate(((rates_children[child],), (rates[p],),), ranks=ranks)
    rates4 = [env.expose(rates[parent0]), env.expose(rates[parent1]), env.expose(rates_children[0]), env.expose(rates_children[1])]
    parent_params = [[elem for elem in params[p]] for p in [parent0, parent1]]
    for p in [parent0, parent1]:
        mx = -10000.0
        idx = -1
        for j in range(4):
            if rates4[j] > mx:
                mx = rates4[j]
                idx = j
        rates4[idx] = -10000.0
        if idx < 2:
            params[p] = [elem for elem in parent_params[idx]]
            parents = [parent0, parent1]
            rates[p] = rates[parents[idx]]
        else:
            params[p] = [elem for elem in children[idx - 2]]
            rates[p] = rates_children[idx - 2]
            print('child won')
    mx = -10000.0
    idx = -1
    for i in range(population):
        rate = env.expose(rates[i])
        if rate > mx:
            mx = rate
            idx = i
    print('max rate', mx, idx)
    with open('param/param_eval_best.txt', 'w') as f:
        for elem in params[idx]:
            f.write(str(elem) + '\n')

        

    