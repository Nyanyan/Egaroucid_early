import subprocess
import sys
from random import randint

hw = 8
hw2 = 64

dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]


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



class reversi:
    def __init__(self):
        self.grid = [[-1 for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = 1
        self.grid[3][4] = 0
        self.grid[4][3] = 0
        self.grid[4][4] = 1
        self.player = 0  # 0: 黒 1: 白
        self.nums = [2, 2]

    def move(self, y, x):
        plus, plus_grid = check(self.grid, self.player, y, x)
        if (not empty(self.grid, y, x)) or (not inside(y, x)) or not plus:
            print('Please input a correct move')
            return
        self.grid[y][x] = self.player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    self.grid[ny][nx] = self.player
        self.nums[self.player] += 1 + plus
        self.nums[1 - self.player] -= plus
        self.player = 1 - self.player

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
                print('O ' if self.grid[y][x] == 0 else 'X ' if self.grid[y]
                      [x] == 1 else '* ' if self.grid[y][x] == 2 else '. ', end='')
            print('')

    def end(self):
        res = True
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == -1 or self.grid[y][x] == 2:
                    res = False
        return res

    def judge(self):
        if self.nums[0] > self.nums[1]:
            print('Black won!', self.nums[0], '-', self.nums[1])
        elif self.nums[1] > self.nums[0]:
            print('White won!', self.nums[0], '-', self.nums[1])
        else:
            print('Draw!', self.nums[0], '-', self.nums[1])


ln = int(input())
'''
raw_early_stages = [input() for _ in range(ln)]
early_stages = []
for grid_str in raw_early_stages:
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]
    for y in range(hw):
        for x in range(hw):
            grid[y][x] = 0 if grid_str[y * hw + x] == '0' else 1 if grid_str[y * hw + x] == '1' else -1
    early_stages.append(grid)
'''
ais = [subprocess.Popen('./decide.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
ais[0].stdin.write('0\n'.encode('utf-8')) # best
ais[1].stdin.write('1\n'.encode('utf-8')) # new
for i in range(2):
    ais[i].stdin.write((str(randint(1, 2000000000)) + '\n').encode('utf-8'))
new_win = 0
best_win = 0
for game_idx in range(ln):
    for player2ai in [[0, 1], [1, 0]]:
        sys.stderr.write('=')
        sys.stderr.flush()
        rv = reversi()
        rv.player = 0
        while True:
            if rv.check_pass() and rv.check_pass():
                break
            stdin = ''
            for y in range(hw):
                for x in range(hw):
                    stdin += '0' if rv.grid[y][x] == rv.player else '1' if rv.grid[y][x] == 1 - rv.player else '.'
            stdin += '\n'
            #sys.stderr.write(stdin)
            ais[player2ai[rv.player]].stdin.write(stdin.encode('utf-8'))
            ais[player2ai[rv.player]].stdin.flush()
            y, x, _ = [float(i) for i in ais[player2ai[rv.player]].stdout.readline().decode().strip().split()]
            y = int(y)
            x = int(x)
            rv.move(y, x)
            #rv.output()
            if rv.end():
                break
        rv.check_pass()
        #rv.output()
        if rv.nums[player2ai[0]] < rv.nums[player2ai[1]]:
            new_win += 1
        elif rv.nums[player2ai[0]] > rv.nums[player2ai[1]]:
            best_win += 1
for i in range(2):
    ais[i].kill()
sys.stderr.write('end ' + str(best_win) + ' ' + str(new_win))
sys.stderr.flush()
print(best_win, new_win)