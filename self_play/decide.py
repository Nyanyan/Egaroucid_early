from random import randint
import subprocess
from os import rename, path, listdir
from tqdm import trange
import numpy as np

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


record_num = digit(sum(path.isfile(path.join('records', name)) for name in listdir('records')), 7)
game_num = 60
new_b_win = 0
best_b_win = 0
new_w_win = 0
best_w_win = 0
all_idx = 0
for n_record in range(game_num // 6):
    for mode in [[0, 1], [1, 0]]:
        for book_mode in [[0, 1], [1, 0], [1, 1]]:
            ais = [subprocess.Popen('./self_play.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
            for i in range(2):
                if mode[i] == 0:
                    param = '0.0\n1.0\n0.7\n0.05\n0.1\n'
                else:
                    param = '0.0\n1.0\n0.7\n0.05\n0.01\n'
                ais[i].stdin.write((str(randint(1, 2000000000)) + '\n' + param + str(i) + '\n' + str(book_mode[i]) + '\n' + str(mode[i]) + '\n').encode('utf-8'))
                ais[i].stdin.flush()
            rv = reversi()
            #boards = [[], []]
            n_turn = 0
            #record = ''
            while True:
                if rv.check_pass() and rv.check_pass():
                    break
                board_str = ''
                for y in range(hw):
                    for x in range(hw):
                        board_str += '0' if rv.grid[y][x] == 0 else '1' if rv.grid[y][x] == 1 else '.'
                    board_str += '\n'
                #print(board_str)
                ais[rv.player].stdin.write(board_str.encode('utf-8'))
                ais[rv.player].stdin.flush()
                coord = ais[rv.player].stdout.readline().decode()
                #record += coord[:2]
                x = int(ord(coord[0]) - ord('a'))
                y = int(coord[1]) - 1
                #if n_turn >= 4:
                #    boards[rv.player].append(board_str + ' ' + str(y * hw + x))
                rv.move(y, x)
                n_turn += 1
            nums = [0, 0]
            for y in range(hw):
                for x in range(hw):
                    if rv.grid[y][x] >= 0:
                        nums[rv.grid[y][x]] += 1
            score = 1 if nums[0] > nums[1] else -1 if nums[0] < nums[1] else 0
            if mode == [0, 1]:
                if score == -1:
                    new_w_win += 1
                elif score == 1:
                    best_b_win += 1
            else:
                if score == 1:
                    new_b_win += 1
                elif score == -1:
                    best_w_win += 1
            for i in range(2):
                ais[i].kill()
            all_idx += 1
            print('\r', end=' ')
            print(all_idx, end=' ')
            print(round(all_idx / game_num, 2), end=' ')
            all_num = 50
            black_win = max(1, best_b_win + new_b_win)
            print('black', end=' ')
            for _ in range(int(all_num * best_b_win / black_win)):
                print('.', end='')
            for _ in range(int(all_num * new_b_win / black_win)):
                print('=', end='')
            print(' ', end='')
            white_win = max(1, best_w_win + new_w_win)
            print('white', end=' ')
            for _ in range(int(all_num * best_w_win / white_win)):
                print('.', end='')
            for _ in range(int(all_num * new_w_win / white_win)):
                print('=', end='')
            for _ in range(10):
                print(' ', end='')
print('')
print('best black', best_b_win)
print('new  black', new_b_win)
print('')
print('best white', best_w_win)
print('new  white', new_w_win)

if new_b_win > best_b_win and new_w_win > best_w_win:
    print('new won')
else:
    print('best won')