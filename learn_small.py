from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle

dict_data = {}
all_data = []
all_labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []

teacher = load_model('model_big.h5')


hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]
pattern_space = []
board_translate = []

consts = [
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63, 5, 14, 23, 4, 13, 22, 31, 3, 12, 21, 30, 39, 2, 11, 20, 29, 38, 47, 1, 10, 19, 28, 37, 46, 55, 0, 9, 18, 27, 36, 45, 54, 63, 8,
    17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52, 61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 10, 8, 8, 8, 8, 4, 4, 8, 2, 4, 54, 63, 62, 61, 60, 59, 58, 57,
    56, 49, 49, 56, 48, 40, 32, 24, 16, 8, 0, 9, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14, 14, 7, 15, 23, 31, 39, 47, 55, 63, 54, 3, 2, 1, 0, 9, 8, 16, 24, 4, 5, 6, 7, 14, 15, 23, 31, 60, 61, 62, 63, 54, 55, 47, 39, 59, 58, 57, 56, 49, 48, 40, 32, 0, 1, 2, 3, 8, 9, 10, 11, 0, 8, 16, 24, 1, 9, 17, 25, 7, 6, 5, 4, 15, 14, 13, 12, 7, 15, 23, 31, 6, 14, 22, 30, 63, 62, 61, 60,
    55, 54, 53, 52, 63, 55, 47, 39, 62, 54, 46, 38, 56, 57, 58, 59, 48, 49, 50, 51, 56, 48, 40, 32, 57, 49, 41, 33, 0, 9, 18, 27, 36, 45, 54, 63, 7, 14, 21, 28, 35, 42, 49, 56, 0, 1, 2, 3, 4, 5, 6, 7, 7, 15, 23, 31, 39, 47, 55, 63, 63, 62, 61, 60, 59, 58, 57, 56, 56, 48, 40, 32, 24, 26, 8, 0
]

consts_idx = 0
for i in range(board_index_num):
    pattern_space.append(consts[consts_idx])
    consts_idx += 1
for i in range(board_index_num):
    board_translate.append([])
    for j in range(pattern_space[i]):
        board_translate[-1].append(consts[consts_idx])
        consts_idx += 1

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

    def pot_canput(self):
        res_p = 0
        res_o = 0
        for i in range(board_index_num):
            arr = []
            for j in board_translate[i]:
                arr.append(self.grid[j // hw][j % hw])
            t1, t2 = pot_canput_line(arr)
            res_p += t1
            res_o += t2
        return res_p, res_o

def collect_data(s):
    global dict_data
    grids = []
    rv = reversi()
    idx = 2
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        turn = 0 if s[idx] == '+' else 1
        x = ord(s[idx + 1]) - ord('a')
        y = int(s[idx + 2]) - 1
        idx += 3
        if rv.move(y, x):
            print('error')
            break
        pot_canput_p, pot_canput_o = rv.pot_canput()
        pot_canput_p = str(pot_canput_p)
        pot_canput_o = str(pot_canput_o)
        grid_str1 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[i][j] == 0:
                    grid_str1 += '1 '
                else:
                    grid_str1 += '0 '
        grid_str2 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[i][j] == 1:
                    grid_str2 += '1 '
                else:
                    grid_str2 += '0 '
        grid_str3 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[i][j] == -1 or rv.grid[i][j] == 2:
                    grid_str3 += '1 '
                else:
                    grid_str3 += '0 '
        #grids.append([1, grid_str1 + grid_str2])
        #grids.append([-1, grid_str2 + grid_str1])
        grids.append([1, grid_str1 + grid_str2 + pot_canput_p + ' ' + pot_canput_o])
        grids.append([-1, grid_str2 + grid_str1 + pot_canput_o + ' ' + pot_canput_p])

        grid_str1 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[j][i] == 0:
                    grid_str1 += '1 '
                else:
                    grid_str1 += '0 '
        grid_str2 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[j][i] == 1:
                    grid_str2 += '1 '
                else:
                    grid_str2 += '0 '
        grid_str3 = ''
        for i in range(hw):
            for j in range(hw):
                if rv.grid[j][i] == -1 or rv.grid[j][i] == 2:
                    grid_str3 += '1 '
                else:
                    grid_str3 += '0 '
        #grids.append([1, grid_str1 + grid_str2])
        #grids.append([-1, grid_str2 + grid_str1])
        grids.append([1, grid_str1 + grid_str2 + pot_canput_p + ' ' + pot_canput_o])
        grids.append([-1, grid_str2 + grid_str1 + pot_canput_o + ' ' + pot_canput_p])
        
        grid_str1 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[i][j] == 0:
                    grid_str1 += '1 '
                else:
                    grid_str1 += '0 '
        grid_str2 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[i][j] == 1:
                    grid_str2 += '1 '
                else:
                    grid_str2 += '0 '
        grid_str3 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[i][j] == -1 or rv.grid[i][j] == 2:
                    grid_str3 += '1 '
                else:
                    grid_str3 += '0 '
        #grids.append([1, grid_str1 + grid_str2])
        #grids.append([-1, grid_str2 + grid_str1])
        grids.append([1, grid_str1 + grid_str2 + pot_canput_p + ' ' + pot_canput_o])
        grids.append([-1, grid_str2 + grid_str1 + pot_canput_o + ' ' + pot_canput_p])

        grid_str1 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[j][i] == 0:
                    grid_str1 += '1 '
                else:
                    grid_str1 += '0 '
        grid_str2 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[j][i] == 1:
                    grid_str2 += '1 '
                else:
                    grid_str2 += '0 '
        grid_str3 = ''
        for i in reversed(range(hw)):
            for j in reversed(range(hw)):
                if rv.grid[j][i] == -1 or rv.grid[j][i] == 2:
                    grid_str3 += '1 '
                else:
                    grid_str3 += '0 '
        #grids.append([1, grid_str1 + grid_str2])
        #grids.append([-1, grid_str2 + grid_str1])
        grids.append([1, grid_str1 + grid_str2 + pot_canput_p + ' ' + pot_canput_o])
        grids.append([-1, grid_str2 + grid_str1 + pot_canput_o + ' ' + pot_canput_p])

        if rv.end():
            break
    rv.check_pass()
    #score = 1 if rv.nums[0] > rv.nums[1] else 0 if rv.nums[0] == rv.nums[1] else -1
    score = rv.nums[0] - rv.nums[1]
    for sgn, grid in grids:
        if grid in dict_data:
            dict_data[grid][0] += sgn * score
            dict_data[grid][1] += 1
        else:
            dict_data[grid] = [sgn * score, 1]

def reshape_data():
    global all_data, all_labels
    for grid_str in dict_data.keys():
        grid = [int(i) for i in grid_str.split()]
        #score = dict_data[grid_str][0] / dict_data[grid_str][1]
        all_data.append(grid)
        #all_labels.append(score)
    #all_data = np.array(all_data)
    #all_labels = np.array(all_labels)

def divide_data(ratio):
    global train_data, train_labels, test_data, test_labels
    test_num = int(len(all_data) * ratio)
    idxes = list(range(len(all_data)))
    shuffle(idxes)
    for i in range(test_num):
        test_data.append(all_data[idxes[i]])
        #test_labels.append(all_labels[idxes[i]])
    for i in range(test_num, len(all_data)):
        train_data.append(all_data[idxes[i]])
        #train_labels.append(all_labels[idxes[i]])
    train_data = np.array(train_data)
    #train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    #test_labels = np.array(test_labels)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    print('mean', mean)
    print('std', std)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    train_labels = teacher.predict(train_data).flatten()
    test_labels = teacher.predict(test_data).flatten()

data_num = 1000
with open('third_party/xxx.gam', 'rb') as f:
    raw_data = f.read()
games = [i for i in raw_data.splitlines()]
for i in trange(data_num):
    collect_data(str(games[i]))
reshape_data()
divide_data(0.1)
print('train', train_data.shape, train_labels.shape)
print('test', test_data.shape, test_labels.shape)
model = Sequential()
model.add(Dense(128, input_shape=(130,)))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.0625))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=1000, validation_split=0.2, callbacks=[early_stop])
'''
with open('param/param.txt', 'w') as f:
    for i in (0, 3):
        for item in model.layers[i].weights[1].numpy():
            f.write(str(item) + '\n')
    for i in (0, 3, 5):
        for arr in model.layers[i].weights[0].numpy():
            for item in arr:
                f.write(str(item) + '\n')
'''
'''
for layer_num, layer in enumerate(model.layers):
    print(layer.weights[0].numpy())
    with open('param/w' + str(layer_num + 1) + '.txt', 'w') as f:
        for i in range(len(layer.weights[0].numpy())):
            if len(layer.weights[0].numpy()[i]) > 1:
                f.write('{')
            for j in range(len(layer.weights[0].numpy()[i])):
                if j == len(layer.weights[0].numpy()[i]) - 1 and len(layer.weights[0].numpy()[i]) > 1:
                    f.write(str(layer.weights[0].numpy()[i][j]))
                else:
                    f.write(str(layer.weights[0].numpy()[i][j]) + ',')
            if len(layer.weights[0].numpy()[i]) > 1:
                if i == len(layer.weights[0].numpy()) - 1:
                    f.write('}')
                else:
                    f.write('},\n')
    if layer_num == len(model.layers) - 1:
        break
    print(layer.weights[1].numpy())
    with open('param/b' + str(layer_num + 1) + '.txt', 'w') as f:
        f.write('{')
        for i in range(len(layer.weights[1].numpy())):
            if i == len(layer.weights[1].numpy()) - 1:
                f.write(str(layer.weights[1].numpy()[i]) + '}')
            else:
                f.write(str(layer.weights[1].numpy()[i]) + ',')
'''
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
test_loss, test_mae = model.evaluate(test_data, test_labels)
print('result', test_loss, test_mae)
test_num = 10
test_num = min(test_labels.shape[0], test_num)
test_predictions = model.predict(test_data[0:test_num]).flatten()
print([round(i, 2) for i in test_labels[0:test_num]])
print([round(i, 2) for i in test_predictions[0:test_num]])
for i in range(5):
    print(list(test_data[i]))