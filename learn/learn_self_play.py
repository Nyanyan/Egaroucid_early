import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp
from os import rename, path, listdir
from time import time
import datetime

selfplay_num = 5
num_self_play_in_one_time_train = 100
num_self_play_in_one_time_test = 20
num_of_decide = 100
n_epochs = 100

hw = 8
hw2 = 64

my_evaluate = subprocess.Popen('./evaluation.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

all_data = []

train_board = []
train_param = []
train_policies = []
train_value = []

test_raw_board = []
test_board = []
test_param = []
test_policies = []
test_value = []

mean = []
std = []

with open('param/mean.txt', 'r') as f:
    mean = np.array([float(i) for i in f.read().splitlines()])
with open('param/std.txt', 'r') as f:
    std = np.array([float(i) for i in f.read().splitlines()])


early_stages = []

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def self_play(num_self_play_in_one_time):
    global all_data
    strt = time()
    sp = []
    one_play_num = num_self_play_in_one_time // selfplay_num
    for i in range(selfplay_num):
        seed = randint(1, 2000000000)
        sp.append(subprocess.Popen(('./self_play.out ' + str(seed) + ' ' + str(one_play_num)).split(), stdout=subprocess.PIPE))
    n_record = sum(path.isfile(path.join('self_play/records', name)) for name in listdir('self_play/records'))
    records = []
    for i in range(selfplay_num):
        idx = 0
        result = sp[i].communicate()[0].decode().splitlines()
        for num in range(one_play_num * 2):
            ln = int(result[idx])
            idx += 1
            score = float(result[idx])
            idx += 1
            for _ in range(ln):
                board = result[idx]
                idx += 1
                policy = int(result[idx])
                idx += 1
                all_data.append([board, policy, score])
        records.extend(result[idx:])
        sp[i].kill()
    with open('self_play/records/' + digit(n_record + i, 7) + '.txt', 'w') as f:
        for record in records:
            f.write(record + '\n')
    print(time() - strt)

def reshape_data_train():
    global train_board, train_param, train_policies, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for i in trange(len(all_data)):
        board, policy, score = all_data[i]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policy, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    for ii in trange(ln):
        board, param, policy, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        train_param.append([float(i) for i in param.split()])
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        train_policies.append(policies)
        train_value.append(score)
    train_board = np.array(train_board)
    train_param = np.array(train_param)
    train_policies = np.array(train_policies)
    train_value = np.array(train_value)
    #mean = train_param.mean(axis=0)
    #std = train_param.std(axis=0)
    train_param = (train_param - mean) / std
    print('train', train_board.shape, train_param.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_param, test_policies, test_value, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for i in trange(len(all_data)):
        board, policy, score = all_data[i]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policy, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    for ii in trange(ln):
        board, param, policy, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
        test_raw_board.append(board)
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        test_param.append([float(i) for i in param.split()])
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    test_param = np.array(test_param)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    test_param = (test_param - mean) / std
    print('test', test_board.shape, test_param.shape, test_policies.shape, test_value.shape)

hw = 8
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

def decide(num):
    ais = [subprocess.Popen('./decide.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
    ais[0].stdin.write('0\n'.encode('utf-8')) # best
    ais[1].stdin.write('1\n'.encode('utf-8')) # 
    new_win = 0
    for game_idx in trange(num):
        rv = reversi()
        player2ai = [0, 1] if game_idx % 2 == 0 else [1, 0]
        '''
        rnd = randint(0, len(early_stages) - 1)
        for y in range(hw):
            for x in range(hw):
                rv.grid[y][x] = early_stages[rnd][y][x]
        '''
        for y in range(hw):
            for x in range(hw):
                rv.grid[y][x] = -1
        rv.grid[3][3] = 1
        rv.grid[3][4] = 0
        rv.grid[4][3] = 0
        rv.grid[4][4] = 0
        rv.grid[4][5] = 0
        rv.player = 1
        while True:
            if rv.check_pass() and rv.check_pass():
                break
            stdin = ''
            for y in range(hw):
                for x in range(hw):
                    stdin += '0' if rv.grid[y][x] == rv.player else '1' if rv.grid[y][x] == 1 - rv.player else '.'
            stdin += '\n'
            #print(stdin)
            ais[player2ai[rv.player]].stdin.write(stdin.encode('utf-8'))
            ais[player2ai[rv.player]].stdin.flush()
            y, x, _ = [float(i) for i in ais[player2ai[rv.player]].stdout.readline().decode().strip().split()]
            y = int(y)
            x = int(x)
            rv.move(y, x)
            if rv.end():
                break
        rv.check_pass()
        #rv.output()
        if rv.nums[player2ai[0]] < rv.nums[player2ai[1]]:
            new_win += 1
    for i in range(2):
        ais[i].kill()
    print('score of new player', new_win)
    if new_win > num * 0.55:
        return 1
    else:
        return 0

def get_early_stages():
    global early_stages
    all_data = None
    with open('param/early_stage.txt', 'r') as f:
        all_data = f.read().splitlines()
    for grid_str in all_data[1:]:
        n_stones = 0
        for elem in grid_str:
            n_stones += elem != '.'
        if n_stones == 10:
            grid = [[-1 for _ in range(hw)] for _ in range(hw)]
            for y in range(hw):
                for x in range(hw):
                    grid[y][x] = 0 if grid_str[y * hw + x] == '0' else 1 if grid_str[y * hw + x] == '1' else -1
            early_stages.append(grid)
    print('len early stages', len(early_stages))

#get_early_stages()


model_updated = True

while True:
    
    #if model_updated:
    all_data = []
    train_board = []
    train_param = []
    train_policies = []
    train_value = []
    print('loading train data')
    self_play(num_self_play_in_one_time_train)
    reshape_data_train()
    all_data = []
    test_raw_board = []
    test_board = []
    test_param = []
    test_policies = []
    test_value = []
    print('loading test data')
    self_play(num_self_play_in_one_time_test)
    reshape_data_test()

    print('start learning')
    model = load_model('param/best.h5')
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Adam(lr=0.0001))
    print(model.evaluate([train_board, train_param], [train_policies, train_value]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit([train_board, train_param], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop])

    print('saving')
    model.save('param/model_new.h5')
    model.save_weights('param/model_new.hdf5')
    '''
    with open('param/mean_new.txt', 'w') as f:
        for i in range(mean.shape[0]):
            f.write(str(mean[i]) + '\n')
    with open('param/std_new.txt', 'w') as f:
        for i in range(std.shape[0]):
            f.write(str(std[i]) + '\n')
    '''
    with open('param/param_new.txt', 'w') as f:
        i = 0
        while True:
            try:
                #print(i, model.layers[i])
                dammy = model.layers[i]
                j = 0
                while True:
                    try:
                        print(model.layers[i].weights[j].shape)
                        if len(model.layers[i].weights[j].shape) == 4:
                            for ll in range(model.layers[i].weights[j].shape[3]):
                                for kk in range(model.layers[i].weights[j].shape[2]):
                                    for jj in range(model.layers[i].weights[j].shape[1]):
                                        for ii in range(model.layers[i].weights[j].shape[0]):
                                            f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 2:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 1:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                        j += 1
                    except:
                        break
                i += 1
            except:
                break
    
    print('decision')
    decision = decide(num_of_decide)
    if decision == 0:
        print('best won')
        model_updated = False
    elif decision == 1:
        print('new won')
        model_updated = True
        model.save('param/best.h5')
        now = datetime.datetime.today()
        print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
        model.save('param/selfplay_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')
        with open('param/param_new.txt', 'r') as f:
            new_params = f.read()
        with open('param/param.txt', 'w') as f:
            f.write(new_params)

my_evaluate.kill()