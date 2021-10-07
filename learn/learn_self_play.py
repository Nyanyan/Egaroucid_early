import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
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

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)

selfplay_num = 8
num_self_play_in_one_time_train = 240
num_self_play_in_one_time_test = 160
decide_num = 160 // 2
n_epochs = 10

hw = 8
hw2 = 64

#my_evaluate = subprocess.Popen('./evaluation.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

all_data = []

train_board = []
train_policies = []
train_value = []

test_raw_board = []
test_board = []
test_policies = []
test_value = []

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
            for _ in range(ln):
                board = result[idx]
                idx += 1
                policies = [float(elem) for elem in result[idx].split()]
                idx += 1
                score = float(result[idx])
                idx += 1
                all_data.append([board, policies, score])
        records.extend(result[idx:])
        sp[i].kill()
    with open('self_play/records/' + digit(n_record + i, 7) + '.txt', 'w') as f:
        for record in records:
            f.write(record + '\n')
    print(time() - strt)

def reshape_data_train():
    global train_board, train_param, train_policies, train_value
    shuffle(all_data)
    ln = len(all_data)
    print('got', ln)
    print('creating train data & labels')
    for ii in trange(ln):
        board, policies, score = all_data[ii]
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
        train_policies.append(policies)
        train_value.append(score)
    train_board = np.array(train_board)
    train_policies = np.array(train_policies)
    train_value = np.array(train_value)
    print('train', train_board.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_param, test_policies, test_value
    shuffle(all_data)
    ln = len(all_data)
    print('got', ln)
    print('creating train data & labels')
    for ii in trange(ln):
        board, policies, score = all_data[ii]
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
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    print('test', test_board.shape, test_policies.shape, test_value.shape)

def decide():
    sp = []
    one_play_num = decide_num // selfplay_num
    for i in range(selfplay_num):
        sp.append(subprocess.Popen('python decide.py'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE))
        stdin_str = str(one_play_num) + '\n'
        sp[i].stdin.write(stdin_str.encode('utf-8'))
        sp[i].stdin.flush()
    best_score = 0
    new_score = 0
    for i in range(selfplay_num):
        bp, np = [int(elem) for elem in sp[i].communicate()[0].decode().split()]
        #print(bp, np)
        best_score += bp
        new_score += np
    print('')
    print('best', best_score)
    print('new ', new_score)
    if new_score >= best_score:
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
        if n_stones == 8:
            early_stages.append(grid_str)
            '''
            grid = [[-1 for _ in range(hw)] for _ in range(hw)]
            for y in range(hw):
                for x in range(hw):
                    grid[y][x] = 0 if grid_str[y * hw + x] == '0' else 1 if grid_str[y * hw + x] == '1' else -1
            early_stages.append(grid)
            '''
    print('len early stages', len(early_stages))

#get_early_stages()
#print(decide())

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
    #model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    print(model.evaluate(train_board, [train_policies, train_value]))
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_board, [train_policies, train_value], epochs=n_epochs, validation_data=(test_board, [test_policies, test_value]), callbacks=[early_stop])

    print('saving')
    model.save('param/model_new.h5')
    #model.save('param/best.h5')
    
    with open('param/param_new.txt', 'w') as f:
        i = 0
        while True:
            try:
                print(i, model.layers[i])
                j = 0
                while True:
                    try:
                        print(model.layers[i].weights[j].shape)
                        if len(model.layers[i].weights[j].shape) == 4:
                            for ll in range(model.layers[i].weights[j].shape[3]):
                                for kk in range(model.layers[i].weights[j].shape[2]):
                                    for jj in range(model.layers[i].weights[j].shape[1]):
                                        for ii in range(model.layers[i].weights[j].shape[0]):
                                            f.write('{:.5f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 2:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    f.write('{:.5f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 1:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                f.write('{:.5f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                        j += 1
                    except:
                        break
                i += 1
            except:
                break
    '''
    with open('param/param.txt', 'w') as f:
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
    now = datetime.datetime.today()
    print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
    model.save('param/selfplay_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')
    '''
    print('decision')
    decision = decide()
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