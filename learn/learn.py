import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import comb, exp
import datetime

hw = 8
hw2 = 64

all_data = {}

n_epochs = 200
game_num = 2000
test_ratio = 0.2
n_boards = 3

kernel_size = 3
n_kernels = 40
n_residual = 1

leakyrelu_alpha = 0.01

train_board = None
train_edge = None
train_policies = None
train_value = None

test_raw_board = []
test_board = None
test_edge = None
test_policies = None
test_value = None

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def join_yx(y, x):
    return y * hw + x

def calc_idx(i, j, rnd):
    if rnd == 0:
        return join_yx(i, j)
    elif rnd == 1:
        return join_yx(j, hw - 1 - i)
    elif rnd == 2:
        return join_yx(hw - 1 - i, hw - 1 - j)
    else:
        return join_yx(hw - 1 - j, i)

def collect_data(num):
    global all_data
    try:
        with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        return
    for datum in data:
        board, policy, score = datum.split()
        policy = int(policy)
        score = float(score)
        #print(board, policy, score)
        all_data.append([board, policy, score])

def reshape_data_train():
    global train_board, train_policies, train_value, train_edge
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    train_idx = 0
    for ii in trange(ln):
        board, policies, score = tmp_data[ii]
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = []
        grid_space1 = []
        grid_space_vacant = []
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0.append(1 if board[idx] == '0' else 0)
                grid_space1.append(1 if board[idx] == '1' else 0)
                grid_space_vacant.append(1 if board[idx] == '.' else 0)
                stone_num += 1 if board[idx] != '.' else 0
        if stone_num < 14 or stone_num > 56:
            continue
        grid_flat = grid_space0
        grid_flat.extend(grid_space1)
        grid_flat.extend(grid_space_vacant)
        for i in range(hw):
            for j in range(hw):
                for k in range(n_boards):
                    train_board[train_idx][i][j][k] = grid_flat[k * hw2 + j * hw + i]
        #for i, elem in zip(range(15), param.split()):
        #    train_param[train_idx][i] = float(elem)
        for i in range(hw2):
            train_policies[train_idx][i] = policies[i]
        train_value[train_idx] = score
        train_idx += 1
    train_board = train_board[0:train_idx]
    #train_param = train_param[0:train_idx]
    train_policies = train_policies[0:train_idx]
    train_value = train_value[0:train_idx]
    #mean = train_param.mean(axis=0)
    #std = train_param.std(axis=0)
    #print('mean', mean)
    #print('std', std)
    #train_param = (train_param - mean) / std
    '''
    print(train_board[0])
    print(train_param[0])
    print(train_policies[0])
    print(train_value[0])
    '''
    print('train', train_board.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_policies, test_value, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        #my_evaluate.stdin.write(board.encode('utf-8'))
        #my_evaluate.stdin.flush()
        #additional_data = my_evaluate.stdout.readline().decode().strip()
        #tmp_data.append([board, additional_data, policies, score])
        tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    test_idx = 0
    for ii in trange(ln):
        board, policies, score = tmp_data[ii]
        test_raw_board.append(board)
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = []
        grid_space1 = []
        grid_space_vacant = []
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0.append(1 if board[idx] == '0' else 0)
                grid_space1.append(1 if board[idx] == '1' else 0)
                grid_space_vacant.append(1 if board[idx] == '.' else 0)
                stone_num += 1 if board[idx] != '.' else 0
        if stone_num < 14 or stone_num > 56:
            continue
        grid_flat = grid_space0
        grid_flat.extend(grid_space1)
        grid_flat.extend(grid_space_vacant)
        for i in range(hw):
            for j in range(hw):
                for k in range(n_boards):
                    test_board[test_idx][i][j][k] = grid_flat[k * hw2 + j * hw + i]
        for i in range(hw2):
            test_policies[test_idx][i] = policies[i]
        test_value[test_idx] = score
        test_idx += 1
    test_board = train_board[0:test_idx]
    test_policies = train_policies[0:test_idx]
    test_value = train_value[0:test_idx]
    print('test', test_board.shape, test_policies.shape, test_value.shape)

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)


inputs = Input(shape=(hw, hw, n_boards,))
x1 = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(inputs)
x1 = LeakyReLU(x1)
for _ in range(n_residual):
    sc = x1
    x1 = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(x1)
    x1 = Add()([x1, sc])
    x1 = LeakyReLU(x1)
x1 = GlobalAveragePooling2D()(x1)

yp = Activation('tanh')(x1)
yp = Dense(hw2)(yp)
yp = Activation('softmax', name='policy')(yp)

yv = Dense(32)(x1)
yv = LeakyReLU(yv)
yv = Dense(1)(yv)
yv = Activation('tanh', name='value')(yv)

model = Model(inputs=inputs, outputs=[yp, yv])

#model = load_model('param/model.h5')
#plot_model(model, to_file='graph/model.png', show_shapes=True, expand_nested=True)

model.summary()

print('collecting data')

n_train_data = int(game_num * (1.0 - test_ratio))
n_test_data = int(game_num * test_ratio)

idxes = list(range(game_num + 100))
shuffle(idxes)

all_data = []
for i in trange(n_train_data):
    collect_data(idxes[i])
train_board = np.zeros((len(all_data), hw, hw, n_boards))
train_edge = np.zeros((len(all_data), hw, 1, 12))
train_policies = np.zeros((len(all_data), hw2))
train_value = np.zeros(len(all_data))
reshape_data_train()

all_data = []
for i in trange(n_train_data, game_num):
    collect_data(idxes[i])
test_board = np.zeros((len(all_data), hw, hw, n_boards))
test_edge = np.zeros((len(all_data), hw, 1, 12))
test_policies = np.zeros((len(all_data), hw2))
test_value = np.zeros(len(all_data))
reshape_data_test()

model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
print(model.evaluate([train_board], [train_policies, train_value]))
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_board, [train_policies, train_value], epochs=n_epochs, validation_data=(test_board, [test_policies, test_value]), callbacks=[early_stop])

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
#model.save('param/additional_learn_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')
model.save('param/model.h5')

for key in ['policy_loss', 'val_policy_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('policy loss')
plt.legend(loc='best')
plt.savefig('graph/policy_loss.png')
plt.clf()

for key in ['value_loss', 'val_value_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('value loss')
plt.legend(loc='best')
plt.savefig('graph/value_loss.png')
plt.clf()

all_data = []
for i in trange(game_num, game_num + 100):
    collect_data(idxes[i])
test_raw_board = []
test_board = []
test_policies = []
test_value = []
reshape_data_test()

print(model.evaluate(test_board, [test_policies, test_value]))

i = 0
while True:
    try:
        #print(i, model.layers[i])
        dammy = model.layers[i]
        j = 0
        while True:
            try:
                print(model.layers[i].weights[j].shape)
                j += 1
            except:
                break
        i += 1
    except:
        break

prediction = model.predict(test_board)
gap = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
gap_prediction = gap.predict(test_board)
for i in range(10):
    print(test_raw_board[i])
    print(gap_prediction[i])
    mx = 0.0
    policy = -1
    for j in range(hw2):
        #print(prediction[0][i][j], end=' ')
        if mx < prediction[0][i][j]:
            mx = prediction[0][i][j]
            policy = j
    #print('')
    print(policy, mx)
    print(prediction[1][i])
    print('')
