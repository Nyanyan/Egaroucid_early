from tqdm import trange
import subprocess

book_num = 30

data_dict = {}
data_proc = []

hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

board_maker = subprocess.Popen('./make_board.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

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
    global data_dict
    score = -1000
    grids = []
    with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
        score = float(f.readline()) / 64.0
        result = 1 if score > 0 else 0 if score == 0 else -1
        all_ln = int(f.readline())
        ln = min(book_num // 2, all_ln)
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            coord = y * hw + x
            if not s in data_dict:
                data_dict[s] = [[0, 0] for _ in range(hw2)]
            data_dict[s][coord][0] += result
            data_dict[s][coord][1] += 1
        for _ in range(ln, all_ln):
            f.readline()
        ln = min(book_num // 2, int(f.readline()))
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            coord = y * hw + x
            if not s in data_dict:
                data_dict[s] = [[0, 0] for _ in range(hw2)]
            data_dict[s][coord][0] -= result
            data_dict[s][coord][1] += 1

def make_board():
    global data_proc
    for key in data_dict:
        board_maker.stdin.write(key.encode('utf-8'))
        board_maker.stdin.flush()
        board = board_maker.stdout.readline().decode().strip()
        policy = -1
        rate = -100.0
        for i in range(hw2):
            if data_dict[key][i][1] == 0:
                continue
            tmp = data_dict[key][i][0] / data_dict[key][i][1]
            if rate < tmp:
                rate = tmp
                policy = i
        data_proc.append([board, policy, rate])

game_num = 11000
game_strt = 0
print('loading data from files')
for i in trange(game_strt, game_strt + game_num):
    collect_data(i)
print(len(data_dict))
make_board()
board_maker.kill()
print(len(data_proc))
with open('param/book.txt', 'w') as f:
    f.write(str(len(data_proc)) + '\n')
    for board_str, policy, rate in data_proc:
        for elem in board_str.split():
            f.write(elem + '\n')
        f.write(str(policy) + '\n')
        f.write(str(rate) + '\n')