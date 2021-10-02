from tqdm import trange
import subprocess

book_num = 7

early_stages = [[] for _ in range(book_num)]

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
    ais = [subprocess.Popen('./ai_probability.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
    win0 = 0
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
            y, x, _ = [int(i) for i in ais[player2ai[rv.player]].stdout.readline().decode().strip().split()]
            rv.move(y, x)
            if rv.end():
                break
        rv.check_pass()
        #rv.output()
        if rv.nums[player2ai[0]] > rv.nums[player2ai[1]]:
            win0 += 1
        elif rv.nums[player2ai[0]] < rv.nums[player2ai[1]]:
            win0 -= 1
    for i in range(2):
        ais[i].kill()
    print('score of best player', best_win)
    if best_win > 0:
        return 0
    else:
        return 1

def get_early_stages():
    global early_stages
    all_data = None
    with open('param/early_stage.txt', 'r') as f:
        all_data = f.read().splitlines()
    for grid_str in all_data[1:]:
        n_stones = 0
        for elem in grid_str:
            n_stones += elem != '.'
        grid = [[-1 for _ in range(hw)] for _ in range(hw)]
        for y in range(hw):
            for x in range(hw):
                grid[y][x] = 0 if grid_str[y * hw + x] == '0' else 1 if grid_str[y * hw + x] == '1' else -1
        early_stages[n_stones - 4].append(grid)
    print('len early stages', len(early_stages))

#get_early_stages()





book_num = 10

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


def collect_data(s):
    rv = reversi()
    score = None
    policy = None
    idx = 0
    while True:
        if idx >= len(s):
            return
        if rv.check_pass() and rv.check_pass():
            break
        x = ord(s[idx]) - ord('A')
        y = int(s[idx + 1]) - 1
        y = hw - 1 - y
        x = hw - 1 - x
        if s[idx + 2] == ' ':
            score = round(float(s[idx + 5:]))
            policy = y * hw + x
            break
        idx += 2
        if rv.move(y, x):
            print('error')
            break
    grid_str = ''
    for i in range(hw):
        for j in range(hw):
            grid_str += '0' if rv.grid[i][j] == rv.player else '1' if rv.grid[i][j] == 1 - rv.player else '.' # 0 to move
    if rv.player != 0:
        score = -score
    #score /= hw2
    score=  round(score)
    result = 1 if score > 0 else -1 if score < 0 else 0
    return grid_str, result, policy


def make_board():
    global data_proc
    for key in data_dict:
        board_maker.stdin.write(key.encode('utf-8'))
        board_maker.stdin.flush()
        board = board_maker.stdout.readline().decode().strip()
        policy = -1
        rate = -100.0
        for i in range(hw2):
            if data_dict[key][i][1] < 2:
                continue
            tmp = data_dict[key][i][0] / data_dict[key][i][1]
            if rate < tmp:
                rate = tmp
                policy = i
        if policy == -1:
            continue
        data_proc.append([board, policy, rate])

with open('param/records.txt', 'r') as f:
    raw_data = f.read()
games = [i for i in raw_data.splitlines()]
print(len(games))
already_unzip = 0
with open('param/book.txt', 'r') as f:
    already_unzip = len(f.read().splitlines())
print(already_unzip)
already_unzip //= 10
print(already_unzip)
if input('sure?:') != 'yes':
    exit()
board_maker = subprocess.Popen('./make_board.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
with open('param/book.txt', 'a') as f:
    for i in trange(already_unzip, len(games)):
        board, value, policy = collect_data(games[i])
        board_maker.stdin.write(board.encode('utf-8'))
        board_maker.stdin.flush()
        board_idxes = [j for j in board_maker.stdout.readline().decode().strip().split()]
        for j in board_idxes:
            f.write(j + '\n')
        f.write(str(value) + '\n')
        f.write(str(policy) + '\n')
board_maker.kill()