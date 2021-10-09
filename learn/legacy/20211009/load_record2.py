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

def collect_data(boards, params):
    param_idx = 0
    data = []
    for board_idx in range(len(boards)):
        while True:
            if param_idx >= len(params):
                print(board_idx)
                return data
            if ':' in params[param_idx]:
                break
            param_idx += 1
        param_split = params[param_idx].split()
        raw_policy = param_split[0][:2]
        raw_score = float(param_split[0][3:])
        policy = (int(raw_policy[1]) - 1) * hw + ord(raw_policy[0]) - ord('A')
        score = round(raw_score)
        score = 1.0 if score >= 1 else -1.0 if score <= -1 else 0.0
        board = boards[board_idx].split()[0]
        board = board.replace('O', '0').replace('*', '1')
        #print(board, policy, score, param_idx)
        #print(boards[board_idx])
        data.append([board, policy, score])
        param_idx += 1
    print(board_idx)
    return data





for num in range(4):
    boards = []
    params = []
    with open('boards/' + digit(num, 7) + '.txt', 'r') as f:
        boards = list(f.read().splitlines())
    with open('boards_mr/' + digit(num, 7) + '.txt', 'r') as f:
        params = list(f.read().splitlines())
    data = collect_data(boards, params)
    with open('learn_data2/' + digit(num, 7) + '.txt', 'w') as f:
        for board, policy, score in data:
            f.write(board + ' ' + str(policy) + ' ' + str(score) + '\n')
