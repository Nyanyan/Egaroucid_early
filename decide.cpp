#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx")

// Reversi AI C++ version 6
// use deep reinforcement learning

#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <unordered_map>
#include <random>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65

#define inf 100000.0
#define board_index_num 38

#define char_s 35
#define char_e 91
#define num_s 93
#define num_e 126

#define hash_table_size 16384
#define hash_mask (hash_table_size - 1)

#define evaluate_count 5
#define c_puct 50.0
#define c_end 1.0

#define n_board_input 3
#define n_add_input 11
#define kernel_size 3
#define n_kernels 16
#define n_residual 2
#define n_dense0 16
#define n_dense1 16
#define n_dense2 32
#define n_joined (n_kernels + n_dense2)
#define conv_size (hw_p1 - kernel_size)
#define conv_start (-(kernel_size / 2))
#define div_pooling (hw2)
#define conv_padding ((hw - conv_size) / 2)
#define epsilon 0.001

struct board_param{
    unsigned long long trans[board_index_num][6561][hw];
    unsigned long long neighbor8[board_index_num][6561][hw];
    bool legal[6561][hw];
    int put[hw2][board_index_num];
    int board_translate[board_index_num][8];
    int board_rev_translate[hw2][4][2];
    int pattern_space[board_index_num];
    int reverse[6561];
    int pow3[15];
    int rev_bit3[6561][8];
    int pop_digit[6561][8];
    int digit_pow[3][10];
    int put_idx[hw2][10];
    int put_idx_num[hw2];
    int restore_p[6561][hw], restore_o[6561][hw], restore_vacant[6561][hw];
};

struct eval_param{
    double weight[hw2];
    double avg_canput[hw2];
    int canput[6561];
    int cnt_p[6561], cnt_o[6561];
    double weight_p[hw][6561], weight_o[hw][6561];
    int confirm_p[6561], confirm_o[6561];
    int pot_canput_p[6561], pot_canput_o[6561];

    double mean[n_add_input];
    double std[n_add_input];
    double input_b[n_board_input][hw][hw];
    double input_p[n_add_input];
    double conv1[n_kernels][n_board_input][kernel_size][kernel_size];
    double conv_residual[n_residual][n_kernels][n_kernels][kernel_size][kernel_size];
    double hidden_conv1[n_kernels][hw][hw];
    double hidden_conv2[n_kernels][hw][hw];
    double hidden_gap0[n_kernels];
    double dense0[n_kernels][n_dense0];
    double bias0[n_kernels];
    double hidden_joined[n_joined];
    double dense1[n_add_input][n_dense1];
    double bias1[n_dense1];
    double hidden_dense1[n_dense1];
    double dense2[n_dense1][n_dense2];
    double bias2[n_dense2];
    double dense3[n_joined][hw2];
    double bias3[hw2];
    double dense4[n_joined];
    double bias4;
};

struct book_elem{
    int policy;
    double rate;
};

struct hash_pair {
    static size_t m_hash_pair_random;
    template<class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        size_t seed = 0;
        seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= m_hash_pair_random + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
size_t hash_pair::m_hash_pair_random = (size_t) random_device()();

struct search_param{
    int max_depth;
    int min_max_depth;
    int strt, tl;
    int turn;
    int searched_nodes;
    vector<int> vacant_lst;
    int vacant_cnt;
    int weak_mode;
    int win_num;
    int lose_num;
    int n_playout;
    unordered_map<pair<unsigned long long, unsigned long long>, book_elem, hash_pair> book;
};

struct board_priority_move{
    int b[board_index_num];
    double priority;
    int move;
    double open_val;
};

struct board_priority{
    int b[board_index_num];
    double priority;
    double n_open_val;
};

struct open_vals{
    double p_open_val, o_open_val;
    int p_cnt, o_cnt;
};

struct mcts_node{
    int board[board_index_num];
    int children[hw2_p1];
    double p[hw2];
    double w;
    int n;
    bool pass;
    bool expanded;
};

struct mcts_param{
    mcts_node nodes[60 * 10 * evaluate_count];
    int used_idx;
};

struct predictions{
    double policies[hw2];
    double value;
};

board_param board_param;
eval_param eval_param;
search_param search_param;
mcts_param mcts_param;

int xorx=123456789, xory=362436069, xorz=521288629, xorw=88675123;
inline double myrandom(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = xorw=(xorw^(xorw>>19))^(t^(t>>8));
    return (double)(xorw) / 2147483648.0;
}

inline int myrandom_int(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = xorw=(xorw^(xorw>>19))^(t^(t>>8));
    return xorw;
}

inline int tim(){
    return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
}

void print_board_line(int tmp){
    int j;
    for (j = 0; j < hw; ++j){
        if (tmp % 3 == 0){
            cerr << ". ";
        }else if (tmp % 3 == 1){
            cerr << "P ";
        }else{
            cerr << "O ";
        }
        tmp /= 3;
    }
}

void print_board(int* board){
    int i, j, idx, tmp;
    for (i = 0; i < hw; ++i){
        tmp = board[i];
        for (j = 0; j < hw; ++j){
            if (tmp % 3 == 0){
                cerr << ". ";
            }else if (tmp % 3 == 1){
                cerr << "P ";
            }else{
                cerr << "O ";
            }
            tmp /= 3;
        }
        cerr << endl;
    }
    cerr << endl;
}

int reverse_line(int a) {
    int res = 0;
    for (int i = 0; i < hw; ++i) {
        res <<= 1;
        res |= 1 & (a >> i);
    }
    return res;
}

inline int check_mobility(const int p, const int o){
	int p1 = p << 1;
    int res = ~(p1 | o) & (p1 + o);
    int p_rev = reverse_line(p), o_rev = reverse_line(o);
    int p2 = p_rev << 1;
    res |= reverse_line(~(p2 | o_rev) & (p2 + o_rev));
    res &= ~(p | o);
    // cerr << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(res) << endl;
    return res;
}

int trans(int pt, int k) {
    if (k == 0)
        return pt >> 1;
    else
        return pt << 1;
}

int move_line(int p, int o, const int place) {
    int rev = 0;
    int rev2, mask, tmp;
    int pt = 1 << place;
    for (int k = 0; k < 2; ++k) {
        rev2 = 0;
        mask = trans(pt, k);
        while (mask && (mask & o)) {
            rev2 |= mask;
            tmp = mask;
            mask = trans(tmp, k);
            if (mask & p)
                rev |= rev2;
        }
    }
    // cerr << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(rev | pt) << endl;
    return rev | pt;
}

int create_p(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 1){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int create_o(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 2){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int board_reverse(int idx){
    int p = create_p(idx);
    int o = create_o(idx);
    int res = 0;
    for (int i = hw_m1; i >= 0; --i){
        res *= 3;
        if (1 & (p >> i))
            res += 2;
        else if (1 & (o >> i))
            ++res;
    }
    return res;
}

void init(){
    int strt = tim();
    int i, j, k, l;
    static int translate[hw2] = {
        0, 1, 2, 3, 3, 2, 1, 0,
        1, 4, 5, 6, 6, 5, 4, 1,
        2, 5, 7, 8, 8, 7, 5, 2,
        3, 6, 8, 9, 9, 8, 6, 3,
        3, 6, 8, 9, 9, 8, 6, 3,
        2, 5, 7, 8, 8, 7, 5, 2,
        1, 4, 5, 6, 6, 5, 4, 1,
        0, 1, 2, 3, 3, 2, 1, 0
    };
    const double params[10] = {
        0.2880, -0.1150, 0.0000, -0.0096,
                -0.1542, -0.0288, -0.0288,
                        0.0000, -0.0096,
                                -0.0096,
    };
    const int consts[476] = {
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63, 5, 14, 23, 4, 13, 22, 31, 3, 12, 21, 30, 39, 2, 11, 20, 29, 38, 47, 1, 10, 19, 28, 37, 46, 55, 0, 9, 18, 27, 36, 45, 54, 63, 8,
        17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52, 61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 10, 8, 8, 8, 8, 4, 4, 8, 2, 4, 54, 63, 62, 61, 60, 59, 58, 57,
        56, 49, 49, 56, 48, 40, 32, 24, 16, 8, 0, 9, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14, 14, 7, 15, 23, 31, 39, 47, 55, 63, 54, 3, 2, 1, 0, 9, 8, 16, 24, 4, 5, 6, 7, 14, 15, 23, 31, 60, 61, 62, 63, 54, 55, 47, 39, 59, 58, 57, 56, 49, 48, 40, 32, 0, 1, 2, 3, 8, 9, 10, 11, 0, 8, 16, 24, 1, 9, 17, 25, 7, 6, 5, 4, 15, 14, 13, 12, 7, 15, 23, 31, 6, 14, 22, 30, 63, 62, 61, 60,
        55, 54, 53, 52, 63, 55, 47, 39, 62, 54, 46, 38, 56, 57, 58, 59, 48, 49, 50, 51, 56, 48, 40, 32, 57, 49, 41, 33, 0, 9, 18, 27, 36, 45, 54, 63, 7, 14, 21, 28, 35, 42, 49, 56, 0, 1, 2, 3, 4, 5, 6, 7, 7, 15, 23, 31, 39, 47, 55, 63, 63, 62, 61, 60, 59, 58, 57, 56, 56, 48, 40, 32, 24, 26, 8, 0
    };
    const string super_compress_pattern = "";
    const double compress_vals[char_e - char_s + 1] = 
        {-0.99191575, -0.955417, -0.925217, -0.87192775, -0.8353087499999999, -0.79376225, -0.7521912222222222, -0.7211734999999999, -0.6842236666666666, -0.6495354444444446, -0.6066062333333334, -0.5705911935483873, -0.5333852142857143, -0.4977529599999999, -0.4617034339622642, -0.4280493521126759, -0.3930658846153848, -0.3562839680851063, -0.32210842748091595, -0.28638591366906474, -0.25082044382022484, -0.2177653593073593, -0.18336263157894744, -0.14849799452054788, -0.11322629255319143, -0.07861064571428576, -0.044194587947882745, -0.009447826356589157, 0.0, 0.02449980906148867, 0.058887281355932165, 0.09310184199134201, 0.1286132636103152, 0.16182661875000015, 0.19795722314049594, 0.23227418264840172, 0.267653596153846, 0.30229703875969, 0.33605759829059817, 0.3711898414634147, 0.40819006249999995, 0.44264849206349216, 0.4775844999999999, 0.5102675952380951, 0.54893288, 0.5832057878787877, 0.6154508, 0.6539295789473684, 0.6925377777777778, 0.734762625, 0.7674997500000001, 0.7988967777777778, 0.83530875, 0.87192775, 0.9324133333333333, 0.9774676666666666, 0.999644};
    const double avg_canput[hw2] = {
        0.00, 0.00, 0.00, 0.00, 4.00, 3.00, 4.00, 2.00,
        9.00, 5.00, 6.00, 6.00, 5.00, 8.38, 5.69, 9.13,
        5.45, 6.98, 6.66, 9.38, 6.98, 9.29, 7.29, 9.32, 
        7.37, 9.94, 7.14, 9.78, 7.31, 10.95, 7.18, 9.78, 
        7.76, 9.21, 7.33, 8.81, 7.20, 8.48, 7.23, 8.00, 
        6.92, 7.57, 6.62, 7.13, 6.38, 6.54, 5.96, 6.18, 
        5.62, 5.64, 5.18, 5.18, 4.60, 4.48, 4.06, 3.67, 
        3.39, 3.11, 2.66, 2.30, 1.98, 1.53, 1.78, 0.67
    };
    for (i = 0; i < hw2; ++i)
        eval_param.avg_canput[i] = avg_canput[i];
    for (i = 0; i < hw2; i++)
        eval_param.weight[i] = params[translate[i]];
    int all_idx = 0;
    for (i = 0; i < board_index_num; ++i)
        board_param.pattern_space[i] = consts[all_idx++];
    for (i = 0; i < board_index_num; ++i){
        for (j = 0; j < board_param.pattern_space[i]; ++j)
            board_param.board_translate[i][j] = consts[all_idx++];
    }
    int idx;
    for (i = 0; i < hw2; ++i){
        idx = 0;
        for (j = 0; j < board_index_num; ++j){
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i){
                    board_param.board_rev_translate[i][idx][0] = j;
                    board_param.board_rev_translate[i][idx++][1] = k;
                }
            }
        }
        for (j = idx; j < 4; ++j)
            board_param.board_rev_translate[i][j][0] = -1;
    }
    for (i = 0; i < hw2; ++i){
        for (j = 0; j < board_index_num; ++j){
            board_param.put[i][j] = -1;
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i)
                    board_param.put[i][j] = k;
            }
        }
    }
    /*
    all_idx = 0;
    for (i = 0; i < super_compress_pattern.length(); ++i){
        if ((int)super_compress_pattern[i] >= num_s){
            for (j = 0; j < (int)super_compress_pattern[i] - num_s + 1; ++j){
                compress_pattern[all_idx] = compress_pattern[all_idx - 1];
                ++all_idx;
            }
        } else {
            compress_pattern[all_idx] = super_compress_pattern[i];
            ++all_idx;
        }
    }
    //cerr << "unziped elems: " << all_idx << endl;
    for (i = 0; i < pattern_elem_num; ++i)
        patterns[i] = compress_vals[compress_pattern[i] - char_s];
    all_idx = 0;
    for (i = 0; i < pattern_num; ++i){
        for (j = 0; j < (int)pow(3, eval_param.pattern_space[i]); ++j)
            eval_param.pattern[i][j] = patterns[all_idx++];
    }
    */
    FILE *fp;
    char cbuf[1024];
    int mode;
    cin >> mode;
    if (mode == 0){
        if ((fp = fopen("param/param.txt", "r")) == NULL){
            printf("param file not exist");
            exit(1);
        }
    } else{
        if ((fp = fopen("param/param_new.txt", "r")) == NULL){
            printf("param file not exist");
            exit(1);
        }
    }
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_board_input; ++j){
            for (k = 0; k < kernel_size; ++k){
                for (l = 0; l < kernel_size; ++l){
                    if (!fgets(cbuf, 1024, fp)){
                        printf("param file broken");
                        exit(1);
                    }
                    eval_param.conv1[i][j][k][l] = atof(cbuf);
                }
            }
        }
    }
    int residual_i;
    for (residual_i = 0; residual_i < n_residual; ++residual_i){
        for (i = 0; i < n_kernels; ++i){
            for (j = 0; j < n_kernels; ++j){
                for (k = 0; k < kernel_size; ++k){
                    for (l = 0; l < kernel_size; ++l){
                        if (!fgets(cbuf, 1024, fp)){
                            printf("param file broken");
                            exit(1);
                        }
                        eval_param.conv_residual[residual_i][i][j][k][l] = atof(cbuf);
                    }
                }
            }
        }
    }
    for (i = 0; i < n_add_input; ++i){
        for (j = 0; j < n_dense1; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.dense1[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < n_dense1; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.bias1[i] = atof(cbuf);
    }
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_dense0; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.dense0[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < n_dense0; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.bias0[i] = atof(cbuf);
    }
    for (i = 0; i < n_dense1; ++i){
        for (j = 0; j < n_dense2; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.dense2[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < n_dense2; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.bias2[i] = atof(cbuf);
    }
    for (i = 0; i < n_joined; ++i){
        for (j = 0; j < hw2; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.dense3[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < hw2; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.bias3[i] = atof(cbuf);
    }
    for (i = 0; i < n_joined; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.dense4[i] = atof(cbuf);
    }
    if (!fgets(cbuf, 1024, fp)){
        printf("param file broken");
        exit(1);
    }
    eval_param.bias4 = atof(cbuf);
    if (mode == 0){
        if ((fp = fopen("param/mean.txt", "r")) == NULL){
            printf("mean file not exist");
            exit(1);
        }
    } else{
        if ((fp = fopen("param/mean_new.txt", "r")) == NULL){
            printf("mean file not exist");
            exit(1);
        }
    }
    for (i = 0; i < n_add_input; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("mean file broken");
            exit(1);
        }
        eval_param.mean[i] = atof(cbuf);
    }
    if (mode == 0){
        if ((fp = fopen("param/std.txt", "r")) == NULL){
            printf("std file not exist");
            exit(1);
        }
    } else{
        if ((fp = fopen("param/std_new.txt", "r")) == NULL){
            printf("std file not exist");
            exit(1);
        }
    }
    for (i = 0; i < n_add_input; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("std file broken");
            exit(1);
        }
        eval_param.std[i] = atof(cbuf);
    }
    /*
    if ((fp = fopen("param/book.txt", "r")) == NULL){
        printf("book file not exist");
        exit(1);
    }
    if (!fgets(cbuf, 1024, fp)){
        printf("book file broken");
        exit(1);
    }
    int book_len = atoi(cbuf);
    int policy;
    double rate;
    unsigned long long up, uo;
    pair<unsigned long long, unsigned long long> key;
    for (i = 0; i < book_len; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("book file broken");
            exit(1);
        }
        up = atoll(cbuf);
        if (!fgets(cbuf, 1024, fp)){
            printf("book file broken");
            exit(1);
        }
        uo = atoll(cbuf);
        if (!fgets(cbuf, 1024, fp)){
            printf("book file broken");
            exit(1);
        }
        policy = atoi(cbuf);
        if (!fgets(cbuf, 1024, fp)){
            printf("book file broken");
            exit(1);
        }
        rate = atof(cbuf);
        //cerr << up << " " << uo << " " << policy << " " << rate << endl;
        key.first = up;
        key.second = uo;
        search_param.book[key].policy = policy;
        search_param.book[key].rate = rate;
    }
    */
    int p, o, mobility, canput_num, rev;
    for (i = 0; i < 6561; ++i){
        board_param.reverse[i] = board_reverse(i);
        p = reverse_line(create_p(i));
        o = reverse_line(create_o(i));
        eval_param.cnt_p[i] = 0;
        eval_param.cnt_o[i] = 0;
        for (j = 0; j < hw; ++j){
            board_param.restore_p[i][j] = 1 & (p >> (hw_m1 - j));
            board_param.restore_o[i][j] = 1 & (o >> (hw_m1 - j));
            board_param.restore_vacant[i][j] = 1 & ((~(p | o)) >> (hw_m1 - j));
            eval_param.cnt_p[i] += board_param.restore_p[i][j];
            eval_param.cnt_o[i] += board_param.restore_o[i][j];
        }
        mobility = check_mobility(p, o);
        canput_num = 0;
        for (j = 0; j < hw; ++j){
            if (1 & (mobility >> (hw_m1 - j))){
                rev = move_line(p, o, hw_m1 - j);
                ++canput_num;
                board_param.legal[i][j] = true;
                for (k = 0; k < board_index_num; ++k){
                    board_param.trans[k][i][j] = 0;
                    for (l = 0; l < board_param.pattern_space[k]; ++l)
                        board_param.trans[k][i][j] |= (unsigned long long)(1 & (rev >> (7 - l))) << board_param.board_translate[k][l];
                    board_param.neighbor8[k][i][j] = 0;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) << 1;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) >> 1;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) << hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) >> hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_p1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_p1;
                    board_param.neighbor8[k][i][j] &= ~board_param.trans[k][i][j];
                }
            } else
                board_param.legal[i][j] = false;
        }
        eval_param.canput[i] = canput_num;
    }
    for (i = 0; i < hw2; ++i){
        board_param.put_idx_num[i] = 0;
        for (j = 0; j < board_index_num; ++j){
            if (board_param.put[i][j] != -1)
                board_param.put_idx[i][board_param.put_idx_num[i]++] = j;
        }
    }
    for (i = 0; i < 15; ++i)
        board_param.pow3[i] = (int)pow(3, i);
    for (i = 0; i < 6561; ++i){
        for (j = 0; j < 8; ++j){
            board_param.rev_bit3[i][j] = board_param.pow3[j] * (2 - (i / board_param.pow3[j]) % 3);
            board_param.pop_digit[i][j] = i / board_param.pow3[j] % 3;
        }
    }
    for (i = 0; i < hw; ++i){
        for (j = 0; j < 6561; ++j){
            eval_param.weight_p[i][j] = 0.0;
            eval_param.weight_o[i][j] = 0.0;
            for (k = 0; k < 8; ++k){
                if (board_param.pop_digit[j][k] == 1)
                    eval_param.weight_p[i][j] += eval_param.weight[i * hw + k];
                else if (board_param.pop_digit[j][k] == 2)
                    eval_param.weight_o[i][j] += eval_param.weight[i * hw + k];
            }
        }
    }
    bool flag;
    for (i = 0; i < 6561; ++i){
        eval_param.confirm_p[i] = 0;
        eval_param.confirm_o[i] = 0;
        flag = true;
        for (j = 0; j < hw; ++j)
            if (!board_param.pop_digit[i][j])
                flag = false;
        if (flag){
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] == 1)
                    ++eval_param.confirm_p[i];
                else
                    ++eval_param.confirm_o[i];
            }
        } else {
            flag = true;
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] != 1)
                    break;
                ++eval_param.confirm_p[i];
                if (k == hw_m1)
                    flag = false;
            }
            if (flag){
                for (j = hw_m1; j >= 0; --j){
                    if (board_param.pop_digit[i][j] != 1)
                        break;
                    ++eval_param.confirm_p[i];
                    if (k == hw_m1)
                        flag = false;
                }
            }
            flag = true;
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] != 2)
                    break;
                ++eval_param.confirm_o[i];
                if (k == hw_m1)
                    flag = false;
            }
            if (flag){
                for (j = hw_m1; j >= 0; --j){
                    if (board_param.pop_digit[i][j] != 2)
                        break;
                    ++eval_param.confirm_o[i];
                    if (k == hw_m1)
                        flag = false;
                }
            }
        }
    }
    for (i = 0; i < 6561; ++i){
        eval_param.pot_canput_p[i] = 0;
        eval_param.pot_canput_o[i] = 0;
        for (j = 0; j < hw_m1; ++j){
            if (board_param.pop_digit[i][j] == 0){
                if (board_param.pop_digit[i][j + 1] == 2)
                    ++eval_param.pot_canput_p[i];
                else if (board_param.pop_digit[i][j + 1] == 1)
                    ++eval_param.pot_canput_o[i];
            }
        }
        for (j = 1; j < hw; ++j){
            if (board_param.pop_digit[i][j] == 0){
                if (board_param.pop_digit[i][j - 1] == 2)
                    ++eval_param.pot_canput_p[i];
                else if (board_param.pop_digit[i][j - 1] == 1)
                    ++eval_param.pot_canput_o[i];
            }
        }
    }
    for (i = 0; i < 3; ++i){
        for (j = 0; j < 10; ++j)
            board_param.digit_pow[i][j] = i * board_param.pow3[j];
    }
}

inline double leaky_relu(double x){
    return max(x, 0.01 * x);
}

inline predictions predict(const int *board){
    int i, j, sy, sx, y, x, residual_i;
    predictions res;
    /*
    for (i = 0; i < hw2; ++i)
        res.policies[i] = myrandom();
    res.value = myrandom();
    return res;
    */
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            eval_param.input_b[0][i][j] = board_param.restore_p[board[i]][j];
            eval_param.input_b[1][i][j] = board_param.restore_o[board[i]][j];
            eval_param.input_b[2][i][j] = board_param.restore_vacant[board[i]][j];
        }
    }
    for (i = 0; i < n_add_input; ++i)
        eval_param.input_p[i] = 0.0;
    for (i = 0; i < hw; ++i){
        eval_param.input_p[0] += eval_param.cnt_p[board[i]];
        eval_param.input_p[1] += eval_param.cnt_o[board[i]];
    }
    for (i = 0; i < board_index_num; ++i)
        eval_param.input_p[2] += eval_param.canput[board[i]];
    eval_param.input_p[3] = eval_param.avg_canput[search_param.turn];
    for (i = 0; i < hw; ++i){
        eval_param.input_p[4] += eval_param.weight_p[i][board[i]];
        eval_param.input_p[5] += eval_param.weight_o[i][board[i]];
    }
    eval_param.input_p[6] = eval_param.confirm_p[board[0]] + eval_param.confirm_p[board[7]] + eval_param.confirm_p[board[8]] + eval_param.confirm_p[board[15]];
    eval_param.input_p[7] = eval_param.confirm_o[board[0]] + eval_param.confirm_o[board[7]] + eval_param.confirm_o[board[8]] + eval_param.confirm_o[board[15]];
    for (i = 0; i < board_index_num; ++i){
        eval_param.input_p[8] += eval_param.pot_canput_p[board[i]];
        eval_param.input_p[9] += eval_param.pot_canput_o[board[i]];
    }
    eval_param.input_p[10] = search_param.turn;
    for (i = 0; i < n_add_input; ++i){
        eval_param.input_p[i] -= eval_param.mean[i];
        eval_param.input_p[i] /= eval_param.std[i];
    }
    // conv and normalization and leaky-relu for input_b
    for (i = 0; i < n_kernels; ++i){
        for (y = 0; y < hw; ++y){
            for (x = 0; x < hw; ++x)
                eval_param.hidden_conv1[i][y][x] = 0.0;
        }
        for (j = 0; j < n_board_input; ++j){
            for (sy = 0; sy < hw; ++sy){
                for (sx = 0; sx < hw; ++sx){
                    for (y = 0; y < kernel_size; ++y){
                        for (x = 0; x < kernel_size; ++x){
                            if (sy + y + conv_start < 0 || sy + y + conv_start >= hw || sx + x + conv_start < 0 || sx + x + conv_start >= hw)
                                continue;
                            eval_param.hidden_conv1[i][sy][sx] += eval_param.conv1[i][j][y][x] * eval_param.input_b[j][sy + y + conv_start][sx + x + conv_start];
                        }
                    }
                }
            }
        }
        for (y = 0; y < hw; ++y){
            for (x = 0; x < hw; ++x)
                eval_param.hidden_conv1[i][y][x] = leaky_relu(eval_param.hidden_conv1[i][y][x]);
        }
    }
    // residual-error-block for input_b
    for (residual_i = 0; residual_i < n_residual; ++residual_i){
        for (i = 0; i < n_kernels; ++i){
            for (y = 0; y < hw; ++y){
                for (x = 0; x < hw; ++x)
                    eval_param.hidden_conv2[i][y][x] = 0.0;
            }
            for (j = 0; j < n_kernels; ++j){
                for (sy = 0; sy < hw; ++sy){
                    for (sx = 0; sx < hw; ++sx){
                        for (y = 0; y < kernel_size; ++y){
                            for (x = 0; x < kernel_size; ++x){
                                if (sy + y + conv_start < 0 || sy + y + conv_start >= hw || sx + x + conv_start < 0 || sx + x + conv_start >= hw)
                                    continue;
                                eval_param.hidden_conv2[i][sy][sx] += eval_param.conv_residual[residual_i][i][j][y][x] * eval_param.hidden_conv1[j][sy + y + conv_start][sx + x + conv_start];
                            }
                        }
                    }
                }
            }
        }
        for (i = 0; i < n_kernels; ++i){
            for (y = 0; y < hw; ++y){
                for (x = 0; x < hw; ++x)
                    eval_param.hidden_conv1[i][y][x] = leaky_relu(eval_param.hidden_conv1[i][y][x] + eval_param.hidden_conv2[i][y][x]);
            }
        }
    }
    // global-average-pooling for input_b
    for (i = 0; i < n_kernels; ++i){
        eval_param.hidden_gap0[i] = 0.0;
        for (y = 0; y < hw; ++y){
            for (x = 0; x < hw; ++x)
                eval_param.hidden_gap0[i] += eval_param.hidden_conv1[i][y][x];
        }
        eval_param.hidden_gap0[i] /= div_pooling;
    }
    // dense0 and bias and leaky-relu for input_b
    for (i = 0; i < n_dense0; ++i)
        eval_param.hidden_dense1[i] = 0.0;
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_dense0; ++j)
            eval_param.hidden_joined[j] += eval_param.dense0[i][j] * eval_param.hidden_gap0[i];
    }
    for (i = 0; i < n_dense0; ++i)
        eval_param.hidden_joined[i] = leaky_relu(eval_param.hidden_joined[i] + eval_param.bias0[i]);
    // dense1 and bias and leaky-relu for input_p
    for (i = 0; i < n_dense1; ++i)
        eval_param.hidden_dense1[i] = 0.0;
    for (i = 0; i < n_add_input; ++i){
        for (j = 0; j < n_dense1; ++j)
            eval_param.hidden_dense1[j] += eval_param.dense1[i][j] * eval_param.input_p[i];
    }
    for (i = 0; i < n_dense1; ++i)
        eval_param.hidden_dense1[i] = leaky_relu(eval_param.hidden_dense1[i] + eval_param.bias1[i]);
    // dense2 and bias and leaky-relu and join for input_p
    for (i = 0; i < n_dense2; ++i)
        eval_param.hidden_joined[n_kernels + i] = 0.0;
    for (i = 0; i < n_dense1; ++i){
        for (j = 0; j < n_dense2; ++j)
            eval_param.hidden_joined[n_kernels + j] += eval_param.hidden_dense1[i] * eval_param.dense2[i][j];
    }
    for (i = 0; i < n_dense2; ++i)
        eval_param.hidden_joined[n_kernels + i] = leaky_relu(eval_param.hidden_joined[n_kernels + i] + eval_param.bias2[i]);
    // dense and bias and softmax for policy output *don't need softmax because use softmax later
    for (i = 0; i < hw2; ++i)
        res.policies[i] = 0.0;
    for (i = 0; i < n_joined; ++i){
        for (j = 0; j < hw2; ++j)
            res.policies[j] += eval_param.hidden_joined[i] * eval_param.dense3[i][j];
    }
    for (i = 0; i < hw2; ++i)
        res.policies[i] += eval_param.bias3[i];
    /*
    double policy_sum = 0.0;
    for (i = 0; i < hw2; ++i){
        res.policies[i] = exp(max(-32.0, min(10.0, res.policies[i] + eval_param.bias3[i])));
        policy_sum += res.policies[i];
    }
    for (i = 0; i < hw2; ++i)
        res.policies[i] /= policy_sum;
    */
    // dense and bias and tanh for value output
    res.value = 0.0;
    for (i = 0; i < n_joined; ++i)
        res.value += eval_param.hidden_joined[i] * eval_param.dense4[i];
    res.value = tanh(res.value + eval_param.bias4);
    // return
    return res;
}

inline void move(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1)
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
    }
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            for (j = 0; j < 4; ++j){
                if (board_param.board_rev_translate[i][j][0] == -1)
                    break;
                res[board_param.board_rev_translate[i][j][0]] += board_param.rev_bit3[res[board_param.board_rev_translate[i][j][0]]][board_param.board_rev_translate[i][j][1]];
            }
        }
    }
}

inline int end_game(const int *board){
    int res = 0, i, j, p, o;
    for (i = 0; i < hw; ++i){
        res += eval_param.cnt_p[board[i]];
        res -= eval_param.cnt_o[board[i]];
    }
    return res;
}

inline double end_game_evaluate(int idx, int player){
    double value = c_end * min(1.0, max(-1.0, (double)end_game(mcts_param.nodes[idx].board)));
    if (value * player > 0.0)
        ++search_param.win_num;
    else if (value * player < 0.0)
        ++search_param.lose_num;
    ++search_param.n_playout;
    mcts_param.nodes[idx].w += value;
    ++mcts_param.nodes[idx].n;
    return value;
}

double evaluate(int idx, bool passed, int player){
    double value = 0.0;
    int i, j, cell;
    if (!mcts_param.nodes[idx].expanded){
        // when children not expanded
        // expand children
        mcts_param.nodes[idx].expanded = true;
        bool legal[hw2];
        mcts_param.nodes[idx].pass = true;
        for (cell = 0; cell < hw2; ++cell){
            mcts_param.nodes[idx].children[cell] = -1;
            legal[cell] = false;
            for (i = 0; i < board_index_num; ++i){
                if (board_param.put[cell][i] != -1){
                    if (board_param.legal[mcts_param.nodes[idx].board[i]][board_param.put[cell][i]]){
                        mcts_param.nodes[idx].pass = false;
                        legal[cell] = true;
                        break;
                    }
                }
            }
        }
        mcts_param.nodes[idx].children[hw2] = -1;
        if (!mcts_param.nodes[idx].pass){
            //predict and create policy array
            predictions pred = predict(mcts_param.nodes[idx].board);
            mcts_param.nodes[idx].w += pred.value;
            value = pred.value;
            ++mcts_param.nodes[idx].n;
            double p_sum = 0.0;
            for (i = 0; i < hw2; ++i){
                if (legal[i]){
                    mcts_param.nodes[idx].p[i] = exp(max(-32.0, min(10.0, pred.policies[i])));
                    p_sum += mcts_param.nodes[idx].p[i];
                } else{
                    mcts_param.nodes[idx].p[i] = 0.0;
                }
            }
            for (i = 0; i < hw2; ++i)
                mcts_param.nodes[idx].p[i] /= p_sum;
        }
    }
    if (!mcts_param.nodes[idx].pass){
        // children already expanded
        // select next move
        int a_cell = -1;
        value = -inf;
        double tmp_value;
        double t_sqrt = sqrt((double)mcts_param.nodes[idx].n);
        for (cell = 0; cell < hw2; ++cell){
            if (mcts_param.nodes[idx].p[cell] != 0.0){
                if (mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n > 0)
                    tmp_value = mcts_param.nodes[mcts_param.nodes[idx].children[cell]].w / mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n;
                else
                    tmp_value = 0.0;
                tmp_value += c_puct * mcts_param.nodes[idx].p[cell] * t_sqrt / (1.0 + (double)mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n);
                if (value < tmp_value){
                    value = tmp_value;
                    a_cell = cell;
                }
            }
        }
        if (mcts_param.nodes[idx].children[a_cell] == -1){
            mcts_param.nodes[idx].children[a_cell] = mcts_param.used_idx;
            mcts_param.nodes[mcts_param.used_idx].w = 0.0;
            mcts_param.nodes[mcts_param.used_idx].n = 0;
            mcts_param.nodes[mcts_param.used_idx].pass = true;
            mcts_param.nodes[mcts_param.used_idx].expanded = false;
            move(mcts_param.nodes[idx].board, mcts_param.nodes[mcts_param.used_idx++].board, a_cell);
        }
        value = -evaluate(mcts_param.nodes[idx].children[a_cell], false, -player);
        mcts_param.nodes[idx].w += value;
        ++mcts_param.nodes[idx].n;
    } else{
        // pass
        if (passed){
            return end_game_evaluate(idx, player);
        } else{
            if (mcts_param.nodes[idx].children[hw2] == -1){
                mcts_param.nodes[idx].children[hw2] = mcts_param.used_idx;
                mcts_param.nodes[mcts_param.used_idx].w = 0.0;
                mcts_param.nodes[mcts_param.used_idx].n = 0;
                mcts_param.nodes[mcts_param.used_idx].pass = true;
                mcts_param.nodes[mcts_param.used_idx].expanded = false;
                for (i = 0; i < board_index_num; ++i)
                    mcts_param.nodes[mcts_param.used_idx].board[i] = board_param.reverse[mcts_param.nodes[idx].board[i]];
                ++mcts_param.used_idx;
            }
            value = -evaluate(mcts_param.nodes[idx].children[hw2], true, -player);
            mcts_param.nodes[idx].w += value;
            ++mcts_param.nodes[idx].n;
        }
    }
    return value;
}

inline int next_action(int *board){
    int i, cell, mx = 0, res = -1;
    mcts_param.used_idx = 1;
    for (i = 0; i < board_index_num; ++i)
        mcts_param.nodes[0].board[i] = board[i];
    mcts_param.nodes[0].w = 0.0;
    mcts_param.nodes[0].n = 0;
    mcts_param.nodes[0].pass = true;
    mcts_param.nodes[0].expanded = true;
    // expand children
    bool legal[hw2];
    for (cell = 0; cell < hw2; ++cell){
        mcts_param.nodes[0].children[cell] = -1;
        legal[cell] = false;
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[cell][i] != -1){
                if (board_param.legal[board[i]][board_param.put[cell][i]]){
                    mcts_param.nodes[0].pass = false;
                    legal[cell] = true;
                    break;
                }
            }
        }
    }
    //predict and create policy array
    predictions pred = predict(board);
    mcts_param.nodes[0].w += pred.value;
    ++mcts_param.nodes[0].n;
    double p_sum = 0.0;
    for (i = 0; i < hw2; ++i){
        if (legal[i]){
            mcts_param.nodes[0].p[i] = exp(max(-32.0, min(10.0, pred.policies[i])));
            p_sum += mcts_param.nodes[0].p[i];
        } else{
            mcts_param.nodes[0].p[i] = 0.0;
        }
    }
    for (i = 0; i < hw2; ++i)
        mcts_param.nodes[0].p[i] /= p_sum;
    int strt = tim();
    for (i = 0; i < evaluate_count; ++i){
        evaluate(0, false, 1);
        if (tim() - strt > search_param.tl)
            break;
    }
    for (i = 0; i < hw2; ++i){
        if (mcts_param.nodes[0].children[i] != -1){
            //cerr << i << " " << mcts_param.nodes[mcts_param.nodes[0].children[i]].n << endl;
            if (mx < mcts_param.nodes[mcts_param.nodes[0].children[i]].n){
                mx = mcts_param.nodes[mcts_param.nodes[0].children[i]].n;
                res = i;
            }
        }
    }
    return res;
}

int main(){
    init();
    cerr << "initialized" << endl;
    int i, j, board_tmp, ai_player, policy;
    char elem;
    unsigned long long p, o;
    int board[board_index_num];
    double rnd, sm;
    pair<unsigned long long, unsigned long long> key;
    while (true){
        search_param.turn = 0;
        search_param.win_num = 0;
        search_param.lose_num = 0;
        search_param.n_playout = 0;
        p = 0;
        o = 0;
        //cin >> ai_player;
        //cin >> search_param.tl;
        for (i = 0; i < hw2; ++i){
            cin >> elem;
            if (elem != '.'){
                ++search_param.turn;
                p |= (unsigned long long)(elem == '0') << i;
                o |= (unsigned long long)(elem == '1') << i;
            }
        }
        if (ai_player == 1)
            swap(p, o);
        key.first = p;
        key.second = o;
        //cerr << key.first << " " << key.second << endl;
        /*
        if (search_param.book.find(key) != search_param.book.end()){
            cerr << "BOOK " << search_param.book[key].policy << " " << 100.0 * search_param.book[key].rate << endl;
            cout << search_param.book[key].policy / hw << " " << search_param.book[key].policy % hw << " " << 100.0 * search_param.book[key].rate << endl;
            continue;
        }
        */
        for (i = 0; i < board_index_num; ++i){
            board_tmp = 0;
            for (j = 0; j < board_param.pattern_space[i]; ++j){
                if (1 & (p >> board_param.board_translate[i][j]))
                    board_tmp += board_param.pow3[j];
                else if (1 & (o >> board_param.board_translate[i][j]))
                    board_tmp += 2 * board_param.pow3[j];
            }
            board[i] = board_tmp;
        }
        /*
        print_board(board);
        predictions tmp = predict(board);
        double mx = -1000.0;
        int mx_idx = -1;
        for (i = 0; i < hw2; ++i){
            if (mx < tmp.policies[i]){
                mx = tmp.policies[i];
                mx_idx = i;
            }
        }
        cerr << mx_idx << " " << mx << " " << tmp.value << endl;
        return 0;
        */
        policy = next_action(board);
        //cerr << "SEARCH " << search_param.win_num << " " << search_param.lose_num << "  " << search_param.n_playout << " " << mcts_param.used_idx << endl;
        //cout << policy / hw << " " << policy % hw << " " << 100.0 * (double)(search_param.win_num - search_param.lose_num) / search_param.n_playout << endl;
        //cout << policy / hw << " " << policy % hw << " " << 100.0 * (double)search_param.win_num / search_param.n_playout << endl;
        cout << policy / hw << " " << policy % hw << endl;
    }
    return 0;
}
