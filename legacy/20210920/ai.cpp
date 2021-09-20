#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx")

// Reversi AI C++ version 5
// previous 11th rate 30.44
// use Negascout

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
#define window 0.00001
#define simple_threshold 3
#define inf 100000.0
#define param_num 36
#define board_index_num 38
#define pattern_num 5
#define char_s 35
#define char_e 91
#define num_s 93
#define num_e 126
#define pattern_elem_num 85293
#define hash_table_size 16384
#define hash_mask (hash_table_size - 1)
#define stage1 137
#define stage2 128
#define stage3 64

struct node_t{
    int k[hw];
    double v;
    node_t* p_n_node;
};

inline int calc_hash(const int *p){
    int seed = 0;
    for (int i = 0; i < hw; ++i)
        seed ^= p[i] << (i / 4);
    return seed & hash_mask;
}

inline void hash_table_init(node_t** hash_table){
    for(int i = 0; i < hash_table_size; ++i)
        hash_table[i] = NULL;
}

inline node_t* node_init(const int *key, double val){
    node_t* p_node = NULL;
    p_node = (node_t*)malloc(sizeof(node_t));
    for (int i = 0; i < hw; ++i)
        p_node->k[i] = key[i];
    p_node->v = val;
    p_node->p_n_node = NULL;
    return p_node;
}

inline bool compare_key(const int *a, const int *b){
    for (int i = 0; i < hw; ++i){
        if (a[i] != b[i])
            return false;
    }
    return true;
}

inline void register_hash(node_t** hash_table, const int *key, int hash, double val){
    if(hash_table[hash] == NULL){
        hash_table[hash] = node_init(key, val);
    } else {
        node_t *p_node = p_node = hash_table[hash];
        node_t *p_pre_node = NULL;
        p_pre_node = p_node;
        while(p_node != NULL){
            if(compare_key(key, p_node->k)){
                p_node->v = val;
                return;
            }
            p_pre_node = p_node;
            p_node = p_node->p_n_node;
        }
        p_pre_node->p_n_node = node_init(key, val);
    }
}

inline double get_val_hash(node_t** hash_table, const int *key, int hash){
    node_t *p_node = hash_table[hash];
    while(p_node != NULL){
        if(compare_key(key, p_node->k))
            return p_node->v;
        p_node = p_node->p_n_node;
    }
    return -inf;
}

inline void hash_table_copy(node_t** to_table, node_t** fr_table){
    for(int i = 0; i < hash_table_size; ++i){
        
    }
}

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
    int restore_p[6561][hw], restore_o[6561][hw];
};

struct eval_param{
    double weight[hw2];
    double pattern_weight, cnt_weight, canput_weight, weight_weight, confirm_weight, pot_canput_weight, open_weight;
    double cnt_bias;
    double weight_sme[param_num];
    double avg_canput[hw2];
    int canput[6561];
    int cnt_p[6561], cnt_o[6561];
    double weight_p[hw][6561], weight_o[hw][6561];
    int pattern_variation[pattern_num], pattern_space[pattern_num];
    int pattern_translate[pattern_num][8][10][2];
    double pattern_each_weight[pattern_num];
    double pattern[pattern_num][59049];
    int confirm_p[6561], confirm_o[6561];
    int pot_canput_p[6561], pot_canput_o[6561];
    double open_eval[40];
    double nodes1[stage1];
    double nodes2[stage2];
    double nodes3[stage3];
    double b1[stage2];
    double b2[stage3];
    double w1[stage1][stage2];
    double w2[stage2][stage3];
    double w3[stage3];
    double mean[stage1];
    double std[stage1];
};

struct search_param{
    node_t *memo_lb[hash_table_size];
    node_t *memo_ub[hash_table_size];
    node_t *previous_memo[hash_table_size];
    int max_depth;
    int min_max_depth;
    int strt, tl;
    int turn;
    int searched_nodes;
    vector<int> vacant_lst;
    int vacant_cnt;
    int weak_mode;
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

board_param board_param;
eval_param eval_param;
search_param search_param;

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
    int compress_pattern[pattern_elem_num];
    double patterns[pattern_elem_num];
    for (i = 0; i < hw2; i++)
        eval_param.weight[i] = params[translate[i]];
    int all_idx = 0;
    for (i = 0; i < board_index_num; ++i)
        board_param.pattern_space[i] = consts[all_idx++];
    for (i = 0; i < board_index_num; ++i){
        for (j = 0; j < board_param.pattern_space[i]; ++j)
            board_param.board_translate[i][j] = consts[all_idx++];
    }
    for (i = 0; i < pattern_num; ++i)
        eval_param.pattern_space[i] = consts[all_idx++];
    for (i = 0; i < pattern_num; ++i)
        eval_param.pattern_variation[i] = consts[all_idx++];
    for (i = 0; i < pattern_num; ++i){
        for (j = 0; j < eval_param.pattern_variation[i]; ++j){
            for (k = 0; k < eval_param.pattern_space[i]; ++k){
                eval_param.pattern_translate[i][j][k][0] = consts[all_idx] / hw;
                eval_param.pattern_translate[i][j][k][1] = consts[all_idx++] % hw;
            }
        }
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
    if ((fp = fopen("param/param.txt", "r")) == NULL){
        printf("param file not exist");
        exit(1);
    }
    for (i = 0; i < stage2; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.b1[i] = atof(cbuf);
    }
    for (i = 0; i < stage3; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.b2[i] = atof(cbuf);
    }
    for (i = 0; i < stage1; ++i){
        for (j = 0; j < stage2; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.w1[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < stage2; ++i){
        for (j = 0; j < stage3; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param file broken");
                exit(1);
            }
            eval_param.w2[i][j] = atof(cbuf);
        }
    }
    for (i = 0; i < stage3; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.w3[i] = atof(cbuf);
    }
    if ((fp = fopen("param/mean.txt", "r")) == NULL){
        printf("mean file not exist");
        exit(1);
    }
    for (i = 0; i < stage1; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("mean file broken");
            exit(1);
        }
        eval_param.mean[i] = atof(cbuf);
    }
    if ((fp = fopen("param/std.txt", "r")) == NULL){
        printf("std file not exist");
        exit(1);
    }
    for (i = 0; i < stage1; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("std file broken");
            exit(1);
        }
        eval_param.std[i] = atof(cbuf);
    }
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
    for (i = 0; i < 40; ++i)
        eval_param.open_eval[i] = min(1.0, pow(2.0, 2.0 - 0.667 * i) - 1.0);
}

inline double leaky_relu(double x){
    if (x >= 0.0)
        return x;
    return 0.01 * x;
}

inline double evaluate(const int *board){
    int i, j;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            eval_param.nodes1[i * hw + j] = board_param.restore_p[board[i]][j];
            eval_param.nodes1[hw2 + i * hw + j] = board_param.restore_o[board[i]][j];
        }
    }
    for (i = hw22; i < stage1; ++i)
        eval_param.nodes1[i] = 0.0;
    for (i = 0; i < hw; ++i){
        eval_param.nodes1[128] += eval_param.cnt_p[board[i]];
        eval_param.nodes1[129] += eval_param.cnt_o[board[i]];
    }
    for (i = 0; i < board_index_num; ++i)
        eval_param.nodes1[130] += eval_param.canput[board[i]];
    for (i = 0; i < hw; ++i){
        eval_param.nodes1[131] += eval_param.weight_p[i][board[i]];
        eval_param.nodes1[132] += eval_param.weight_o[i][board[i]];
    }
    eval_param.nodes1[133] = eval_param.confirm_p[board[0]] + eval_param.confirm_p[board[7]] + eval_param.confirm_p[board[8]] + eval_param.confirm_p[board[15]];
    eval_param.nodes1[134] = eval_param.confirm_o[board[0]] + eval_param.confirm_o[board[7]] + eval_param.confirm_o[board[8]] + eval_param.confirm_o[board[15]];
    for (i = 0; i < board_index_num; ++i){
        eval_param.nodes1[135] += eval_param.pot_canput_p[board[i]];
        eval_param.nodes1[136] += eval_param.pot_canput_o[board[i]];
    }
    for (i = 0; i < stage1; ++i)
        eval_param.nodes1[i] = (eval_param.nodes1[i] - eval_param.mean[i]) / eval_param.std[i];
    /*
    double tmp[stage1] = {-0.41372130668216023, -0.5065175019356489, -0.7081910303318515, -0.7666956026373731, -0.7730619956078894, -0.7191197952725302, -0.5259955425505751, -0.4141067613470382, -0.5247618983043321, 2.091507847623098, 1.1967176545798253, -0.8693230431438846, 1.1390069055614969, 1.2268746494759817, -0.4789003976854073, -0.5061131521431709, 1.3919602632960089, 1.2267145073103332, 1.0488607901029134, 1.0520256067605769, -0.9702351549841688, -0.9538000938858063, -0.8366073187899705, -0.707863561674811, 1.294521522261411, 1.1394733208862007, 1.0308902358822054, 1.005372549167396, 1.0046892839106225, -0.9515023951988345, -0.8705607103152984, -0.7664235387988397, -0.7663059039321946, -0.8689028430745848, -0.9509053001351485, -0.99567099466472, 1.004653335570387, -0.9710694229817692, -0.8778859198414188, -0.7726264315718214, 1.413682416581901, 1.1963750557873651, -0.9528513662604832, -0.970078804762781, -0.9498484157653805, 1.0493498988977628, -0.8149044034394626, -0.7184971759123431, -0.5062525898143757, -0.4780815923060661, -0.8137574220288323, -0.8764970416035841, 1.151843116302268, 1.19629771315921, 2.0891328619975345, -0.5249629119830072, -0.4143216618460377, -0.5250807427231838, -0.7184685544349083, -0.7730472287310365, -0.7662250349913002, -0.7073084047251496, -0.5060643472867916, -0.41334311964973375, -0.38753203404453157, -0.5025480719334369, 1.410091689223894, 1.372878191752746, 1.4081960173833894, 1.4394428863800712, 1.9987485473006745, -0.38712733280734707, -0.5008161080455217, -0.4953329103695055, -0.8636034302113038, 1.0934711120838105, -0.9088846639246873, -0.8594826962988134, -0.4945065425089955, -0.5027854279216738, -0.6948971996488319, -0.8603718533449376, -0.9391429667788405, -0.981089911431135, 1.0206619203548744, 1.0656137550390894, 1.1600896611865512, -0.7094729197339064, -0.7104704880790357, -0.9089338919990434, -0.980045726991202, -1.005372549167396, -1.0046892839106223, 1.0205705871459614, 1.095850305197933, -0.7289074529238385, 1.3730815521198496, 1.093826122733287, 1.0197307204290642, 0.99567099466472, -1.004653335570387, 1.0217128643232811, 1.1004482736449297, 1.4058535036799968, -0.7095584069082691, -0.862948542476314, 1.0643703978775627, 1.020552321525698, 1.0186089767121655, -0.9390248987691339, 1.1623094063625847, -0.6953926076568459, -0.5028342926455851, -0.4951858663720385, 1.160344280726683, 1.0987213567712066, -0.9150220185235145, -0.863256217002165, -0.4951928686763166, -0.5007183008207675, -0.38689052690401315, -0.5007043280520537, 1.436997979632391, 1.4063609700293183, 1.3720652359342316, 1.4088321515300273, -0.5033438275054203, -0.3883179524746891, -0.3916239733878557, 0.647901969686857, 0.9820030031696403, -0.7416724970971031, -0.11351914155623194, -0.659935723328636, -0.6283933144644066, 0.7948045470447653, -0.6531841324367358};
    for (i = 0; i < stage1; ++i)
        eval_param.nodes1[i] = tmp[i];
    */
    for (i = 0; i < stage2; ++i)
        eval_param.nodes2[i] = 0.0;
    for (i = 0; i < stage3; ++i)
        eval_param.nodes3[i] = 0.0;
    // 1st layer
    for (i = 0; i < stage1; ++i){
        for (j = 0; j < stage2; ++j)
            eval_param.nodes2[j] += eval_param.nodes1[i] * eval_param.w1[i][j];
    }
    for (i = 0; i < stage2; ++i){
        eval_param.nodes2[i] += eval_param.b1[i];
        eval_param.nodes2[i] = leaky_relu(eval_param.nodes2[i]);
        if (!(myrandom_int() & 0b1111)){
            eval_param.nodes2[i] = 0.0;
        }
    }
    // 2nd layer
    for (i = 0; i < stage2; ++i){
        for (j = 0; j < stage3; ++j)
            eval_param.nodes3[j] += eval_param.nodes2[i] * eval_param.w2[i][j];
    }
    for (i = 0; i < stage3; ++i){
        eval_param.nodes3[i] += eval_param.b2[i];
        eval_param.nodes3[i] = leaky_relu(eval_param.nodes3[i]);
    }
    // 3rd layer
    double res = 0.0;
    for (i = 0; i < stage3; ++i)
        res += eval_param.nodes3[i] * eval_param.w3[i];
    return res;
}

inline double end_game(const int *board){
    int res = 0, i, j, p, o;
    for (i = 0; i < hw; ++i){
        res += eval_param.cnt_p[board[i]];
        res -= eval_param.cnt_o[board[i]];
    }
    res *= 1000;
    if (search_param.weak_mode)
        return -(double)res;
    return (double)res;
}

inline int move_open(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0, neighbor = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1){
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
            neighbor |= board_param.neighbor8[i][board[i]][board_param.put[coord][i]];
        }
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
    int open_val = 0;
    for (i = 0; i < hw2; ++i){
        if(1 & (neighbor >> i))
            open_val += (int)(board_param.pop_digit[board[i >> 3]][i & 0b111] == 0);
    }
    return open_val;
}

inline open_vals open_val_forward(int *board, int depth, bool player){
    open_vals res;
    if (depth == 0){
        res.p_open_val = 0.0;
        res.o_open_val = 0.0;
        res.p_cnt = 0;
        res.o_cnt = 0;
        return res;
    }
    --depth;
    int i, j;
    int n_board[board_index_num];
    open_vals tmp;
    res.p_open_val = -inf;
    res.o_open_val = inf;
    double open_val = -inf;
    bool passed = false;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                passed = false;
                open_val = max(open_val, eval_param.open_eval[move_open(board, n_board, cell)]);
                tmp = open_val_forward(n_board, depth, !player);
                if (res.p_open_val < tmp.p_open_val){
                    res.p_open_val = tmp.p_open_val;
                    res.p_cnt = tmp.p_cnt;
                }
                if (res.o_open_val > tmp.o_open_val){
                    res.o_open_val = tmp.o_open_val;
                    res.o_cnt = tmp.o_cnt;
                }
            }
        }
    }
    if (passed){
        res.p_open_val = 0.0;
        res.o_open_val = 0.0;
        res.p_cnt = 0;
        res.o_cnt = 0;
        return res;
    }
    if (player){
        res.p_open_val += open_val;
        ++res.p_cnt;
    } else {
        res.o_open_val += open_val;
        ++res.o_cnt;
    }
    return res;
}

int cmp(board_priority p, board_priority q){
    return p.priority > q.priority;
}

double nega_alpha_light(int *board, const int depth, double alpha, double beta, const int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    ++search_param.searched_nodes;
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    else if (depth == 0)
        return evaluate(board);
    bool is_pass = true;
    int i, j, k;
    double v = -65000.0, g;
    int n_board[board_index_num];
    ++p_cnt;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[cell][i] != -1){
                if (board_param.legal[board[i]][board_param.put[cell][i]]){
                    is_pass = false;
                    g = -nega_alpha_light(n_board, depth - 1, -beta, -alpha, 0, o_open_val, p_open_val + eval_param.open_eval[move_open(board, n_board, cell)], o_cnt, p_cnt);
                    if (beta <= g)
                        return g;
                    alpha = max(alpha, g);
                    v = max(v, g);
                    break;
                }
            }
        }
    }
    if (is_pass){
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha_light(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt - 1);
    }
    return v;
}

double nega_alpha(int *board, const int depth, double alpha, double beta, const int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    if (depth < simple_threshold)
        return nega_alpha_light(board, depth, alpha, beta, skip_cnt, p_open_val, o_open_val, p_cnt, o_cnt);
    ++search_param.searched_nodes;
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    else if (depth == 0)
        return evaluate(board);
    int hash = calc_hash(board);
    double lb, ub;
    lb = get_val_hash(search_param.memo_lb, board, hash);
    ub = get_val_hash(search_param.memo_ub, board, hash);
    if (lb != -inf){
        alpha = max(alpha, lb);
        if (alpha >= beta)
            return alpha;
    }
    if (ub != -inf){
        beta = min(beta, ub);
        if (alpha >= beta)
            return beta;
    }
    int i, j, k, canput = 0;
    double v = -65000.0, g;
    board_priority lst[30];
    ++p_cnt;
    double previous_val;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                lst[canput].n_open_val = p_open_val + eval_param.open_eval[move_open(board, lst[canput].b, cell)];
                previous_val = get_val_hash(search_param.previous_memo, lst[canput].b, calc_hash(lst[canput].b));
                if (previous_val != -inf){
                    lst[canput].priority = 1000.0 + previous_val;
                } else {
                    lst[canput].priority = lst[canput].n_open_val / p_cnt - o_open_val / max(1, o_cnt);
                }
                ++canput;
                break;
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt - 1);
    }
    if (canput > 1)
        sort(lst, lst + canput, cmp);
    for (i = 0; i < canput; ++i){
        g = -nega_alpha(lst[i].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
        if (fabs(g) == inf)
            return -inf;
        if (beta < g){
            register_hash(search_param.memo_lb, board, hash, g);
            return g;
        }
        alpha = max(alpha, g);
        v = max(v, g);
    }
    if (v == alpha)
        register_hash(search_param.memo_lb, board, hash, v);
    register_hash(search_param.memo_ub, board, hash, v);
    return v;
}

double nega_scout(int *board, int depth, double alpha, double beta, int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (depth < simple_threshold)
        return nega_alpha_light(board, depth, alpha, beta, skip_cnt, p_open_val, o_open_val, p_cnt, o_cnt);
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    if (depth == 0)
        return evaluate(board);
    int hash = calc_hash(board);
    double lb, ub;
    lb = get_val_hash(search_param.memo_lb, board, hash);
    ub = get_val_hash(search_param.memo_ub, board, hash);
    if (lb != -inf){
        alpha = max(alpha, lb);
        if (alpha >= beta)
            return alpha;
    }
    if (ub != -inf){
        beta = min(beta, ub);
        if (alpha >= beta)
            return beta;
    }
    int i, j, canput = 0;
    board_priority lst[30];
    int n_p_cnt = p_cnt + 1, n_depth = depth - 1;
    double previous_val;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                previous_val = get_val_hash(search_param.previous_memo, lst[canput].b, calc_hash(lst[canput].b));
                if (previous_val != -inf){
                    lst[canput].priority = 1000.0 + previous_val;
                    lst[canput].n_open_val = p_open_val + eval_param.open_eval[move_open(board, lst[canput].b, cell)];
                } else {
                    lst[canput].priority = eval_param.open_eval[move_open(board, lst[canput].b, cell)];
                    lst[canput].n_open_val = p_open_val + lst[canput].priority;
                }
                ++canput;
                break;
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_scout(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt);
    }
    if (canput > 2)
        sort(lst, lst + canput, cmp);
    double v, g;
    g = -nega_scout(lst[0].b, n_depth, -beta, -alpha, 0, o_open_val, lst[0].n_open_val, o_cnt, n_p_cnt);
    if (fabs(g) == inf)
        return -inf;
    if (beta <= g)
        return g;
    v = g;
    alpha = max(alpha, g);
    for (i = 1; i < canput; ++i){
        g = -nega_alpha(lst[i].b, n_depth, -alpha - window, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, n_p_cnt);
        if (fabs(g) == inf)
            return -inf;
        if (beta <= g)
            return g;
        if (alpha < g){
            alpha = g;
            g = -nega_scout(lst[i].b, n_depth, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, n_p_cnt);
            if (beta <= g)
                return g;
            alpha = max(alpha, g);
        }
        v = max(v, g);
    }
    return v;
}

double nega_scout_heavy(int *board, int depth, double alpha, double beta, int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (depth <= search_param.max_depth - 3)
        return nega_scout(board, depth, alpha, beta, skip_cnt, p_open_val, o_open_val, p_cnt, o_cnt);
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    if (depth == 0)
        return evaluate(board);
    int hash = calc_hash(board);
    double lb, ub;
    lb = get_val_hash(search_param.memo_lb, board, hash);
    ub = get_val_hash(search_param.memo_ub, board, hash);
    if (lb != -inf){
        alpha = max(alpha, lb);
        if (alpha >= beta)
            return alpha;
    }
    if (ub != -inf){
        beta = min(beta, ub);
        if (alpha >= beta)
            return beta;
    }
    int i, j, canput = 0;
    board_priority lst[30];
    int n_p_cnt = p_cnt + 1, n_depth = depth - 1;
    open_vals tmp_open_vals;
    double previous_val;
    for (j = 0; j < search_param.vacant_cnt; ++j){
        for (i = 0; i < board_param.put_idx_num[search_param.vacant_lst[j]]; ++i){
            if (board_param.legal[board[board_param.put_idx[search_param.vacant_lst[j]][i]]][board_param.put[search_param.vacant_lst[j]][board_param.put_idx[search_param.vacant_lst[j]][i]]]){
                lst[canput].n_open_val = p_open_val + eval_param.open_eval[move_open(board, lst[canput].b, search_param.vacant_lst[j])];
                previous_val = get_val_hash(search_param.previous_memo, lst[canput].b, calc_hash(lst[canput].b));
                if (previous_val != -inf){
                    lst[canput].priority = 1000.0 + previous_val;
                } else {
                    tmp_open_vals = open_val_forward(lst[canput].b, 1, true);
                    //cerr << tmp_open_vals.p_cnt << " " << tmp_open_vals.p_open_val << " " << tmp_open_vals.o_cnt << " " << tmp_open_vals.o_open_val << endl;
                    if (o_cnt + tmp_open_vals.p_cnt)
                        lst[canput].priority = (lst[canput].n_open_val + tmp_open_vals.o_open_val) / (n_p_cnt + tmp_open_vals.o_cnt) - (o_open_val + tmp_open_vals.p_open_val) / (o_cnt + tmp_open_vals.p_cnt);
                    else
                        lst[canput].priority = (lst[canput].n_open_val + tmp_open_vals.o_open_val) / (n_p_cnt + tmp_open_vals.o_cnt);
                }
                ++canput;
                break;
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_scout_heavy(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt);
    }
    if (canput > 2)
        sort(lst, lst + canput, cmp);
    double v, g;
    g = -nega_scout_heavy(lst[0].b, n_depth, -beta, -alpha, 0, o_open_val, lst[0].n_open_val, o_cnt, n_p_cnt);
    if (fabs(g) == inf)
        return -inf;
    if (beta <= g)
        return g;
    v = g;
    alpha = max(alpha, g);
    for (i = 1; i < canput; ++i){
        g = -nega_alpha(lst[i].b, n_depth, -alpha - window, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, n_p_cnt);
        if (fabs(g) == inf)
            return -inf;
        if (beta <= g)
            return g;
        if (alpha < g){
            alpha = g;
            g = -nega_scout_heavy(lst[i].b, n_depth, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, n_p_cnt);
            if (beta <= g)
                return g;
            alpha = max(alpha, g);
        }
        v = max(v, g);
    }
    return v;
}

double map_double(double y1, double y2, double y3, double x){
    double a, b, c;
    double x1 = 4.0 / hw2, x2 = 25.0 / hw2, x3 = 64.0 / hw2;
    a = ((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2)) / ((x1 - x2) * (x1 - x3) * (x2 - x3));
    b = (y1 - y2) / (x1 - x2) - a * (x1 + x2);
    c = y1 - a * x1 * x1 - b * x1;
    return a * x * x + b * x + c;
}

double map_linar(double s, double e, double x){
    return s + (e - s) * x;
}

int cmp_main(board_priority_move p, board_priority_move q){
    return p.priority > q.priority;
}

int cmp_vacant(int p, int q){
    return eval_param.weight[p] > eval_param.weight[q];
}

int main(){
    int outy, outx, i, j, k, canput, former_depth = 7, former_vacant = hw2 - 4;
    double score, max_score;
    unsigned long long p, o;
    int board[board_index_num];
    int put;
    vector<board_priority_move> lst;
    char elem;
    int action_count;
    double game_ratio;
    int ai_player;
    int board_tmp;
    int y, x;
    double final_score;
    int board_size;
    string action;
    double avg_score;
    int avg_div_num;

    init();
    cerr << "AI initialized" << endl;
    
    while (true){
        outy = -1;
        outx = -1;
        avg_score = 0.0;
        avg_div_num = 0;
        search_param.vacant_cnt = 0;
        search_param.vacant_lst.clear();
        p = 0;
        o = 0;
        canput = 0;
        search_param.weak_mode = 0;
        cin >> ai_player;
        cin >> search_param.tl;
        if (search_param.tl < 0) {
            if (search_param.tl == -10) {
                search_param.tl = -search_param.tl;
                search_param.weak_mode = 2;
            } else {
                search_param.tl = -search_param.tl;
                search_param.weak_mode = 1;
            }
        }
        //cerr << "AI: " << ai_player << " timeout in " << search_param.tl << "ms" << endl;
        for (i = 0; i < hw2; ++i){
            //if (i && !(i & 0b111))
            //    cerr << endl;
            cin >> elem;
            //cerr << elem;
            if (elem == '.'){
                //search_param.vacant_lst[search_param.vacant_cnt++] = i;
                search_param.vacant_lst.push_back(i);
                ++search_param.vacant_cnt;
            }else{
                p |= (unsigned long long)(elem == '0') << i;
                o |= (unsigned long long)(elem == '1') << i;
            }
        }
        //cerr << endl;
        if (search_param.vacant_cnt > 1){
            //sort(search_param.vacant_lst, search_param.vacant_lst + search_param.vacant_cnt, cmp_vacant);
            sort(search_param.vacant_lst.begin(), search_param.vacant_lst.end(), cmp_vacant);
        }
        if (ai_player == 1)
            swap(p, o);
        search_param.strt = tim();
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
        //evaluate(board);
        //return 0;
        search_param.min_max_depth = 2; //min(10, former_depth + search_param.vacant_cnt - former_vacant);            
        search_param.max_depth = search_param.min_max_depth;
        former_vacant = search_param.vacant_cnt;
        lst.clear();
        for (j = 0; j < search_param.vacant_cnt; ++j){
            for (i = 0; i < board_index_num; ++i){
                if (board_param.put[search_param.vacant_lst[j]][i] != -1){
                    if (board_param.legal[board[i]][board_param.put[search_param.vacant_lst[j]][i]]){
                        ++canput;
                        board_priority_move tmp;
                        tmp.open_val = eval_param.open_eval[move_open(board, tmp.b, search_param.vacant_lst[j])];
                        tmp.priority = tmp.open_val;
                        tmp.move = search_param.vacant_lst[j];
                        lst.push_back(tmp);
                        break;
                    }
                }
            }
        }
        if (canput > 1)
            sort(lst.begin(), lst.end(), cmp_main);
        outy = -1;
        outx = -1;
        search_param.searched_nodes = 0;
        while (tim() - search_param.strt < search_param.tl){
            hash_table_init(search_param.previous_memo);
            //hash_table_copy(search_param.previous_memo, search_param.memo_ub);
            swap(search_param.previous_memo, search_param.memo_ub);
            hash_table_init(search_param.memo_lb);
            //hash_table_init(search_param.memo_ub);
            search_param.turn = min(63, hw2 - search_param.vacant_cnt + search_param.max_depth);
            game_ratio = (double)search_param.turn / hw2;
            score = -nega_scout_heavy(lst[0].b, search_param.max_depth - 1, -65000.0, 65000.0, 0, 0.0, lst[0].open_val, 0, 1);
            if (fabs(score) == inf){
                max_score = -inf;
            } else {
                max_score = score;
                lst[0].priority = score;
                for (i = 1; i < canput; ++i){
                    score = -nega_alpha(lst[i].b, search_param.max_depth - 1, -max_score - window, -max_score, 0, 0.0, lst[i].open_val, 0, 1);
                    if (max_score <= score){
                        max_score = score;
                        score = -nega_scout_heavy(lst[i].b, search_param.max_depth - 1, -65000.0, -max_score, 0, 0.0, lst[i].open_val, 0, 1);
                        if (fabs(score) == inf){
                            max_score = -inf;
                            break;
                        }
                        lst[i].priority = score;
                        max_score = score;
                    } else
                        lst[i].priority = score;
                }
            }
            if (max_score == -inf){
                //cerr << "depth " << search_param.max_depth << " timeoout" << endl;
                break;
            }
            final_score = max_score;
            avg_score += final_score;
            ++avg_div_num;
            former_depth = search_param.max_depth;
            if (canput > 1)
                sort(lst.begin(), lst.end(), cmp_main);
            outx = lst[0].move % hw;
            outy = lst[0].move / hw;
            //cerr << "depth " << search_param.max_depth << " nodes " << search_param.searched_nodes << " nps " << ((unsigned long long)search_param.searched_nodes * 1000 / max(1, tim() - search_param.strt));
            //cerr << "  " << outy << outx << " " << lst[0].priority;
            //cerr << " time " << tim() - search_param.strt << endl;
            if (fabs(max_score) >= 1000.0 || search_param.max_depth >= hw2){
                avg_score = max_score / 1000.0;
                avg_div_num = 1;
                //cerr << "game end" << endl;
                break;
            }
            ++search_param.max_depth;
        }
        //cout << (char)(outx + 97) << (outy + 1) << " " << "MSG " << former_depth << " " << final_score << endl;
        avg_score /= avg_div_num;
        if (search_param.weak_mode)
            avg_score = -avg_score;
        //cerr << (char)(outx + 97) << (outy + 1) << " " << avg_score << " " << tim() - search_param.strt << "ms" << endl;
        cout << outy << " " << outx << " " << avg_score << endl;
    }
    return 0;
}
