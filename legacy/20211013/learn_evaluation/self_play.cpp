#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx")

// Egaroucid2

#include <iostream>
#include <fstream>
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
#define n_line 6561

#define inf 2000000000.0
#define b_idx_num 38

#define n_div 1000000
#define tanh_min -5.0
#define tanh_max 5.0
#define exp_min -30.0
#define exp_max 30.0

#define compress_digit 16
#define ln_char 2
#define ln_repair 90
#define n_board_input 3
#define kernel_size 3
#define n_kernels 50
#define n_dense1_value 16
#define conv_size (hw_p1 - kernel_size)
#define conv_padding (kernel_size / 2)
#define conv_padding2 (conv_padding * 2)

#define evaluate_count 100
#define mcts_comp_stones 8
#define comp_stones 8
double c_puct; // 0.7
double p_offset; // 0.05
double div_puct; // 0.1

#define hash_table_size 16384
#define hash_mask (hash_table_size - 1)
#define book_stones (16 + 4)
#define ln_repair_book 27

int xorx=123456789, xory=362436069, xorz=521288629, xorw=88675123;
inline double myrandom(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = xorw=(xorw^(xorw>>19))^(t^(t>>8));
    return (double)(xorw) / 2147483648.0;
}

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline int map_liner(double x, double mn, double mx){
    return max(0, min(n_div - 1, (int)((x - mn) / (mx - mn) * n_div)));
}

inline double rev_map_liner(int x, double mn, double mx){
    return (double)x / (double)n_div * (mx - mn) + mn;
}

int cmp_vacant(int p, int q);

struct predictions{
    double policies[hw2];
    double value;
};

class board_c{
    private:
        int reverse[n_line];
        int cnt_p[n_line], cnt_o[n_line];
        int restore_p[n_line][hw], restore_o[n_line][hw], restore_vacant[n_line][hw];
        bool legal[n_line][hw];
        unsigned long long trans_move[b_idx_num][n_line][hw];
        int bit2idx[b_idx_num][hw];
        int idx2bit[hw2][4][2];
        int pattern_space[b_idx_num];
        int put[hw2][b_idx_num];
        int pow3[15];
        int rev_bit3[n_line][hw];
        int pop_digit[n_line][hw];
        int turn_board[4][hw2];

    public:
        int direction;
        int ai_player;
        vector<int> vacant_lst;
        double weight[hw2];
        int n_stones;

    private:
        inline int reverse_line(int a) {
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
            int p_rev = board_c::reverse_line(p), o_rev = board_c::reverse_line(o);
            int p2 = p_rev << 1;
            res |= board_c::reverse_line(~(p2 | o_rev) & (p2 + o_rev));
            res &= ~(p | o);
            return res;
        }

        inline int move_trans(int pt, int k) {
            if (k == 0)
                return pt >> 1;
            else
                return pt << 1;
        }

        inline int move_line(int p, int o, const int place) {
            int rev = 0;
            int rev2, mask, tmp;
            int pt = 1 << place;
            for (int k = 0; k < 2; ++k) {
                rev2 = 0;
                mask = board_c::move_trans(pt, k);
                while (mask && (mask & o)) {
                    rev2 |= mask;
                    tmp = mask;
                    mask = board_c::move_trans(tmp, k);
                    if (mask & p)
                        rev |= rev2;
                }
            }
            return rev | pt;
        }

        inline int create_po(int idx, int check){
            int res = 0;
            for (int i = 0; i < hw; ++i){
                if (idx % 3 == check){
                    res |= 1 << i;
                }
                idx /= 3;
            }
            return res;
        }

        inline int board_reverse(int idx){
            int p = board_c::create_po(idx, 1);
            int o = board_c::create_po(idx, 2);
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

    public:
        inline void init(){
            int p, o, mobility, canput_num, rev, i, j, k, l;
            const int translate[hw2] = {
                0, 1, 2, 3, 3, 2, 1, 0,
                1, 4, 5, 6, 6, 5, 4, 1,
                2, 5, 7, 8, 8, 7, 5, 2,
                3, 6, 8, 9, 9, 8, 6, 3,
                3, 6, 8, 9, 9, 8, 6, 3,
                2, 5, 7, 8, 8, 7, 5, 2,
                1, 4, 5, 6, 6, 5, 4, 1,
                0, 1, 2, 3, 3, 2, 1, 0
            };
            const double weight10[10] = {
                0.2880, -0.1150, 0.0000, -0.0096,
                        -0.1542, -0.0288, -0.0288,
                                0.0000, -0.0096,
                                        -0.0096
            };
            const int consts[476] = {
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63, 5, 14, 23, 4, 13, 22, 31, 3, 12, 21, 30, 39, 2, 11, 20, 29, 38, 47, 1, 10, 19, 28, 37, 46, 55, 0, 9, 18, 27, 36, 45, 54, 63, 8,
                17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52, 61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 10, 8, 8, 8, 8, 4, 4, 8, 2, 4, 54, 63, 62, 61, 60, 59, 58, 57,
                56, 49, 49, 56, 48, 40, 32, 24, 16, 8, 0, 9, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14, 14, 7, 15, 23, 31, 39, 47, 55, 63, 54, 3, 2, 1, 0, 9, 8, 16, 24, 4, 5, 6, 7, 14, 15, 23, 31, 60, 61, 62, 63, 54, 55, 47, 39, 59, 58, 57, 56, 49, 48, 40, 32, 0, 1, 2, 3, 8, 9, 10, 11, 0, 8, 16, 24, 1, 9, 17, 25, 7, 6, 5, 4, 15, 14, 13, 12, 7, 15, 23, 31, 6, 14, 22, 30, 63, 62, 61, 60,
                55, 54, 53, 52, 63, 55, 47, 39, 62, 54, 46, 38, 56, 57, 58, 59, 48, 49, 50, 51, 56, 48, 40, 32, 57, 49, 41, 33, 0, 9, 18, 27, 36, 45, 54, 63, 7, 14, 21, 28, 35, 42, 49, 56, 0, 1, 2, 3, 4, 5, 6, 7, 7, 15, 23, 31, 39, 47, 55, 63, 63, 62, 61, 60, 59, 58, 57, 56, 56, 48, 40, 32, 24, 26, 8, 0
            };
            for (i = 0; i < hw2; ++i)
                board_c::weight[i] = weight10[translate[i]];
            int all_idx = 0;
            for (i = 0; i < b_idx_num; ++i)
                board_c::pattern_space[i] = consts[all_idx++];
            for (i = 0; i < b_idx_num; ++i){
                for (j = 0; j < board_c::pattern_space[i]; ++j)
                    board_c::bit2idx[i][j] = consts[all_idx++];
            }
            int idx;
            for (i = 0; i < hw2; ++i){
                idx = 0;
                for (j = 0; j < b_idx_num; ++j){
                    for (k = 0; k < board_c::pattern_space[j]; ++k){
                        if (board_c::bit2idx[j][k] == i){
                            board_c::idx2bit[i][idx][0] = j;
                            board_c::idx2bit[i][idx++][1] = k;
                        }
                    }
                }
                for (j = idx; j < 4; ++j)
                    board_c::idx2bit[i][j][0] = -1;
            }
            for (i = 0; i < hw2; ++i){
                for (j = 0; j < b_idx_num; ++j){
                    board_c::put[i][j] = -1;
                    for (k = 0; k < board_c::pattern_space[j]; ++k){
                        if (board_c::bit2idx[j][k] == i)
                            board_c::put[i][j] = k;
                    }
                }
            }
            for (i = 0; i < n_line; ++i){
                board_c::reverse[i] = board_c::board_reverse(i);
                p = board_c::reverse_line(create_po(i, 1));
                o = board_c::reverse_line(create_po(i, 2));
                board_c::cnt_p[i] = 0;
                board_c::cnt_o[i] = 0;
                for (j = 0; j < hw; ++j){
                    board_c::restore_p[i][j] = 1 & (p >> (hw_m1 - j));
                    board_c::restore_o[i][j] = 1 & (o >> (hw_m1 - j));
                    board_c::restore_vacant[i][j] = 1 & ((~(p | o)) >> (hw_m1 - j));
                    board_c::cnt_p[i] += board_c::restore_p[i][j];
                    board_c::cnt_o[i] += board_c::restore_o[i][j];
                }
                mobility = board_c::check_mobility(p, o);
                canput_num = 0;
                for (j = 0; j < hw; ++j){
                    if (1 & (mobility >> (hw_m1 - j))){
                        rev = board_c::move_line(p, o, hw_m1 - j);
                        ++canput_num;
                        board_c::legal[i][j] = true;
                        for (k = 0; k < b_idx_num; ++k){
                            board_c::trans_move[k][i][j] = 0;
                            for (l = 0; l < board_c::pattern_space[k]; ++l)
                                board_c::trans_move[k][i][j] |= (unsigned long long)(1 & (rev >> (7 - l))) << board_c::bit2idx[k][l];
                        }
                    } else
                        board_c::legal[i][j] = false;
                }
            }
            for (i = 0; i < 15; ++i)
                board_c::pow3[i] = (int)pow(3, i);
            for (i = 0; i < n_line; ++i){
                for (j = 0; j < hw; ++j){
                    board_c::rev_bit3[i][j] = board_c::pow3[j] * (2 - (i / board_c::pow3[j]) % 3);
                    board_c::pop_digit[i][j] = i / board_c::pow3[j] % 3;
                }
            }
            for (i = 0; i < hw; ++i){
                for (j = 0; j < hw; ++j){
                    board_c::turn_board[0][i * hw + j] = i * hw + j;
                    board_c::turn_board[1][i * hw + j] = j * hw + i;
                    board_c::turn_board[2][i * hw + j] = (hw_m1 - i) * hw + (hw_m1 - j);
                    board_c::turn_board[3][i * hw + j] = (hw_m1 - j) * hw + (hw_m1 - i);
                }
            }
            board_c::direction = -1;
        }

        inline void print_board(const int* board){
            int i, j, idx, tmp;
            for (i = 0; i < hw; ++i){
                tmp = board[i];
                for (j = 0; j < hw; ++j){
                    if (tmp % 3 == 0){
                        //cerr << ". ";
                    }else if (tmp % 3 == 1){
                        //cerr << "P ";
                    }else{
                        //cerr << "O ";
                    }
                    tmp /= 3;
                }
                //cerr << endl;
            }
            //cerr << endl;
        }

        inline int input_board(int (&board)[b_idx_num]){
            int i, j;
            unsigned long long p = 0, o = 0;
            char elem;
            int vacant_cnt = 0;
            int action_count;
            board_c::n_stones = 0;
            board_c::vacant_lst = {};
            for (i = 0; i < hw2; ++i){
                cin >> elem;
                if (elem != '.'){
                    p |= (unsigned long long)(elem == '0') << board_c::turn_board[board_c::direction][i];
                    o |= (unsigned long long)(elem == '1') << board_c::turn_board[board_c::direction][i];
                    ++board_c::n_stones;
                } else{
                    ++vacant_cnt;
                    board_c::vacant_lst.push_back(board_c::turn_board[board_c::direction][i]);
                }
            }
            if (board_c::ai_player == 1)
                swap(p, o);
            if (vacant_cnt > 1)
                sort(board_c::vacant_lst.begin(), board_c::vacant_lst.end(), cmp_vacant);
            for (i = 0; i < b_idx_num; ++i){
                board[i] = 0;
                for (j = 0; j < board_c::pattern_space[i]; ++j){
                    if (1 & (p >> board_c::bit2idx[i][j]))
                        board[i] += board_c::pow3[j];
                    else if (1 & (o >> board_c::bit2idx[i][j]))
                        board[i] += 2 * board_c::pow3[j];
                }
            }
            return vacant_cnt;
        }

        inline void create_predict_board(const int *board, double input_board[n_board_input][hw + conv_padding2][hw + conv_padding2]){
            int i, j, k;
            for (i = 0; i < hw; ++i){
                for (j = 0; j < hw; ++j){
                    input_board[0][i + conv_padding][j + conv_padding] = board_c::restore_p[board[i]][j];
                    input_board[1][i + conv_padding][j + conv_padding] = board_c::restore_o[board[i]][j];
                    input_board[2][i + conv_padding][j + conv_padding] = board_c::restore_vacant[board[i]][j];
                }
                for (k = 0; k < n_board_input; ++k){
                    for (j = 0; j < hw + conv_padding2; ++j){
                        input_board[k][0][j] = 0.0;
                        input_board[k][hw_m1 + conv_padding2][j] = 0.0;
                        input_board[k][j][0] = 0.0;
                        input_board[k][j][hw_m1 + conv_padding2] = 0.0;
                    }
                }
            }
        }

        inline bool check_legal(const int *board, int cell){
            int i;
            for (i = 0; i < b_idx_num; ++i){
                if (board_c::put[cell][i] != -1){
                    if (board_c::legal[board[i]][board_c::put[cell][i]])
                        return true;
                }
            }
            return false;
        }

        inline void move(const int *board, int res[b_idx_num], int coord){
            int i, j, tmp;
            unsigned long long rev = 0;
            for (i = 0; i < b_idx_num; ++i){
                res[i] = board_c::reverse[board[i]];
                if (board_c::put[coord][i] != -1)
                    rev |= board_c::trans_move[i][board[i]][board_c::put[coord][i]];
            }
            for (i = 0; i < hw2; ++i){
                if (1 & (rev >> i)){
                    for (j = 0; j < 4; ++j){
                        if (board_c::idx2bit[i][j][0] == -1)
                            break;
                        res[board_c::idx2bit[i][j][0]] += board_c::rev_bit3[res[board_c::idx2bit[i][j][0]]][board_c::idx2bit[i][j][1]];
                    }
                }
            }
        }

        inline void reverse_board(const int *board, int n_board[b_idx_num]){
            int i;
            for (i = 0; i < b_idx_num; ++i)
                n_board[i] = board_c::reverse[board[i]];
        }

        inline double end_game(const int *board){
            int res = 0, i, j, p, o;
            for (i = 0; i < hw; ++i){
                res += board_c::cnt_p[board[i]];
                res -= board_c::cnt_o[board[i]];
            }
            if (res > 0)
                return 1.0;
            else if (res < 0)
                return -1.0;
            return 0.0;
        }

        inline string coord_str(int policy){
            string res;
            res += (char)(board_c::turn_board[board_c::direction][policy] % hw + 97);
            res += to_string(board_c::turn_board[board_c::direction][policy] / hw + 1);
            return res;
        }
};
board_c board_c;

int cmp_vacant(int p, int q){
    return board_c.weight[p] > board_c.weight[q];
}

class predict_c{
    private:
        double tanh_arr[n_div];

        double conv1[n_kernels][n_board_input][kernel_size][kernel_size];
        double conv_residual[n_kernels][n_kernels][kernel_size][kernel_size];

        double dense1_policy[hw2][n_kernels];
        double bias1_policy[hw2];

        double dense1_value[n_dense1_value][n_kernels];
        double bias1_value[n_dense1_value];
        double dense2_value[n_dense1_value];
        double bias2_value;

        inline double get_element(char *cbuf, FILE *fp){
            if (!fgets(cbuf, 1024, fp)){
                //cerr << "param file broken" << endl;
                exit(1);
            }
            return atof(cbuf);
        }

        inline double leaky_relu(double x){
            return max(x, 0.01 * x);
        }

    public:
        inline void init(){
            int i, j, k, l;
            for (i = 0; i < n_div; ++i)
                predict_c::tanh_arr[i] = tanh(rev_map_liner(i, tanh_min, tanh_max));

            FILE *fp;
            char cbuf[1024];
            int mode;
            if ((fp = fopen("param/param.txt", "r")) == NULL){
                printf("param file not exist");
                exit(1);
            }
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < n_board_input; ++j){
                    for (k = 0; k < kernel_size; ++k){
                        for (l = 0; l < kernel_size; ++l){
                            predict_c::conv1[i][j][k][l] = get_element(cbuf, fp);
                        }
                    }
                }
            }
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < n_kernels; ++j){
                    for (k = 0; k < kernel_size; ++k){
                        for (l = 0; l < kernel_size; ++l){
                            predict_c::conv_residual[i][j][k][l] = get_element(cbuf, fp);
                        }
                    }
                }
            }
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < n_dense1_value; ++j){
                    predict_c::dense1_value[j][i] = get_element(cbuf, fp);
                }
            }
            for (i = 0; i < n_dense1_value; ++i){
                predict_c::bias1_value[i] = get_element(cbuf, fp);
            }
            for (i = 0; i < n_kernels; ++i){
                for (j = 0; j < hw2; ++j){
                    predict_c::dense1_policy[j][i] = get_element(cbuf, fp);
                }
            }
            for (i = 0; i < hw2; ++i){
                predict_c::bias1_policy[i] = get_element(cbuf, fp);
            }

            for (i = 0; i < n_dense1_value; ++i){
                predict_c::dense2_value[i] = get_element(cbuf, fp);
            }
            predict_c::bias2_value = get_element(cbuf, fp);
        }

        inline predictions predict(const int *board){
            int i, j, k, sy, sx, y, x, coord1, coord2, residual_i;
            predictions res;
            double input_board[n_board_input][hw + conv_padding2][hw + conv_padding2];
            double hidden_conv1[n_kernels][hw + conv_padding2][hw + conv_padding2];
            double hidden_conv2[n_kernels][hw + conv_padding2][hw + conv_padding2];
            double after_conv[n_kernels];
            double hidden1[64];
            // reshape input
            board_c.create_predict_board(board, input_board);
            // conv and leaky-relu
            for (i = 0; i < n_kernels; ++i){
                for (y = 0; y < hw + conv_padding2; ++y){
                    for (x = 0; x < hw + conv_padding2; ++x)
                        hidden_conv1[i][y][x] = 0.0;
                }
                for (j = 0; j < n_board_input; ++j){
                    for (sy = 0; sy < hw; ++sy){
                        for (sx = 0; sx < hw; ++sx){
                            for (y = 0; y < kernel_size; ++y){
                                for (x = 0; x < kernel_size; ++x)
                                    hidden_conv1[i][sy + conv_padding][sx + conv_padding] += predict_c::conv1[i][j][y][x] * input_board[j][sy + y][sx + x];
                            }
                        }
                    }
                }
                for (y = conv_padding; y < hw + conv_padding; ++y){
                    for (x = conv_padding; x < hw + conv_padding; ++x)
                        hidden_conv1[i][y][x] = predict_c::leaky_relu(hidden_conv1[i][y][x]);
                }
            }
            // residual-error-block
            for (i = 0; i < n_kernels; ++i){
                for (y = 0; y < hw + conv_padding2; ++y){
                    for (x = 0; x < hw + conv_padding2; ++x)
                        hidden_conv2[i][y][x] = 0.0;
                }
                for (j = 0; j < n_kernels; ++j){
                    for (sy = 0; sy < hw; ++sy){
                        for (sx = 0; sx < hw; ++sx){
                            for (y = 0; y < kernel_size; ++y){
                                for (x = 0; x < kernel_size; ++x)
                                    hidden_conv2[i][sy + conv_padding][sx + conv_padding] += predict_c::conv_residual[i][j][y][x] * hidden_conv1[j][sy + y][sx + x];
                            }
                        }
                    }
                }
            }
            for (i = 0; i < n_kernels; ++i){
                for (y = conv_padding; y < hw + conv_padding; ++y){
                    for (x = conv_padding; x < hw + conv_padding; ++x)
                        hidden_conv1[i][y][x] = predict_c::leaky_relu(hidden_conv1[i][y][x] + hidden_conv2[i][y][x]);
                }
            }
            // global-average-pooling
            for (i = 0; i < n_kernels; ++i){
                after_conv[i] = 0.0;
                for (y = 0; y < hw; ++y){
                    for (x = 0; x < hw; ++x)
                        after_conv[i] += hidden_conv1[i][y + conv_padding][x + conv_padding];
                }
                after_conv[i] /= hw2;
            }
            // tanh for policy
            for (i = 0; i < n_kernels; ++i)
                hidden1[i] = predict_c::tanh_arr[map_liner(after_conv[i], tanh_min, tanh_max)];
            // dense1 for policy
            for (j = 0; j < hw2; ++j){
                res.policies[j] = predict_c::bias1_policy[j];
                for (i = 0; i < n_kernels; ++i)
                    res.policies[j] += predict_c::dense1_policy[j][i] * hidden1[i];
            }
            // dense1 for value
            for (j = 0; j < n_dense1_value; ++j){
                hidden1[j] = predict_c::bias1_value[j];
                for (i = 0; i < n_kernels; ++i)
                    hidden1[j] += predict_c::dense1_value[j][i] * after_conv[i];
                hidden1[j] = predict_c::leaky_relu(hidden1[j]);
            }
            // dense3 for value
            res.value = predict_c::bias2_value;
            for (i = 0; i < n_dense1_value; ++i)
                res.value += predict_c::dense2_value[i] * hidden1[i];
            res.value = predict_c::tanh_arr[map_liner(res.value, tanh_min, tanh_max)];

            // return
            return res;
        }
};

predict_c predict_c;

class book_c{
    private:        
        struct node_t{
            int k[hw];
            int policy;
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

        inline node_t* node_init(const int *key, int policy){
            node_t* p_node = NULL;
            p_node = (node_t*)malloc(sizeof(node_t));
            for (int i = 0; i < hw; ++i)
                p_node->k[i] = key[i];
            p_node->policy = policy;
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

        inline void register_hash(node_t** hash_table, const int *key, int hash, int policy){
            if(hash_table[hash] == NULL){
                hash_table[hash] = node_init(key, policy);
            } else {
                node_t *p_node = p_node = hash_table[hash];
                node_t *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(compare_key(key, p_node->k)){
                        p_node->policy = policy;
                        return;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = node_init(key, policy);
            }
        }

        int n_book = 0;
        unordered_map<char, int> char_keys;
        book_c::node_t *book[hash_table_size];
        const string chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abc";
        string param_compressed1;

    public:
        inline int get_val_hash(const int *key){
            node_t *p_node = book_c::book[calc_hash(key)];
            while(p_node != NULL){
                if(compare_key(key, p_node->k)){
                    return p_node->policy;
                }
                p_node = p_node->p_n_node;
            }
            return -1;
        }

        inline void init(){
            int i, j;
            for (i = 0; i < hw2; ++i)
                book_c::char_keys[book_c::chars[i]] = i;
            ifstream ifs("book/param/book.txt");
            if (ifs.fail()){
                //cerr << "book file not exist" << endl;
                exit(1);
            }
            getline(ifs, book_c::param_compressed1);
            int ln = book_c::param_compressed1.length();
            int coord;
            int board[b_idx_num], n_board[b_idx_num];
            const int first_board[b_idx_num] = {0, 0, 0, 189, 702, 0, 0, 0, 0, 0, 0, 189, 216, 162, 0, 0, 0, 0, 0, 0, 216, 189, 54, 0, 0, 0, 0, 0, 0, 0, 0, 27, 216, 54, 18, 0, 0, 0};
            book_c::hash_table_init(book_c::book);
            int data_idx = 0;
            while (data_idx < ln){
                for (i = 0; i < b_idx_num; ++i)
                    board[i] = first_board[i];
                while (true){
                    if (param_compressed1[data_idx] == ' '){
                        ++data_idx;
                        break;
                    }
                    coord = book_c::char_keys[param_compressed1[data_idx++]];
                    board_c.move(board, n_board, coord);
                    swap(board, n_board);
                }
                coord = book_c::char_keys[param_compressed1[data_idx++]];
                ////cerr << coord << endl;
                //board_c.print_board(board);
                book_c::register_hash(book_c::book, board, book_c::calc_hash(board), coord);
                ++book_c::n_book;
            }
            //cerr << book_c::n_book << " boards in book" << endl;
        }
};

//book_c book_c;

struct mcts_node{
    int board[b_idx_num];
    int children[hw2_p1];
    double p[hw2];
    bool legal[hw2];
    double w;
    double v;
    int n_v;
    int n_w;
    bool pass;
    bool expanded;
    int n_canput;
    int n_stones;
};

#define n_s_input 7
#define n_s_dense1 8

class search_c{
    private:
        vector<mcts_node> nodes;
        double exp_arr[n_div];
        double dense1[n_s_dense1][n_s_input];
        double bias1[n_s_dense1];
        double dense2[n_s_dense1];
        double bias2;
        double c_bias;
        double tanh_arr[n_div];
    
    public:
        int used_idx;

    private:
        inline double get_element(char *cbuf, FILE *fp){
            if (!fgets(cbuf, 1024, fp)){
                cerr << "param file broken" << endl;
                exit(1);
            }
            return atof(cbuf);
        }

        inline double leaky_relu(double x){
            return max(x, 0.01 * x);
        }

        double nega_alpha(int *board, double alpha, double beta, const int skip_cnt){
            if (skip_cnt == 2)
                return board_c.end_game(board);
            bool is_pass = true;
            double g;
            int n_board[b_idx_num];
            for (const int& cell : board_c.vacant_lst){
                if (board_c.check_legal(board, cell)){
                    is_pass = false;
                    board_c.move(board, n_board, cell);
                    g = -search_c::nega_alpha(n_board, -beta, -alpha, 0);
                    if (beta <= g)
                        return g;
                    alpha = max(alpha, g);
                }
            }
            if (is_pass){
                board_c.reverse_board(board, n_board);
                return -nega_alpha(n_board, -beta, -alpha, skip_cnt + 1);
            }
            return alpha;
        }

        inline void update_nv(int idx, double value){
            if (fabs(value) > 100.0){
                ++search_c::nodes[idx].n_w;
                if (value > 0.0)
                    search_c::nodes[idx].w += value - 1000.0;
                else
                    search_c::nodes[idx].w += value + 1000.0;
            } else{
                ++search_c::nodes[idx].n_v;
                search_c::nodes[idx].v += value;
            }
        }

        inline double mcts_end(int idx){
            double result = 1000.0 + (double)nega_alpha(search_c::nodes[idx].board, -1.0, 1.0, 0);
            update_nv(idx, result);
            return result;
        }

        inline void expand_children(int idx){
            int i;
            search_c::nodes[idx].expanded = true;
            search_c::nodes[idx].pass = true;
            search_c::nodes[idx].n_canput = 0;
            for (i = 0; i < hw2; ++i)
                search_c::nodes[idx].children[i] = -1;
            for (const int& cell : board_c.vacant_lst){
                search_c::nodes[idx].legal[cell] = false;
                if (board_c.check_legal(search_c::nodes[idx].board, cell)){
                    search_c::nodes[idx].pass = false;
                    search_c::nodes[idx].legal[cell] = true;
                    ++search_c::nodes[idx].n_canput;
                }
            }
            search_c::nodes[idx].children[hw2] = -1;
        }

        inline double expand_policy(int idx){
            predictions pred = predict_c.predict(search_c::nodes[idx].board);
            update_nv(idx, pred.value);
            double p_sum = 0.0;
            for (const int& cell : board_c.vacant_lst){
                if (search_c::nodes[idx].legal[cell]){
                    search_c::nodes[idx].p[cell] = search_c::exp_arr[map_liner(pred.policies[cell], exp_min, exp_max)];
                    p_sum += search_c::nodes[idx].p[cell];
                } else{
                    search_c::nodes[idx].p[cell] = 0.0;
                }
            }
            for (const int& cell : board_c.vacant_lst)
                search_c::nodes[idx].p[cell] /= p_sum;
            return pred.value;
        }


        inline double expand_pass_policy(int idx){
            for (const int& cell : board_c.vacant_lst)
                search_c::nodes[idx].p[cell] = 0.0;
            double value = predict_c.predict(search_c::nodes[idx].board).value;
            update_nv(idx, value);
            return value;
        }

        inline double evaluate_node(double policy, double value, double weight, int n_parent_value, int n_parent_weight, int n_child_value, int n_child_weight){
            double inputs[n_s_input] = {policy, value, weight, (double)n_parent_value, (double)n_parent_weight, (double)n_child_value, (double)n_child_weight};
            double hidden[8];
            double res;
            int i, j;
            for (j = 0; j < n_s_dense1; ++j){
                hidden[j] = search_c::bias1[j];
                for (i = 0; i < n_s_input; ++i)
                    hidden[j] += search_c::dense1[j][i] * inputs[i];
                hidden[j] = search_c::leaky_relu(hidden[j]);
            }
            res = search_c::bias2;
            for (i = 0; i < n_s_dense1; ++i)
                res += search_c::dense2[i] * hidden[i];
            res = search_c::tanh_arr[map_liner(res, tanh_min, tanh_max)];
            return res + c_bias * sqrt((double)(n_parent_value + n_parent_weight)) / (double)(1 + n_child_value + n_child_weight);
        }

        inline int get_next_child(int idx, int depth, int player){
            int a_cell = -1;
            /*
            if (player == board_c.ai_player && search_c::nodes[idx].n_stones < book_stones){
                a_cell = book_c.get_val_hash(search_c::nodes[idx].board);
                if (a_cell != -1)
                    return a_cell;
            }
            */
            double max_priority = -inf;
            double priority;
            for (const int& cell : board_c.vacant_lst){
                if (search_c::nodes[idx].legal[cell]){
                    ////cerr << search_c::nodes[idx].p[cell] << " ";
                    if (search_c::nodes[idx].children[cell] != -1){
                        priority = evaluate_node(search_c::nodes[idx].p[cell], search_c::nodes[search_c::nodes[idx].children[cell]].v, search_c::nodes[search_c::nodes[idx].children[cell]].w, search_c::nodes[idx].n_v, search_c::nodes[idx].n_w, search_c::nodes[search_c::nodes[idx].children[cell]].n_v, search_c::nodes[search_c::nodes[idx].children[cell]].n_w);
                    } else{
                        if (depth > 0)
                            return cell;
                        priority = evaluate_node(search_c::nodes[idx].p[cell], 0.0, 0.0, search_c::nodes[idx].n_v, search_c::nodes[idx].n_w, 0, 0);
                    }
                    if (max_priority < priority){
                        max_priority = priority;
                        a_cell = cell;
                    }
                }
            }
            return a_cell;
        }

        inline void create_node(int idx, int a_cell, int n_stones){
            search_c::nodes[idx].children[a_cell] = search_c::used_idx;
            mcts_node node;
            search_c::nodes.push_back(node);
            search_c::nodes[search_c::used_idx].w = 0.0;
            search_c::nodes[search_c::used_idx].v = 0.0;
            search_c::nodes[search_c::used_idx].n_w = 0;
            search_c::nodes[search_c::used_idx].n_v = 0;
            search_c::nodes[search_c::used_idx].pass = true;
            search_c::nodes[search_c::used_idx].expanded = false;
            search_c::nodes[search_c::used_idx].n_stones = n_stones + 1;
            board_c.move(search_c::nodes[idx].board, search_c::nodes[search_c::used_idx].board, a_cell);
            ++search_c::used_idx;
        }

        inline void create_pass_node(int idx, int n_stones){
            search_c::nodes[idx].children[hw2] = search_c::used_idx;
            mcts_node node;
            search_c::nodes.push_back(node);
            search_c::nodes[search_c::used_idx].w = 0.0;
            search_c::nodes[search_c::used_idx].v = 0.0;
            search_c::nodes[search_c::used_idx].n_w = 0;
            search_c::nodes[search_c::used_idx].n_v = 0;
            search_c::nodes[search_c::used_idx].pass = true;
            search_c::nodes[search_c::used_idx].expanded = false;
            search_c::nodes[search_c::used_idx].n_stones = n_stones;
            board_c.reverse_board(search_c::nodes[idx].board, search_c::nodes[search_c::used_idx].board);
            ++search_c::used_idx;
        }

        double evaluate(int idx, bool passed, int n_stones, int depth, int player){
            double value = 0.0;
            if (n_stones >= hw2 - mcts_comp_stones){
                // find the result of the game
                return search_c::mcts_end(idx);
            }
            if (!search_c::nodes[idx].expanded){
                // expand children
                search_c::expand_children(idx);
                //predict and create policy array
                if (search_c::nodes[idx].pass){
                    // when pass
                    value = search_c::expand_pass_policy(idx);
                } else{
                    value = search_c::expand_policy(idx);
                }
                return value;
            }
            if (!search_c::nodes[idx].pass){
                // when children already expanded
                int a_cell = search_c::get_next_child(idx, depth, player);
                if (search_c::nodes[idx].children[a_cell] == -1){
                    // create child node if tree does not have the node
                    search_c::create_node(idx, a_cell, n_stones);
                }
                value = -search_c::evaluate(search_c::nodes[idx].children[a_cell], false, n_stones + 1, depth - 1, 1 - player);
            } else{
                // when passed
                if (passed){
                    // game over when passed twice
                    value = board_c.end_game(search_c::nodes[idx].board) + 1000.0;
                } else{
                    if (search_c::nodes[idx].children[hw2] == -1){
                        // create node when next node does not exist
                        search_c::create_pass_node(idx, n_stones);
                    }
                    value = -search_c::evaluate(search_c::nodes[idx].children[hw2], true, n_stones, depth, 1 - player);
                }
            }
            update_nv(idx, value);
            return value;
        }

        int get_parent_idx(const int *board, const int n_stones, int idx){
            int i;
            if (search_c::nodes[idx].n_stones == n_stones){
                for (i = 0; i < hw; ++i){
                    if (search_c::nodes[idx].board[i] != board[i])
                        return -1;
                }
                return idx;
            }
            int res;
            for (i = 0; i < hw2; ++i){
                if (search_c::nodes[idx].children[i] != -1){
                    res = get_parent_idx(board, n_stones, search_c::nodes[idx].children[i]);
                    if (res != -1)
                        return res;
                }
            }
            return -1;
        }

        inline void init_parent(const int *board, int n_stones){
            int i;
            int board_idx = 0;
            search_c::nodes = {};
            mcts_node node;
            search_c::nodes.push_back(node);
            search_c::used_idx = board_idx;
            // set parent node
            for (i = 0; i < b_idx_num; ++i)
                search_c::nodes[board_idx].board[i] = board[i];
            search_c::nodes[board_idx].w = 0.0;
            search_c::nodes[board_idx].v = 0.0;
            search_c::nodes[board_idx].n_w = 0;
            search_c::nodes[board_idx].n_v = 0;
            search_c::nodes[board_idx].pass = true;
            search_c::nodes[board_idx].expanded = true;
            search_c::nodes[board_idx].n_stones = n_stones;
            // expand children
            search_c::expand_children(board_idx);
            //predict and create policy array
            search_c::expand_policy(board_idx);
            ++search_c::used_idx;
        }

    public:
        inline void init(){
            int i, j;
            for (i = 0; i < n_div; ++i){
                search_c::exp_arr[i] = exp(rev_map_liner(i, exp_min, exp_max));
                search_c::tanh_arr[i] = tanh(rev_map_liner(i, tanh_min, tanh_max));
            }
            FILE *fp;
            char cbuf[1024];
            int mode;
            cin >> mode;
            if (mode == 0){
                if ((fp = fopen("param/param_eval0.txt", "r")) == NULL){
                    printf("param file not exist");
                    exit(1);
                }
            } else{
                if ((fp = fopen("param/param_eval1.txt", "r")) == NULL){
                    printf("param file not exist");
                    exit(1);
                }
            }
            for (i = 0; i < n_s_input; ++i){
                for (j = 0; j < n_s_dense1; ++j){
                    dense1[j][i] = get_element(cbuf, fp);
                }
            }
            for (i = 0; i < n_s_dense1; ++i){
                bias1[i] = get_element(cbuf, fp);
            }
            for (i = 0; i < n_s_dense1; ++i){
                dense2[i] = get_element(cbuf, fp);
            }
            bias2 = get_element(cbuf, fp);
            c_bias = get_element(cbuf, fp);
        }

        inline void first_search(){
            //cerr << "first search" << endl;
            int i;
            board_c.vacant_lst = {};
            for (i = 0; i < hw2; ++i)
                board_c.vacant_lst.push_back(i);
            int board[b_idx_num] = {0, 0, 0, 189, 702, 0, 0, 0, 0, 0, 0, 189, 216, 162, 0, 0, 0, 0, 0, 0, 216, 189, 54, 0, 0, 0, 0, 0, 0, 0, 0, 27, 216, 54, 18, 0, 0, 0};
            search_c::init_parent(board, 10);
            for (i = 0; i < evaluate_count; ++i){
                search_c::evaluate(0, false, 5, 10, 1);
            }
            //cerr << i << " times searched" << endl;
        }

        inline int mcts(const int *board, int former_idx, int player){
            int i, cell;
            int board_idx = search_c::get_parent_idx(board, board_c.n_stones, former_idx);
            //cerr << "board idx " << board_idx << endl;
            if (board_idx == -1){
                board_idx = 0;
                search_c::init_parent(board, board_c.n_stones);
            }
            //cerr << "start searching" << endl;
            for (i = 0; i < evaluate_count; ++i){
                search_c::evaluate(board_idx, false, board_c.n_stones, 3, player);
            }
            int policy = -1;
            int mx = -inf;
            for (const int& cell : board_c.vacant_lst){
                if (search_c::nodes[board_idx].children[cell] != -1){
                    if (mx < search_c::nodes[search_c::nodes[board_idx].children[cell]].n_w + search_c::nodes[search_c::nodes[board_idx].children[cell]].n_v){
                        mx = search_c::nodes[search_c::nodes[board_idx].children[cell]].n_w + search_c::nodes[search_c::nodes[board_idx].children[cell]].n_v;
                        policy = cell;
                    }
                }
            }
            cout << board_c.coord_str(policy) << endl;
            return search_c::nodes[board_idx].children[policy];
        }

        inline int book_mcts(const int *board, int former_idx, int player){
            int i, cell;
            int board_idx = search_c::get_parent_idx(board, board_c.n_stones, former_idx);
            //cerr << "board idx " << board_idx << endl;
            if (board_idx == -1){
                board_idx = 0;
                search_c::init_parent(board, board_c.n_stones);
            }
            for (i = 0; i < evaluate_count; ++i){
                search_c::evaluate(board_idx, false, board_c.n_stones, 3, player);
            }
            //cerr << "ADDITIONAL SEARCH " << search_c::used_idx << endl;
            return board_idx;
        }

        inline void complete(const int *board){
            int lose_policy = -1, draw_policy = -1;
            int n_board[b_idx_num];
            double g;
            for (const int& cell : board_c.vacant_lst){
                if (board_c.check_legal(board, cell)){
                    board_c.move(board, n_board, cell);
                    g = -search_c::nega_alpha(n_board, -1.0, 1.0, 0);
                    if (g == 1.0){
                        cout << board_c.coord_str(cell) << endl;
                        return;
                    } else if (g == 0.0)
                        draw_policy = cell;
                    lose_policy = cell;
                }
            }
            if (draw_policy != -1){
                cout << board_c.coord_str(draw_policy) << endl;
                return;
            }
            cout << board_c.coord_str(lose_policy) << endl;
        }
};

search_c search_c;

int main(){
    int i, j, vacant_cnt, mcts_idx, policy;
    char elem;
    int board[b_idx_num], n_board[b_idx_num];
    pair<unsigned long long, unsigned long long> key;
    long long first_tl;
    cin >> board_c.ai_player;
    board_c.init();
    predict_c.init();
    //book_c.init();
    search_c.init();
    //cerr << "initialized" << endl;
    search_c.first_search();
    //cerr << "searched" << endl;
    board_c.direction = 0;
    if (board_c.ai_player == 0){
        string raw_board;
        cin >> raw_board; cin.ignore();
        policy = 37;
        //cerr << "FIRST direction " << board_c.direction << endl; 
        //cerr << "book policy " << policy << endl;
        cout << board_c.coord_str(policy) << endl;
    }
    mcts_idx = 0;
    while (true){
        vacant_cnt = board_c.input_board(board);
        /*
        for (i = 0; i < b_idx_num; ++i)
            //cerr << board[i] << ", ";
        //cerr << endl;
        board_c.print_board(board);
        predictions tmp = predict_c.predict(board);
        double mx = -1000.0;
        int mx_idx = -1;
        for (i = 0; i < hw2; ++i){
            //cerr << tmp.policies[i] << " ";
            if (mx < tmp.policies[i]){
                mx = tmp.policies[i];
                mx_idx = i;
            }
        }
        //cerr << endl;
        //cerr << mx_idx << " " << mx << " " << tmp.value << endl;
        return 0;
        */
        if (vacant_cnt >= hw2 - 4 - 5){
            vector<int> legal;
            for (i = 0; i < hw2; ++i){
                if (board_c.check_legal(board, i))
                    legal.push_back(i);
            }
            int idx = (int)(myrandom() * legal.size());
            cout << board_c.coord_str(legal[idx]) << endl;
            continue;
        }
        if (vacant_cnt > comp_stones)
            mcts_idx = search_c.mcts(board, mcts_idx, board_c.ai_player);
        else
            search_c.complete(board);
    }
    return 0;
}