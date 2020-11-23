import hitrate_calculator
from hitrate_calculator import parse_subseqs

PATH_DIR = '../lstm_attn/lstm_linear_model/'

# lstm2_attn = 'above_top2std_subsequences_testData_avgLens5.csv'
# lstm2_rand = 'random_above_top2std_subsequences_testData_avgLens5.csv'
# lstm2_perm = 'permute_above_top2std_subsequences_testData_avgLens5.csv'
# lstm1_attn = 'above_top1std_subsequences_testData_avgLens3.csv'
# lstm1_rand ='random_above_top1std_subsequences_testData_avgLens3.csv'
# lstm1_perm ='permute_above_top1std_subsequences_testData_avgLens3.csv'

# cnn_lstm_attn = 'above_top2std_subsequences_testData_avgLens5.csv'
# cnn_lstm_rand = 'random_above_top2std_subsequences_testData_avgLens5.csv'
# cnn_lstm_perm = 'permute_above_top2std_subsequences_testData_avgLens5.csv'

# cnn_lstm_attn = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# cnn_lstm_rand = 'random_cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# cnn_lstm_perm = 'permute_cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'

# print('model ', parse_subseqs(PATH_DIR, cnn_lstm_attn))
# print('permuted ', parse_subseqs(PATH_DIR, cnn_lstm_perm, use_k_set=True))
# print('random ', parse_subseqs(PATH_DIR, cnn_lstm_rand))


print("---MLP---")
uni_lstm_attn = 'preattn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_rand = 'random_preattn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_perm = 'permute_preattn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
print('model ', parse_subseqs(PATH_DIR, uni_lstm_attn))
print('permuted ', parse_subseqs(PATH_DIR, uni_lstm_perm, use_k_set=True))
print('random ', parse_subseqs(PATH_DIR, uni_lstm_rand))

print("---MLP Attn---")
uni_lstm_attn = 'attn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_rand = 'random_attn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_perm = 'permute_attn_mlp_above_top2std_subsequences_testData_avgLens5.csv'
print('model ', parse_subseqs(PATH_DIR, uni_lstm_attn))
print('permuted ', parse_subseqs(PATH_DIR, uni_lstm_perm, use_k_set=True))
print('random ', parse_subseqs(PATH_DIR, uni_lstm_rand))

print("---MLP Uniform---")
uni_lstm_attn = 'uniform_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_rand = 'random_uniform_mlp_above_top2std_subsequences_testData_avgLens5.csv'
uni_lstm_perm = 'permute_uniform_mlp_above_top2std_subsequences_testData_avgLens5.csv'
print('model ', parse_subseqs(PATH_DIR, uni_lstm_attn))
print('permuted ', parse_subseqs(PATH_DIR, uni_lstm_perm, use_k_set=True))
print('random ', parse_subseqs(PATH_DIR, uni_lstm_rand))