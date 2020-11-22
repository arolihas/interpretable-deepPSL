import hitrate_calculator
from hitrate_calculator import parse_subseqs

PATH_DIR = '../lstm_attn/lstm_linear_model/subsequences/'

# lstm2_attn = 'above_top2std_subsequences_testData_avgLens5.csv'
# lstm2_rand = 'random_above_top2std_subsequences_testData_avgLens5.csv'
# lstm2_perm = 'permute_above_top2std_subsequences_testData_avgLens5.csv'
lstm1_attn = 'above_top1std_subsequences_testData_avgLens3.csv'
lstm1_rand ='random_above_top1std_subsequences_testData_avgLens3.csv'
lstm1_perm ='permute_above_top1std_subsequences_testData_avgLens3.csv'

# print('model ', parse_subseqs(PATH_DIR, lstm1_attn))
# print('permuted ', parse_subseqs(PATH_DIR, lstm1_perm, use_k_set=True))
print('random ', parse_subseqs(PATH_DIR, lstm1_rand))
