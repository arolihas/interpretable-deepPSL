import hitrate_calculator
from hitrate_calculator import parse_subseqs

PATH_DIR = '../DeepLoc/subsequences/'

# uni_lstm_attn = 'uniform_above_top2std_subsequences_testData_avgLens5.csv'
# uni_lstm_rand = 'random_uniform_above_top2std_subsequences_testData_avgLens5.csv'
# uni_lstm_perm = 'permute_uniform_above_top2std_subsequences_testData_avgLens5.csv'

# attn_integrad = 'above_top2std_subseqs_testData_beforeAfter2_AttnModel_intgradMethod2.csv'
# base_integrad = 'above_top2std_subseqs_testData_beforeAfter2_BaseModel_intgradMethod2.csv'

# out1 = parse_subseqs(PATH_DIR, attn_integrad)
# print('attn_integrad ', out1)  # (0.5013162209561454), (0.4798704375948095)
# out2 = parse_subseqs(PATH_DIR, base_integrad, use_k_set=False)
# print('base_integrad ', out2)  # (error), (0.4633674345162402)


# cnn_lstm35711 = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-BaseModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711)
# print("cnn_lstm35711", out1)

# cnn_lstm35711_attn = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-AttnModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_attn)
# print("cnn_lstm35711_attn", out1)

# cnn_lstm = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm-BaseModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm)
# print('cnn_lstm', out)

# cnn_lstm_attn = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm_attn-AttnModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_attn)
# print('cnn_lstm_attn', out)

# cnn_lstm35711_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-BaseModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_inc)
# print("cnn_lstm35711_inc", out1)

# cnn_lstm35711_attn_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-AttnModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_attn_inc)
# print("cnn_lstm35711_attn_inc", out1)

# cnn_lstm_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm-BaseModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_inc)
# print('cnn_lstm_inc', out)

# cnn_lstm_attn_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm_attn-AttnModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_attn_inc)
# print('cnn_lstm_attn_inc', out)

PATH_DIR = '../lstm_attn/lstm_linear_model/subsequences/'
# cnn_att = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att)
# print("cnn_att", out)

# cnn_att_neg = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att_neg, use_incorrect=True)
# print("cnn_att", out)

# two.loc[two.predicted == 'ER', 'predicted'] = 'Endoplasmic.reticulum'
# two.loc[two.true == 'ER', 'true'] = 'Endoplasmic.reticulum'
# two.loc[two.predicted == 'Golgi', 'predicted'] = 'Golgi.apparatus'
# two.loc[two.true == 'Golgi', 'true'] = 'Golgi.apparatus'
# two.loc[two.predicted == 'Membrane', 'predicted'] = 'Cell.membrane'
# two.loc[two.true == 'Membrane', 'true'] = 'Cell.membrane'
# two.loc[two.predicted == 'Lysosome', 'predicted'] = 'Lysosome/Vacuole'
# two.loc[two.true == 'Lysosome', 'true'] = 'Lysosome/Vacuole'
# print('random ', parse_subseqs(PATH_DIR, uni_lstm_rand))
