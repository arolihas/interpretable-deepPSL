import hitrate_calculator
from hitrate_calculator import parse_subseqs
import os
from time import time
from datetime import timedelta

def parse(PATH_DIR, fname, use_k_set=False, use_incorrect=False, use_true_class=False, special_label=''):
    start = time()
    # print("Processing:", os.path.join(PATH_DIR, fname))
    print('--------------------------', special_label, '----------------------------')
    print("use_k_set:", use_k_set, "| use_incorrect:", use_incorrect, "| use_true_class:", use_true_class, "| special_label:", special_label)
    out = parse_subseqs(PATH_DIR, fname, use_k_set=use_k_set, use_incorrect=use_incorrect, use_true_class=use_true_class)
    print(special_label, "| Hitrate:", out)
    elapsed = str(timedelta(seconds=time() - start))
    print("Elapsed time:", elapsed)

PATH_DIR = '../DeepLoc/subsequences/'

attn_integrad = 'above_top2std_subseqs_testData_beforeAfter2_AttnModel_intgradMethod2.csv'
# base_integrad = 'above_top2std_subseqs_testData_beforeAfter2_BaseModel_intgradMethod2.csv'

# out1 = parse_subseqs(PATH_DIR, attn_integrad)
# print('attn_integrad ', out1)  # (0.5013162209561454), (0.4798704375948095)
# parse(PATH_DIR, attn_integrad, special_label='LSTM_ATTN-IntGrad')
# parse(PATH_DIR, attn_integrad, use_incorrect=True, special_label='LSTM_ATTN-IntGrad false, pred')
# parse(PATH_DIR, attn_integrad, use_incorrect=True, use_true_class=True, special_label='LSTM_ATTN-IntGrad, false, true')

# out2 = parse_subseqs(PATH_DIR, base_integrad, use_k_set=False)
# print('base_integrad ', out2)  # (error), (0.4633674345162402)

# out = parse_subseqs(PATH_DIR, base_integrad, use_incorrect=True)
# print('base_lstm incorrect', out)
# out = parse_subseqs(PATH_DIR, base_integrad, use_incorrect=True, use_true_class=True)
# print('base_lstm incorrect trueclass', out)

fname = 'above_top2std_subseqs_testData_beforeAfter2_Model-model.cnn_lstm35711_regularAttn.csv'
# parse(PATH_DIR, fname, special_label='CNN_LSTM_ATTNreg')
# parse(PATH_DIR, fname, use_incorrect=True, special_label='CNN_LSTM_ATTNreg false, pred')
parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTNreg false, true')
exit()


lstm_uniform = 'above_top2std_subseqs_testData_beforeAfter2_model.net_uniform-BaseModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, lstm_uniform)
# print("lstm_uniform", out1)
# parse(PATH_DIR, lstm_uniform, special_label='LSTM-uniform-ATTN')
# parse(PATH_DIR, lstm_uniform, use_incorrect=True, use_true_class=False, special_label='LSTM-uniform-ATTN false, pred')
# parse(PATH_DIR, lstm_uniform, use_incorrect=True, use_true_class=True, special_label='LSTM-uniform-ATTN false, true')


# cnn_lstm35711 = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-BaseModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711)
# print("cnn_lstm35711", out1)

cnn_lstm35711_attn = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-AttnModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_attn)
# print("cnn_lstm35711_attn", out1)
# parse(PATH_DIR, cnn_lstm35711_attn, special_label='CNN_LSTM_ATTN-IG')
# parse(PATH_DIR, cnn_lstm35711_attn, use_incorrect=True, special_label='CNN_LSTM_ATTN-IG false, pred')
# parse(PATH_DIR, cnn_lstm35711_attn, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTN-IG false, true')

# cnn_lstm = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm-BaseModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm)
# print('cnn_lstm', out)

# cnn_lstm_attn = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm_attn-AttnModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_attn)
# print('cnn_lstm_attn', out)

# cnn_lstm35711_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-BaseModel_intgradMethod.csv'
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_inc, use_incorrect=True, use_true_class=True)
# print("cnn_lstm35711_inc_true", out1)

# cnn_lstm_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm-BaseModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_inc)
# print('cnn_lstm_inc', out)

# cnn_lstm_attn_inc = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm_attn-AttnModel_intgradMethod.csv'
# out = parse_subseqs(PATH_DIR, cnn_lstm_attn_inc)
# print('cnn_lstm_attn_inc', out)

PATH_DIR = '../lstm_attn/lstm_linear_model/subsequences/'
cnn_att = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att, period=89)
# print("cnn_att", out)
# parse(PATH_DIR, cnn_att, special_label='CNN_LSTM_ATTN')
# parse(PATH_DIR, cnn_att, use_incorrect=True, special_label='CNN_LSTM_ATTN false, pred')
# parse(PATH_DIR, cnn_att, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTN false, true')

# cnn_att_neg = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att_neg, use_incorrect=True)
# print("cnn_att_neg", out)

fname = 'above_top2std_subsequences_testData_avgLens5.csv'
# parse(PATH_DIR, fname, special_label='lstm_attention')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention false,true')
fname = 'random_above_top2std_subsequences_testData_avgLens5.csv'
# parse(PATH_DIR, fname, special_label='lstm_attention_random')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention_random false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention_random false,true')
fname = 'permute_above_top2std_subsequences_testData_avgLens5.csv'
# parse(PATH_DIR, fname, special_label='lstm_attention_permute')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention_permute false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention_permute false,true')

# cnn_att_neg = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att_neg, use_incorrect=True)
# print("cnn_att_neg true", out)


