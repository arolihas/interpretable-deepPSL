import hitrate_calculator
from hitrate_calculator import parse_subseqs
import os
from time import time
from datetime import timedelta

results_file = 'parse_results.txt'

def outp(string, file=results_file):
    print(string)
    with open(file, 'a') as f:
        f.write(string + "\n")

def parse(PATH_DIR, fname, use_k_set=False, use_incorrect=False, use_true_class=False, special_label='', period=100):
    start = time()
    # print("Processing:", os.path.join(PATH_DIR, fname))
#     outfile = f'{special_label}.txt'
    outfile = results_file
    outp(f'-------------------------- {special_label} ----------------------------', file=outfile)
    outp(f"use_k_set: {use_k_set} | use_incorrect: {use_incorrect} | use_true_class: {use_true_class} | special_label: {special_label} | period: {period}", file=outfile)
    out = parse_subseqs(PATH_DIR, fname, use_k_set=use_k_set, use_incorrect=use_incorrect, use_true_class=use_true_class, period=period)
    outp(f"{special_label} | Hitrate: {out}", file=outfile)
    elapsed = str(timedelta(seconds=time() - start))
    outp(f"Elapsed time: {elapsed}", file=outfile)
    
def parseThree(PATH_DIR, fname, special_label='', period=100):
    parse(PATH_DIR, fname, special_label=special_label, period=period)
    parse(PATH_DIR, fname, use_incorrect=True, special_label=f'{special_label} false, pred', period=period, use_k_set=False)
    parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label=f'{special_label} false, true', period=period, use_k_set=False)
    

PATH_DIR = '../DeepLoc/subsequences/'

# RUNNING (Server)
attn_integrad = 'above_top2std_subseqs_testData_beforeAfter2_AttnModel_intgradMethod2.csv'
# parseThree(PATH_DIR, attn_integrad, special_label='LSTM_ATTN-IG', period=500)
# parse(PATH_DIR, attn_integrad, special_label='LSTM_ATTN-IntGrad')
# parse(PATH_DIR, attn_integrad, use_incorrect=True, special_label='LSTM_ATTN-IntGrad false, pred')
# parse(PATH_DIR, attn_integrad, use_incorrect=True, use_true_class=True, special_label='LSTM_ATTN-IntGrad, false, true')


# RUNNING (Server)
base_integrad = 'above_top2std_subseqs_testData_beforeAfter2_BaseModel_intgradMethod2.csv'
# parseThree(PATH_DIR, base_integrad, special_label='LSTM_BASE', period=500)
# parse(PATH_DIR, base_integrad, special_label='LSTM_BASE')
# parse(PATH_DIR, base_integrad, use_incorrect=True, special_label='LSTM_BASE false, pred')
# parse(PATH_DIR, base_integrad, use_incorrect=True, use_true_class=True, special_label='LSTM_BASE false, true')

# RUNNING (Server)
fname = 'above_top2std_subseqs_testData_beforeAfter2_Model-model.cnn_lstm35711_regularAttn.csv'
# parseThree(PATH_DIR, fname, special_label='CNN_LSTM_ATTNreg', period=500)
# parse(PATH_DIR, fname, special_label='CNN_LSTM_ATTNreg')
# parse(PATH_DIR, fname, use_incorrect=True, special_label='CNN_LSTM_ATTNreg false, pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTNreg false, true')

# RUNNING (Server)
fname = 'above_top2std_subseqs_testData_beforeAfter2_Model-model.cnn_lstm35711_permuteAttn.csv'
# parseThree(PATH_DIR, fname, special_label='CNN_LSTM_ATTNpermute', period=500)
# parse(PATH_DIR, fname, special_label='CNN_LSTM_ATTNpermute')
# parse(PATH_DIR, fname, use_incorrect=True, special_label='CNN_LSTM_ATTNpermute false, pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTNpermute false, true')

# RUNNING (Server)
fname = 'above_top2std_subseqs_testData_beforeAfter2_Model-model.cnn_lstm35711_randomAttn.csv'
# parseThree(PATH_DIR, fname, special_label='CNN_LSTM_ATTNrandom', period=500)
# parse(PATH_DIR, fname, special_label='CNN_LSTM_ATTNrandom')
# parse(PATH_DIR, fname, use_incorrect=True, special_label='CNN_LSTM_ATTNrandom false, pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTNrandom false, true')
# exit()

# RUNNING (Server)
lstm_uniform = 'above_top2std_subseqs_testData_beforeAfter2_model.net_uniform-BaseModel_intgradMethod.csv'
# parseThree(PATH_DIR, lstm_uniform, special_label='LSTM-uniform-ATTN', period=500)
# parse(PATH_DIR, lstm_uniform, special_label='LSTM-uniform-ATTN')
# parse(PATH_DIR, lstm_uniform, use_incorrect=True, use_true_class=False, special_label='LSTM-uniform-ATTN false, pred')
# parse(PATH_DIR, lstm_uniform, use_incorrect=True, use_true_class=True, special_label='LSTM-uniform-ATTN false, true')

# RUNNING (Server)
cnn_lstm35711 = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-BaseModel_intgradMethod.csv'
# parseThree(PATH_DIR, cnn_lstm35711, special_label='CNN_LSTM_BASE-IG', period=500)
# parse(PATH_DIR, cnn_lstm35711, special_label='CNN_LSTM_BASE-IG')
# parse(PATH_DIR, cnn_lstm35711, use_incorrect=True, special_label='CNN_LSTM_BASE-IG false, pred')
# parse(PATH_DIR, cnn_lstm35711, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_BASE-IG false, true')

# RUNNING ()
cnn_lstm35711_attn = 'above_top2std_subseqs_testData_beforeAfter2_model.cnn_lstm35711-AttnModel_intgradMethod.csv'
# parseThree(PATH_DIR, cnn_lstm35711_attn, special_label='CNN_LSTM_ATTN-IG', period=500)
# out1 = parse_subseqs(PATH_DIR, cnn_lstm35711_attn)
# print("cnn_lstm35711_attn", out1)
# parse(PATH_DIR, cnn_lstm35711_attn, special_label='CNN_LSTM_ATTN-IG') # not rerunning (averaged 4!)
# parse(PATH_DIR, cnn_lstm35711_attn, use_incorrect=True, special_label='CNN_LSTM_ATTN-IG false, pred')
# parse(PATH_DIR, cnn_lstm35711_attn, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTN-IG false, true')

PATH_DIR = '../lstm_attn/lstm_linear_model/subsequences/'

# RUNNING ()
cnn_att = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# parseThree(PATH_DIR, cnn_att, special_label='CNN_LSTM_ATTN', period=500)
# out = parse_subseqs(PATH_DIR, cnn_att, period=89)
# print("cnn_att", out)
# parse(PATH_DIR, cnn_att, special_label='CNN_LSTM_ATTN')
# parse(PATH_DIR, cnn_att, use_incorrect=True, special_label='CNN_LSTM_ATTN false, pred')
# parse(PATH_DIR, cnn_att, use_incorrect=True, use_true_class=True, special_label='CNN_LSTM_ATTN false, true')

# cnn_att_neg = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att_neg, use_incorrect=True)
# print("cnn_att_neg", out)

# RUNNING ()
fname = 'above_top2std_subsequences_testData_avgLens5.csv'
# parseThree(PATH_DIR, fname, special_label='lstm_attention', period=500)
# parse(PATH_DIR, fname, special_label='lstm_attention')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention false,true')

# RUNNING ()
fname = 'random_above_top2std_subsequences_testData_avgLens5.csv'
# parseThree(PATH_DIR, fname, special_label='lstm_attention_random', period=500)
# parse(PATH_DIR, fname, special_label='lstm_attention_random')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention_random false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention_random false,true')

# RUNNING ()
fname = 'permute_above_top2std_subsequences_testData_avgLens5.csv'
# parseThree(PATH_DIR, fname, special_label='lstm_attention_permute', period=500)
# parse(PATH_DIR, fname, special_label='lstm_attention_permute')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=False, special_label='lstm_attention_permute false,pred')
# parse(PATH_DIR, fname, use_incorrect=True, use_true_class=True, special_label='lstm_attention_permute false,true')

# cnn_att_neg = 'cnn35711a_above_top2std_subsequences_testData_avgLens5.csv'
# out = parse_subseqs(PATH_DIR, cnn_att_neg, use_incorrect=True)
# print("cnn_att_neg true", out)


