
import subseq_utils as su
import pandas as pd
import numpy as np
from tqdm import tqdm

ks = {}
def parse_subseqs(filepath, filename, use_k_set=False):
    subs = pd.read_csv(filepath + filename)
    subtp = subs[subs.classification]
    input_seqs = np.unique(subtp.inputSequence)
    global ks
    period = 200 # max(subtp.inputSequence)
    hitrates = []
    global_vals = []
    if ks == {}:
        k = 0
    for i in tqdm(range(0,period)):
        if not use_k_set:
            while k in ks.values():
                k = np.random.randint(0, len(input_seqs))
            ks[i] = k
        else:
            k = ks[i]
        subseqs = subtp[subtp.inputSequence == input_seqs[k]]
        peptides = list(subseqs.subsequence)
        start_indexes = list(subseqs.left)
        end_indexes = list(subseqs.right)
        loc = list(subseqs.predicted)[0]
        indexMap = dict([(peptides[i], (start_indexes[i], end_indexes[i])) for i in range(len(subseqs))])
        peptides = ', '.join(peptides) # str | A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. (default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK)
        prots = su.peptide_search(peptides)
        hr, vdf = su.hitrate(prots, indexMap, loc)
        hitrates.append(hr)
        global_vals.append(vdf)
        if (i % 25) == 0:
            pd.concat(global_vals).to_csv(filepath+'vals_{}'.format(filename), index=False)
    df = pd.DataFrame(hitrates, columns=['hitrate'])
    df.to_csv(filepath + 'hitrate10_{}'.format(filename))
    global_val = pd.concat(global_vals)
    global_val.to_csv(filepath+'vals_{}'.format(filename), index=False)
    return np.mean(hitrates), np.sum(hitrates)

PATH_DIR = '../lstm_attn/lstm_linear_model/'
attn = 'above_top2std_subsequences_testData_avgLens5.csv'
rand = 'random_above_top2std_subsequences_testData_avgLens5.csv'
perm = 'permute_above_top2std_subsequences_testData_avgLens5.csv'

# print('model ', parse_subseqs(PATH_DIR, attn))
# print('permuted ', parse_subseqs(PATH_DIR, perm, use_k_set=True))
print('random ', parse_subseqs(PATH_DIR, rand))
