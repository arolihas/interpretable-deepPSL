
import subseq_utils as su
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import datetime

ks = {}

# def print_elapsed_time(prefix=''):
#     e_time = time.time()
#     if not hasattr(print_elapsed_time, 's_time'):
#         print_elapsed_time.s_time = e_time
#     else:
#         print(f'{prefix} elapsed time: {e_time - print_elapsed_time.s_time:.2f} sec')
#         print_elapsed_time.s_time = e_time

start = None
def custom_estimate(itr, total):
    global start
    if start is None or itr == 0:
        print("Initializing custom tqdm")
        start = time()
        return
    elapsed = time() - start
    frac_left = (total - itr)/itr
    remaining = str(datetime.timedelta(seconds=elapsed*frac_left))
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Step {itr}/{total} | elapsed: {elapsed} | estimate_remaining: {remaining}")

def parse_subseqs(filepath, filename, use_k_set=False, use_incorrect=False, period=100):
    print("Parsing:", filepath + filename)
    subs = pd.read_csv(filepath + filename)
    if use_incorrect:
        subtp = subs[~subs.classification]
    else:
        subtp = subs[subs.classification]
    input_seqs = np.unique(subtp.inputSequence)
    global ks
    period = period # max(subtp.inputSequence)
    hitrates = []
    global_vals = []
    if not use_k_set:
        k = 0
    # print_elapsed_time()
    for i in range(0,period):
        # print_elapsed_time(f"Step {i}")
        custom_estimate(i, period)
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
