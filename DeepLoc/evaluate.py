"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
from model.data_loader import DataLoader
import pandas as pd
import numpy as np
import captum
import torch
import utils
from model.data_loader import DataLoader
from torch import device as dev
from captum.attr import LayerIntegratedGradients, LayerGradientShap, LayerFeatureAblation, TokenReferenceBase, visualization
from captum_viz import interpret_sequence
from captum_viz import *
# import model.haard_net as net
# import model.attn_lstm as net

net = False
net_name = ""

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--captum_seq', default=False, help="to extract subsequences using gradient captum grad vis", action="store_true")
parser.add_argument('--attn', default=False, help="use attention version of base model", action='store_true')
parser.add_argument('--seqmethod', default='intgrad', help="could be 'intgrad' or 'saliency'")
parser.add_argument('--net', default='model.attn_lstm', help="set whichever module you're using")
# parser.add_argument

attention_model = False
seqmethod = 'intgrad'

def _subseq(sequence, weights, top_std, predicted_label, true_label, before_after, sampleIdx):
    correct = predicted_label == true_label
    # print("predicted", predicted_label, "| true", true_label, "|", correct)

    # print(weights.shape)
    # print(weights[:10])
    mean, std = weights.mean(), weights.std()
    indices = np.where(weights > (mean + top_std*std))[0]

    allseqs = []
    cols = ['left', 'right', 'subsequence', 'predicted', 'true', 'classification', 'inputSequence']
    for idx in indices:
        left = idx - before_after
        if left < 0: left = 0
        right = idx + before_after + 1
        if right > len(sequence): right = len(sequence)
        
        subseq = ''.join(sequence[left:right])
        allseqs.append([left, right, subseq, predicted_label, true_label, correct, sampleIdx])
    # print(pd.DataFrame(allseqs, columns=cols))
    newSeqs = []
    i = 0
    while i < len(allseqs):
        row = allseqs[i]
        newSeqRow = row
        currleft = row[0]
        currright = row[1]
        j = i + 1
        if j < len(allseqs):
            nextLeft = allseqs[j][0]
        else:
            newSeqs.append(row)
            break
        while nextLeft < currright:
            currright = allseqs[j][1]
            j += 1
            if j >= len(allseqs): break
            nextLeft = allseqs[j][0]
        newRight = allseqs[j-1][1]
        newSeqRow[1] = newRight
        newSeqRow[2] = ''.join(sequence[currleft : newRight])
        newSeqs.append(newSeqRow)
        i = j

    allseqs = pd.DataFrame(newSeqs, columns=cols)
    # print(pd.DataFrame(allseqs, columns=cols))
    return allseqs

def captum_subseq(model, data_loader, data_iterator, metrics, params, num_steps, before_after=2, top_std=2):
    """top_std : top 2:25 % sequences (weights > mean + top_std*std)
    """
    mod = "Base" if not attention_model else "Attn"
    mod = net_name + "-" + mod
    fname = f'above_top{top_std}std_subseqs_testData_beforeAfter{before_after}_{mod}Model_{seqmethod}Method.csv'
    print(fname)

    model.eval()

    vis_data_records = [] # passed in a reference

    try:
        if seqmethod == 'intgrad':
            layer_ig = LayerIntegratedGradients(model, model.embedding)
            interpret_sequence_copy(model, data_loader, data_iterator, layer_ig, vis_data_records, num_steps, verbose=False)
        else:
            layer_sal = Saliency(model)
            interpret_sequence_copy(model, data_loader, data_iterator, layer_sal, vis_data_records, num_steps, verbose=True)
    except Exception as e:
        print(e)

    outseq = pd.DataFrame()
    before_after = 2
    print("Extracting subsequences")
    for i, vd in enumerate(vis_data_records):
        sequence = vd.raw_input
        weights = vd.word_attributions
        predicted_label = vd.pred_class
        true_label = vd.true_class
        sampleIdx = i
        out = _subseq(sequence, weights, top_std, predicted_label, true_label, before_after, sampleIdx)
        # print(out)
        outseq = outseq.append(out)
        # if i > 
    seqdir = 'subsequences'
    if not os.path.exists(seqdir): os.makedirs(seqdir)
    
    fname = os.path.join(seqdir, fname)
    outseq.to_csv(fname, index=False)
    print("Saved to", fname)



def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()#loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

import importlib

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    if args.captum_seq: params.batch_size = 1
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    net = importlib.import_module(args.net)
    net_name = args.net
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    seqmethod = args.seqmethod
    if args.attn:
        params.attn = 1
        attention_model = True
    if args.captum_seq:
        captum_subseq(model, data_loader, test_data_iterator, metrics, params, num_steps)
    else:
        test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
        save_path = os.path.join(args.model_dir, "metrics_test_{}_attn{}.json".format(args.restore_file, args.attn))
        utils.save_dict_to_json(test_metrics, save_path)
