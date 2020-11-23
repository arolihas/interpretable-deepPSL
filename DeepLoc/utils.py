import json
import logging
import os
import shutil

import torch
import numpy as np
import pandas as pd


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))


def load_checkpoint(checkpoint, model, optimizer=None):

    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading:", checkpoint)
    checkpoint = torch.load(checkpoint, map_location=device)
    print("Model summary")
    print(model)
    # print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


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

def get_subsequences(model, sequence, label, data_loader, sampleIdx, before_after=2, random=False, permute=False):
    output, attn_weights = model(sequence, random=random, permute=permute)

    classes = list(data_loader.tag_map.keys())
    true_label = label.tolist()[0]
    true_label = classes[true_label]

    sequence = [data_loader.idx_to_vocab[x] for x in sequence.squeeze().tolist()]
    attn_weights = np.array(attn_weights.squeeze().tolist())

    sm = torch.softmax(output.detach(), dim=1).flatten().cpu()
    predicted_label = classes[sm.argmax().item()]

    allseqs = _subseq(sequence, attn_weights, 2, predicted_label, true_label, before_after, sampleIdx)

    return allseqs

