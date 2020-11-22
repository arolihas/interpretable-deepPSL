import webbrowser
import numpy as np
import time
import pandas as pd
import json
import logging
import os
import shutil

import torch

# for coloring
import matplotlib
import matplotlib.pyplot as plt

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
    print("loading", checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Model Summary")
    print(model)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint

def map_sentence_to_color(sequence, attn_weights):
    """untested with GPU, might need to move sentence and weights to cpu"""
    wordmap = matplotlib.cm.get_cmap('OrRd')
    # print(wordmap(attn_weights[0]))
    # print(sum(attn_weights))
    # print(max(attn_weights))
    # print(attn_weights[:5])
    # exit()
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    result = ''
    for word, score in zip(sequence, attn_weights):
        color = matplotlib.colors.rgb2hex(wordmap(score)[:3])
        result += template.format(color, '&nbsp' + word + '&nbsp') + ' '
    return result


def bar_chart(categories, scores, graph_title="Prediction", output_name="prediction_bar_chart.png"):
    y_pos = range(len(categories))

    fig, ax = plt.subplots()
    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, categories)
    plt.ylabel('Softmax Score')
    plt.title(graph_title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30,
             horizontalalignment='right', fontsize='x-small')
    plt.tight_layout()

    plt.savefig(output_name)

    plt.clf()
    plt.close()


# import imgkit

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

# def get_subsequence_nonattn():


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

def getname(random, permute, result, sample_idx, dir=""):
    if not random and not permute:
        vistype = "_regular"
    else:
        vistype = ""
        if random:
            vistype += "_random"
        if permute:
            vistype += "_permute"
    # prefix = time.strftime("%Y-%m-%d-%H.%M.%S_")+str(result)+vistype
    prefix = f"{sample_idx}_{result}{vistype}"
    html_path = prefix + ".html"
    pred_file = pred_path = os.path.join('pred', f'{prefix}_pred.png')
    if dir != "":
        if not os.path.exists(dir): os.makedirs(dir)
        if not os.path.exists(os.path.join(dir, 'pred')): os.makedirs(os.path.join(dir, 'pred'))
        html_path = os.path.join(dir, html_path)
        pred_path = os.path.join(dir, pred_file)
    return prefix, html_path, pred_path, pred_file

# div: https://www.w3schools.com/css/css_image_gallery.asp
html_str = """<html>
        <head>
        <style>
        div.gallery {
        margin: 5px;
        border: 1px solid #ccc;
        float: left;
        width: 30%;
        }

        div.gallery:hover {
        border: 1px solid #777;
        }

        div.gallery img {
        width: 100%;
        height: auto;
        }

        div.desc {
        padding: 15px;
        text-align: center;
        }
        </style>
        </head>
        <body>
        """

html_end = """</body>
            </html>
            """

def getImgDiv(imgFile, desc):
    html_div = """
    <div class="gallery">
    <a target="_blank" href="{}">
        <img src="{}" alt="Cinque Terre" width="800" height="800">
    </a>
    <div class="desc">{}</div>
    </div>
    """.format(imgFile, imgFile, desc)
    # print(html_div)
    # exit()
    return html_div

def runModel(model, sequence, label, data_loader, scale_attn=100, random=False, permute=False):
    """
    label: true label
    """
    output, attn_weights = model(sequence, random=random, permute=permute)
    classes = list(data_loader.tag_map.keys())
    true_label = label.tolist()[0]
    true_label = classes[true_label]
    sequence = [data_loader.idx_to_vocab[x] for x in sequence.squeeze().tolist()]
    attn_weights = attn_weights.squeeze()
    attn_weights = (attn_weights * scale_attn).tolist()
    sm = torch.softmax(output.detach(), dim=1).flatten().cpu()
    predicted_label = classes[sm.argmax().item()]

    return output, attn_weights, predicted_label, true_label, classes, sm, sequence


def visualize(model, sequence, label, data_loader, sample_idx, view_browser=True, random=False, permute=False):
    """
    expects sequence to be batch size 1
    """
    # print("Visualizing...")
    assert sequence.shape[0] == 1 and label.shape[0] == 1, "visualizing sequence should be batch size 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output, attn_weights = model(sequence, random=random)

    # classes = list(data_loader.tag_map.keys())
    # true_label = label.tolist()[0]
    # true_label = classes[true_label]

    # sequence = [data_loader.idx_to_vocab[x] for x in sequence.squeeze().tolist()]
    # attn_weights = attn_weights.squeeze().tolist()
    # attn_weights = np.random.rand(len(sequence))
    result = html_str
    result += "<h2>Attention Visualization</h2>"
    # sm = torch.softmax(output.detach(), dim=1).flatten().cpu()
    # print(sm.argmax().item())
    # predicted_label = classes[sm.argmax().item()]
    random_params = [False, True, False]
    permute_params = [False, False, True]
    run_types = ['Regular Attention', 'Random Attention', 'Permuted Attention']
    for random, permute, runType in zip(random_params, permute_params, run_types):
        output, attn_weights, predicted_label, true_label, classes, sm, seq = runModel(model, sequence, label, data_loader, random=random, permute=permute)
        prefix, fname, pred_path, predfile = getname(random, permute, predicted_label == true_label, sample_idx, dir="vis")

        bar_chart(classes, sm, f'{runType} Prediction', output_name=pred_path)
        # result += f'<br><img src="{predfile}.png"><br>'
        # desc = f'<br>{runType}<br>'
        desc = f'<br>Prediction = <b>{predicted_label}</b> | True label = <b>{true_label}</b><br><br>'
        desc += map_sentence_to_color(seq, attn_weights)
        result += getImgDiv(predfile, desc)

    result += html_end

    with open(fname, 'w') as f:
        f.write(result)
    
    # print("Saved html to", fname)
    fname = 'file://'+os.getcwd()+'/'+fname

    if view_browser:
        print("Opening", fname)
        webbrowser.open_new(fname)

    



