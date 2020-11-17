"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--visualize', default=False, help="vis ?", action="store_true")

def visualize(model, data_loader, data_iterator, metrics, params, num_steps):
    """
    CONTINUE NOTE:
        https://github.com/sharkmir1/Hierarchical-Attention-Network/blob/master/utils.py
        Implement attention using code from above
        TODO: get self.vocab and self.tag_map to view it properly 
        but just visualize it on indices for now. 
        1. call visualize from eval 
        2. return attn_weights from model 
        3. implement vis using above code (test.py and utils.py)
    """
    # call utils.visualize to visualize
    model.eval()

    for _ in range(num_steps):
        data_batch, labels_batch = next(data_iterator)
        print(data_batch.shape)
        print(labels_batch.shape)
        # print(data_batch[0])
        utils.visualize(model, data_batch, labels_batch, data_loader)
        # exit()
        _ = input("ENTER to continue")

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
    for _ in tqdm(range(num_steps)):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)
        
        # compute model output
        output_batch, _ = model(data_batch)
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
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    if args.visualize: params.batch_size = 1
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth'), model)

    num_steps = (params.test_size + 1) // params.batch_size
    if args.visualize:
        visualize(model, data_loader, test_data_iterator, metrics, params, num_steps)
    else:
        # Evaluate
        test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)
