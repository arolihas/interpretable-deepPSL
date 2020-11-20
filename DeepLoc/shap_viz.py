import shap
import torch
import utils
import model.net as net
from model.data_loader import DataLoader
from torch import device as dev

MODEL_DIR = 'experiments/base_model/'
DATA_DIR = 'data/'
params = utils.Params(MODEL_DIR+'params.json')
params.vocab_size = 25
params.number_of_classes = 10
params.cuda = torch.cuda.is_available()

weights = MODEL_DIR + 'best.pth'

model = net.Net(params).cuda() if params.cuda else net.Net(params)
checkpoint = torch.load(weights, map_location=dev('cpu'))
model.load_state_dict(checkpoint['state_dict'])

data_loader = DataLoader(DATA_DIR, params)
data = data_loader.load_data(['train', 'val'], DATA_DIR)
train_data = data['train']
train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
train_batch, _ = next(train_data_iterator)

val_data = data['val']
val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
val_batch, _ = next(val_data_iterator)

explainer = shap.DeepExplainer(model, train_batch[:100])
vals = train_batch[:10]

shap_values = explainer.shap_values(train_batch[:10])
shap.force_plot(explainer.expected_value[0], shap_values[0][0], train_batch[:10])
