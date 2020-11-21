import captum
import torch
import utils
import model.net as net
from model.data_loader import DataLoader
from torch import device as dev
from captum.attr import LayerIntegratedGradients, LayerFeatureAblation, Saliency, TokenReferenceBase, visualization

MODEL_DIR = 'experiments/base_model/'
DATA_DIR = 'data/'
params = utils.Params(MODEL_DIR+'params.json')
params.vocab_size = 25
params.number_of_classes = 10
params.cuda = torch.cuda.is_available()

weights = MODEL_DIR + 'best.pth'
classes = ['Extracellular', 'Plastid', 'Cytoplasm', 'Mitochondrion', 
'Nucleus', 'ER', 'Golgi', 'Membrane', 'Lysosome', 'Peroxisome']

model = net.Net(params).cuda() if params.cuda else net.Net(params)
checkpoint = torch.load(weights, map_location=dev('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

loader = DataLoader(DATA_DIR, params)
data = loader.load_data(['train', 'val'], DATA_DIR)
train_data = data['train']
train_data_iterator = loader.data_iterator(train_data, params, shuffle=False)
train_batch, label_batch = next(train_data_iterator)
sentences_file = DATA_DIR + 'train/sentences.txt'
labels_file = DATA_DIR + 'train/labels.txt'
sentences = []
with open(sentences_file) as f:
    for sent in f.read().splitlines():
        sentences.append(sent)
labels = []
with open(labels_file) as f:
    for lab in f.read().splitlines():
        labels.append(lab)
token_reference = TokenReferenceBase(reference_token_idx=loader.pad_ind)

layer_ig = LayerIntegratedGradients(model, model.embedding)
vis_data_records = []

def interpret_sequence(model, sentences, data, attribution, records):
    model.zero_grad()
    for i, sentence in enumerate(sentences):
        seq_len = len(data[i])
        inp = data[i].unsqueeze(0)
        reference_indices = token_reference.generate_reference(seq_len, device=dev('cpu')).unsqueeze(0)
        pred = torch.sigmoid(model(inp))
        prob = pred.max().item()
        pred_ind = round(pred.argmax().item()) 
        if type(attribution) == LayerIntegratedGradients:
            attributions, delta = attribution.attribute(inp, reference_indices, n_steps=500, return_convergence_delta=True, target=label_batch[i])
        elif type(attribution) == Saliency:
            attributions = attribution.attribute(inp.long(), label_batch[i].long())
            delta = -1
        print('pred: ', classes[pred_ind], '(', '%.2f'%prob ,')', ', delta: ', abs(delta.numpy()[0]))
        add_attr_viz(attributions, sentence, prob, pred_ind, label_batch[i], delta, records)

def add_attr_viz(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions /= torch.norm(attributions) 
    attributions = attributions.cpu().detach().numpy()
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        classes[pred_ind],
        classes[label],
        'location',
        attributions.sum(),
        text,
        delta))

# interpret_sequence(model, sentences[:5], train_batch, vis_data_records)
# visualization.visualize_text(vis_data_records)
def ablate(data, target, ablator):
    inp = data.unsqueeze(0)
    attributions = ablator.attribute(inp.long(), target=target.long())
    return attributions