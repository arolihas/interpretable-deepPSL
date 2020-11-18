import torch
x = torch.load('w_layer.pt')
torch.save(x.state_dict(),'w_layer_state.pt')
y = torch.load('u_layer.pt')
torch.save(y.state_dict(),'u_layer_state.pt')
print('save state successful')
torch.save(x.state_dict()['weight'],'w_omega.pt')
torch.save(y.state_dict()['weight'],'u_omega.pt')
print('save omega successful')
