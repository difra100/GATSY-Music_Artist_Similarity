import os
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch.optim import lr_scheduler
import random
from random import choice,randrange
import matplotlib.pyplot as plt
import math
import time
from torch_geometric import seed_everything
from sklearn.manifold import TSNE
import pprint


from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
import wandb
import argparse

# Internal libraries 
from src.utils import *
from src.architectures import *
from src.hyperparameters import *
from src.datamodule import *
from src.train import * 

set_determinism_the_old_way(deterministic = True)

parser = argparse.ArgumentParser()


# train_mode : True and test_mode : False ---> Hyperparameter Search 
# train_mode : True and test_mode : True ---> Test Model on validation set 
# train_mode : False and test_mode : True ---> Test Model on test set


parser.add_argument("-train_mode", type = bool, default = False)
parser.add_argument("-test_mode", type = bool, default = False)
parser.add_argument("-model_name", type = str, default = '')
parser.add_argument("-wb", type = bool, default = False)
parser.add_argument("-resume_sweep", type = str, default = '')
parser.add_argument("-count", type = int, default = 220)

parser.add_argument("-save_model", type = bool, default = False)
parser.add_argument("-lr", type = float, default = 1e-3)
parser.add_argument("-wd", type = float, default = 0)
parser.add_argument("-n_layers", type = int, default = 2)
parser.add_argument("-n_heads", type = int, default = 2)
parser.add_argument("-num_epochs", type = int, default = 100)

parser.add_argument("-random_feat", type = bool, default = False)



args = parser.parse_args()


train_mode = args.train_mode
test_mode = args.test_mode
random_feat = args.random_feat
wb = args.wb
model_name = args.model_name
resume_sweep = args.resume_sweep
count = args.count
save_model = args.save_model
lr = args.lr
wd = args.wd
n_layers = args.n_layers
n_heads = args.n_heads
num_epochs = args.num_epochs

if model_name != 'GAT':
  n_heads = None

loss_mode = 'triplet'


config = {
        "lr" : lr,
        "wd" : wd,
        "n_layers" : n_layers,
        "model_name": model_name,
        "random_feat": random_feat,
    }

if n_heads != None:
  config['n_heads'] = n_heads



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device selected in this execution is: ", device)

if random_feat == True:
  print("RANDOM MODE HAS BEEN SELECTED..........")
  X = torch.load('./data/random_instance.pt', map_location=torch.device(device))

else:
  print("LOW-LEVEL Feat. MODE HAS BEEN SELECTED..........")
  X = torch.load('./data/instance.pt', map_location=torch.device(device))      # Instance matrix

A1 = torch.load('./data/adjacency.pt', map_location=torch.device(device)) 

A = torch.load('./data/adjacencyCOO.pt', map_location=torch.device(device))    # Adjacency matrix in the COO format, that is that supported by torch geometric


data_for_train_ = n_of_vtrain
data_for_train = n_of_train
fin_val = n_of_test 


''' This variable contains the indices for the splitting, that are necessary to compute the masks, according to the torch geometric pipeline '''
data_summary = {'train_with_val' : {'low' : 0, 'high': data_for_train_},
                'train' : {'low' : 0, 'high' : data_for_train},
                'val' : {'low' : data_for_train_, 'high' : data_for_train},
                'test' : {'low' : data_for_train, 'high' : fin_val}}



total_mask = torch.zeros(X.shape[0], dtype = torch.bool)


vtrain_mask = total_mask.clone()
train_mask = total_mask.clone()
val_mask = total_mask.clone()
test_mask = total_mask.clone()
eval_val = total_mask.clone()


vtrain_mask[data_summary['train_with_val']['low']:data_summary['train_with_val']['high']] = True
val_mask[data_summary['val']['low']:data_summary['val']['high']] = 1
train_mask[data_summary['train']['low']:data_summary['train']['high']] = 1
test_mask[data_summary['test']['low']:data_summary['test']['high']] = 1

eval_val[data_summary['train_with_val']['low']:data_summary['val']['high']] = 1

kwargs = {'vtrain_mask':vtrain_mask, 'train_mask':train_mask, 'val_mask':val_mask, 'test_mask':test_mask}



data = Data(x=X.to(device), edge_index = A.to(device))
data.vtrain_mask = vtrain_mask
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.eval_val = eval_val

if (train_mode == True and test_mode == False) or ((train_mode == True and test_mode == True)):
  trainmask = data.vtrain_mask
  testmask = data.val_mask
  mode = 1
  if ((train_mode == True and test_mode == True)): 
    mode = 3
  
elif train_mode == False and test_mode == True:
  trainmask = data.train_mask
  testmask = data.test_mask
  mode = 2

train_loader = NeighborLoader(data, input_nodes = trainmask, num_neighbors=[n_of_neigh]*n_layers, shuffle = True, batch_size = batch_size)
test_loader = NeighborLoader(data, input_nodes = testmask, num_neighbors=[n_of_neigh]*n_layers, shuffle = False, batch_size = batch_size)

mode_diz = {1: "You have enabled hyperparameter tuning modality.........",
            2: "You have enabled test set evaluation.........",
            3: "You have enabled val set evaluation...."}

print(mode_diz[mode])


def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, resume = True if resume_sweep != '' else False):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        if model_name == 'SAGE':
          print("GRAPHSAGE SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, model_name, None, 3, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        elif model_name == 'GAT':
          print("GATSY SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, model_name, config.n_heads, config.n_layers, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        elif model_name == 'GCN':
          print("GCN SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, model_name, None, config.n_layers, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        elif model_name == 'GIN':
          print("GIN SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, model_name, None, config.n_layers, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        
        
        accuracy_array = np.array(metrics[0])
        mean_acc, std_acc = np.mean(accuracy_array), np.std(accuracy_array)
        print("MEAN AND STANDARD DEVIATION: ", mean_acc, std_acc)
        wandb.log({"nDCG on test (Mean)": mean_acc, "nDCG on test (Std.)": std_acc})



if mode != 1:

  metrics = []
  seeds_list = [43, 1337, 7, 777, 9876, 54321, 123456, 999, 31415, 2022]
  for seed in seeds_list:
    metric = train_generic(seed, train_loader, test_loader, model_name, n_heads, n_layers, lr, wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, config = config, wb = wb, save_model = save_model, random_feat = random_feat)
    metrics.append(metric[0])

  accuracy_array = np.array(metrics)
  mean_acc, std_acc = np.mean(accuracy_array), np.std(accuracy_array)
  print("MEAN AND STANDARD DEVIATION: ", mean_acc, std_acc)
  wandb.log({"nDCG on test (Mean)": mean_acc, "nDCG on test (Std.)": std_acc})

else:
  if model_name == 'GAT':
    sweep_config['parameters'] = parameters_dict_GAT
  elif model_name == 'SAGE': 
    sweep_config['parameters'] = parameters_dict_SAGE
  elif model_name == 'GCN':
    sweep_config['parameters'] = parameters_dict_GCN
  elif model_name == 'GIN':
    sweep_config['parameters'] = parameters_dict_GIN


  if resume_sweep != '':
      sweep_id = resume_sweep
      print("RESUMED PAST SWEEP....")
  else:
      sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)
  pprint.pprint(sweep_config)

    

  wandb.agent(sweep_id, sweep_train, count = count, project = project_name, entity= entity_name)




if wb:
  wandb.finish()










