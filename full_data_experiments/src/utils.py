import json
import torch
import os
import torch_geometric
import pytorch_lightning as pl

def load_data(name):
    jfile = open(name, "r")
    dicti = json.load(jfile)
    return dicti
def save_data(dicti,name):
      jfile = open(name, "w")
      jfile = json.dump(dicti, jfile)   
    
def save_model(checkpoint, path):

  torch.save(checkpoint, path)

def load_model(path, device):

  model = torch.load(path, map_location=device)
  
  return model

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def set_seed(seed_value):
    # Set seed for NumPy
    # np.random.seed(seed_value)

    # # Set seed for Python's random module
    # random.seed(seed_value)

    # # Set seed for PyTorch
    # torch.manual_seed(seed_value)

    # # Set seed for GPU (if available)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)

    #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # Set seed for PyTorch Geometric
    torch_geometric.seed_everything(seed_value)

    # Set seed for PyTorch Lightning
    pl.seed_everything(seed_value)
    print(f"{seed_value} have been correctly set!")
    # if torch.cuda.is_available():
    # #     # torch.cuda.manual_seed(seed_value)
    # #     # torch.cuda.manual_seed_all(seed_value)

    # #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)




