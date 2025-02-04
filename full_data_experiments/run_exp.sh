python main.py -test_mode True -model_name 'SAGE' -wb True -lr 0.0001 -n_layers 3 -wd 0.000001 -num_epochs 1000 -random_feat True -save_model True
python main.py -test_mode True -model_name 'GIN' -wb True -lr 0.0001 -n_layers 2 -wd 0 -num_epochs 1000 -save_model True
python main.py -test_mode True -model_name 'GIN' -wb True -lr 0.0001 -n_layers 2 -wd 0 -num_epochs 1000 -random_feat True -save_model True
python main.py -test_mode True -model_name 'GCN' -wb True -lr 0.0001 -n_layers 3 -wd 0 -num_epochs 1000 -save_model True
python main.py -test_mode True -model_name 'GCN' -wb True -lr 0.0001 -n_layers 3 -wd 0 -num_epochs 1000 -random_feat True -save_model True
python main.py -test_mode True -model_name 'GAT' -wb True -lr 0.001 -n_layers 3 -wd 0 -n_heads 1 -num_epochs 1000 -save_model True
python main.py -test_mode True -model_name 'GAT' -wb True -lr 0.001 -n_layers 3 -wd 0 -n_heads 1 -num_epochs 1000 -random_feat True -save_model True
python main.py -test_mode True -model_name 'SAGE' -wb True -lr 0.0001 -n_layers 3 -wd 0.000001 -num_epochs 1000 -save_model True


