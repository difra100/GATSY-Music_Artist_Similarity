# GATSY: Music Artist Similarity with Graph Attention Network  
Create a new conda environment:  
```
conda create -n GATSY python=3.9 && conda activate GATSY && conda install pip
```  
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
bash install_pyg.sh
```
```
pip install -r requirements.txt
```
To reproduce the experiments, you need to download the data at this [link](https://www.dropbox.com/scl/fi/b89uq8mynb2n6s2rqemjc/data_extracted_olga.zip?rlkey=ioyrzytmpgvhh3jnctv547o5i&st=vl737p5k&dl=0). 

Then, to reproduce the experiments with full data, you should access the full data folder, and run  

``` 
bash run_exp.sh
```
To repoduce the experiments with the reduced data, namely the unsupervised and supervised experiments, you should use the specifications in the notebooks `Train_Artist_similarity.ipynb`, and `Test_Artist_similarity.ipynb`.  
The recommender system demo is in `Test_Artist_similarity.ipynb` for the reduced dataset (you should train a model before).  
While you can use `demo_recommender.ipynb` with the already trained models for the full data recommender system.

 
In the reduced dataset folder insert the `adjacency`, `adjacencyCOO`, `instance` and `random_instance` files in the same folder of `main.py`.   
In the full dataset folder, insert the associated files inside the `data` folder.
