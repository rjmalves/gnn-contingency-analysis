# gnn-contingency-analysis
GNN for k-contingency screening in power system networks.


# Pytorch & PyG installation examples

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```


# Example use

1. Copy .env.example to .env
2. Edit python file with the desired params(ex: cge.py)
3. Run python file and visuailze.py

- CGE

```
python cge.py
python visualize.py cge
```