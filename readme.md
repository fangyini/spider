## Code for the paper "Spider: Deep Learning-driven Sparse Mobile TrafficMeasurement Collection and Reconstruction"

### Requirement: 
Python >=3.8, numpy>=1.12.0, pytorch>=1.9.0, gym>=0.19.0

### Usage

Run the following to train MTRNet:
```bash
python3 mtrnet.py
```

Put the trained MTRNet (mtrnet.pt) under ./data/ directory

Run the following to train the agent:
```bash
python3 train.py
```

