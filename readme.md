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

Then train the selection predictoin model:
1. Create dataset by loading the checkpoint of the agent in train.py (Line 143) and uncomment the saving lines 
   (line 99-103 and line 128-131 )
2. Run selection_prediction.py

To evaluate Spider:
1. add the path of the selection prediction model in evaluate.py (Line 39)
2. run evaluate.py

