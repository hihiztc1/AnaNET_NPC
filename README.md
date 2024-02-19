# AnaNET: Anatomical Network for Aggregated Time Series Forecasting in Multi-Layered Architecture

This repo is the official Pytorch implementation of AnaNET. 

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda env create -f environment.yml
conda activate AnaNET
```

### Data Preparation

We provide the AggTS dataset used in the experiments in the `./dataset` directory.


### Training Example
- In `scripts/ `, we offer scripts for training datasets related to *MD* and *ETT*.

For example:

To train the **AnaNET** on **MD dataset**, you can use the scipt `scripts/run_MD.sh`:
```
sh scripts/run_MD.sh
```

## Acknowledgment
We appreciate the following GitHub repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/cure-lab/LTSF-Linear

https://github.com/yuqinie98/PatchTST

https://github.com/EdgeBigBang/EasyTS

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer
