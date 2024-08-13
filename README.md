# TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts
This is an official Pytorch implementation of TESTAM in the following paper: [TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts](https://openreview.net/forum?id=N0nTk5BSvO), ICLR 2024.


## Updates
Here we describe the changes in official TESTAM code. 
(Ongoing updates)
 - Revision of gating mechanism and meta node bank for multivariate spatio-temporal data
 - Additional implementation of experts -- CNN-based spatial modeling, Graph Attention, etc.
 - Providing processed (and also merged) version of EXPY-TKY dataset (original dataset is not processed)

(2024-08-13)
 - We revised TESTAM routing and fixed some issues -- e.g., some of the previous pseudo label-generation process are problematic with deprecated codes
 - We partially updated TESTAM to be usable for the multivariate spatio-temporal data modeling. It is now capable for the multitask (or multivariate) forecasting with multiple output dimensions
 - We provide multiple additional features such as load balancing loss (as previos MoEs), and uncertainty flag. You may refer to the engine.py
 - We additionally revised some of redundancy in the codes

## Requirements
 - python>=3.8
 - torch>=1.7.1
 - numpy>=1.12.1
 - pandas>=0.19.2
 - scipy>=0.19.0

Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

## Data Preparation

### Download Datasets
The EXPY-TKY dataset can be found in [MegaCRN Github](https://github.com/deepkashiwa20/MegaCRN).
The other datasets, including METR-LA, can be found in [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [Li et al. (DCRNN)](https://github.com/liyaguang/DCRNN).

### Process Datasets
In the data processing stage, We have the same process as [DCRNN](https://github.com/liyaguang/DCRNN).
```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY,EXPY-TKY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_fiilename=data/metr-la.h5 --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_fiilename=data/pems-bay.h5 --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH

# EXPY-TKY
python generate_training_data.py --output_dir=data/EXPY-TKY --traffic_df_fiilename=data/expy-tky.csv --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH
```

## Usage

### Model Training
We provide default training codes in `run.py`. You can train the model as follows:
```
# DATASET: {METR-LA, PEMS-BAY, EXPY-TKY}
# DEVICE: {'cpu', 'cuda:0',...,'cuda:N'}
python run.py --dataset DATASET --device DEVICE
```

For more parameter information, please refer to `train.py`.
We provide a more detailed and complete command description for the training code:

```
python -u train.py --device DEVICE --data DATA --adjdata ADJDATA --adjtype ADJTYPE
 --nhid NHID --in_dim IN_DIM --seq_length OUTDIM --num_nodes N --batch_size B
 --dropout DROPOUT --epochs EPOCHS --print_every PRINT_EVERY --seed SEED
 --save SAVE --expid EXPID --load_path LOAD_PATH --patience PATIENCE --lr_mul LR_MUL
 --n_warmup_steps N_WARMUP_STEPS --quantile Q --is_quantile IS_QUANTILE --warmup_epoch WARMUP_EPOCH
```

The detailed descriptions of the arguments are as follows:

| Argument  | Description  |
|---|---|
|device            | Device ID of GPU (default: cuda:0)|
|data              | Path to the dataset directory (default: ./data/METR-LA)|
|adjdata           | Path to the adjacency matrix file (default: ./data/METR-LA/adj_mx.pkl)|
|adjtype           | Type of adjacency matrix. (default: 'doubletransition'). It could be set to 'scalap', 'normlap', 'symnadj', 'transition', 'doubletransition', 'identity'. It is only used to check the number of nodes|
|out_dim           | Output dimensionality of TESTAM (default: 1 (i.e., speed)). It is implemented for the better use of TESTAM in the other generic spatio-temporal forecasting problems with multivariate setting.|
|nhid              | Dimension of hidden unit (default: 32)|
|in_dim            | Dimension of the input signal (default: 2 (speed, tod))|
|num_nodes         | Number of total nodes (default: 207). If you provide adjdata, `train.py` will calculate appropriate num_nodes automatically|
|batch_size        | The batch size of training input data (default: 64)|
|dropout           | The probability of dropout (default: 0.3)|
|epochs            | Total number of training epochs (default: 100)|
|print_every       | Print out the training loss per P steps  (default: 50)|
|seed              | Random seed for the debugging (default: -1) -1 means we do not provide seed number|
|save              | Path and pre-fix for the model and output files (default: ./experiment/METR-LA_TESTAM)|
|expid             | Experiment ID (default: 1)|
|load_path         | Path to the pre-trained model. If it exists, continue the training from the saved model (default: None)|
|patience          | Patience for the early stopping (default: 15). If validation loss does not improve for previous PATIENCE epochs, the training ends|
|lr_mul            | Learning rate multiplier for the CosineWarmupScheduler (default: 1). Please refer to the [Transformer (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762.pdf) and [Pytorch documents](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)|
|n_warmup_steps    | Number of steps for the CosineWarmupScheduler (default: 4000). Please refer to the [Transformer (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762.pdf) and [Pytorch documents](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)|
|quantile          | Error quantile for the routing loss function (default: 0.7)|
|is_quantile       | Flag for the routing loss function. If True, a routing loss function based on the error quantile will be used. Otherwise, a routing function comparing every expert will be used.|
|warmup_epoch      | Determines the number of warmup epochs (default: 0). During warmup epochs, routing loss is not calculated, and each expert is trained with all data samples.|

### Model Testing
For the testing, you can run the code below:
```
python test.py --device DEVICE --data DATA --adjdata ADJDATA --adjtype ADJTYPE
 --nhid NHID --in_dim IN_DIM --out_dim OUTDIM --num_nodes N --batch_size B
 --save SAVE --load_path LOAD_PATH
```

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{lee2024testam,
 title = {{TESTAM}: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts},
 author = {Hyunwook Lee and Sungahn Ko},
 booktitle = {The Twelfth International Conference on Learning Representations},
 year = {2024},
 URL = {https://openreview.net/forum?id=N0nTk5BSvO}
}
```
