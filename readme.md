# Code Repo for 'Feature Expansion for Graph Neural Networks'
This is the code repository for FEGNN, where the requirements, model's implementation, the best parameters, and examples are provided. 

If you are intserested in our work, you may check more details from the [ICML 2023 Published Version](https://proceedings.mlr.press/v202/sun23p.html).

There are two steps to reproduce our results in the submission.

## Step 1: install the packages
Python 3.8.8 is required.
```txt
numpy==1.20.3
ogb==1.3.3
pandas==1.0.5
scipy==1.7.1
torch==1.9.1
torch_geometric==2.0.1
tqdm==4.61.2
```

## Step 2: run the code
Enter FEGNN_code and run: 
```
python train_FEGNN.py --dataset chameleon
```
where the dataset name and other hyper parameters are given in the train_FEGNN.py file.
```
WARNING:root:The OGB package is out of date. Your version is 1.3.2, while the latest version is 1.3.3.
Namespace(K=3, attn_nhid=8, base=-1, cuda=2, d=0, dataset='chameleon', dropout=0.5, epochs=1000, idx=0, lr=0.01, nhid=64, nl=700, nlx=-1, no_cuda=False, no_earlystop=False, nx=-1, ortho=0.0, patience=200, poly='gpr', runs=10, save_hyper=False, save_model=False, seed=42, share_lx=False, sp1=0.0, sp2=0.0, split='random', train_proportion=0.6, val_proportion=0.2, warmup=50, weight_decay=0.0005, xb=False)
Run: 0, epoch: 251:   0%|                                                                                                                                                                         | 0/10 [00:08<?, ?it/s, train_loss=0.0571, val_acc=0.7604, val_loss=0.8449]0.011345671282874214
Best epoch 51 for run 0: train loss: 0.0571, val loss: 0.7378, val acc: 0.7648, test acc 0.7536
Run: 1, epoch: 251:  10%|████████████████                                                                                                                                                 | 1/10 [00:12<01:16,  8.45s/it, train_loss=0.0585, val_acc=0.7385, val_loss=0.8819]0.011364195081922743
Best epoch 51 for run 1: train loss: 0.0585, val loss: 0.7806, val acc: 0.7363, test acc 0.7332
Run: 2, epoch: 251:  20%|████████████████████████████████▏                                                                                                                                | 2/10 [00:15<00:45,  5.69s/it, train_loss=0.0566, val_acc=0.7275, val_loss=0.9245]0.010892704365745423
Best epoch 52 for run 2: train loss: 0.0566, val loss: 0.8020, val acc: 0.7187, test acc 0.7012
Run: 3, epoch: 251:  30%|████████████████████████████████████████████████▎                                                                                                                | 3/10 [00:19<00:33,  4.77s/it, train_loss=0.0563, val_acc=0.7055, val_loss=0.9330]0.010952710159241206
Best epoch 56 for run 3: train loss: 0.0563, val loss: 0.8389, val acc: 0.7099, test acc 0.7332
Run: 4, epoch: 251:  40%|████████████████████████████████████████████████████████████████▍                                                                                                | 4/10 [00:23<00:26,  4.33s/it, train_loss=0.0538, val_acc=0.7341, val_loss=0.8369]0.011039226774185423
Best epoch 60 for run 4: train loss: 0.0538, val loss: 0.7680, val acc: 0.7429, test acc 0.7464
Run: 5, epoch: 251:  50%|████████████████████████████████████████████████████████████████████████████████▌                                                                                | 5/10 [00:26<00:20,  4.10s/it, train_loss=0.0550, val_acc=0.7495, val_loss=0.7437]0.011109634051247248
Best epoch 55 for run 5: train loss: 0.0550, val loss: 0.6759, val acc: 0.7538, test acc 0.7128
Run: 6, epoch: 251:  60%|████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                | 6/10 [00:30<00:15,  3.96s/it, train_loss=0.0574, val_acc=0.7604, val_loss=0.7276]0.011226953968169197
Best epoch 70 for run 6: train loss: 0.0574, val loss: 0.7014, val acc: 0.7604, test acc 0.7216
Run: 7, epoch: 251:  70%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 7/10 [00:34<00:11,  3.85s/it, train_loss=0.0600, val_acc=0.7495, val_loss=0.7701]0.01124359407122173
Best epoch 65 for run 7: train loss: 0.0600, val loss: 0.7223, val acc: 0.7560, test acc 0.7536
Run: 8, epoch: 251:  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                | 8/10 [00:37<00:07,  3.77s/it, train_loss=0.0580, val_acc=0.7538, val_loss=0.7707]0.011247905473860483
Best epoch 64 for run 8: train loss: 0.0580, val loss: 0.7162, val acc: 0.7538, test acc 0.7157
Run: 9, epoch: 251:  90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                | 9/10 [00:41<00:03,  3.71s/it, train_loss=0.0584, val_acc=0.7253, val_loss=0.9052]0.01114665042786371
Best epoch 55 for run 9: train loss: 0.0584, val loss: 0.8125, val acc: 0.7231, test acc 0.7609
Run: 9, epoch: 251: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:41<00:00,  4.13s/it, train_loss=0.0584, val_acc=0.7253, val_loss=0.9052]
Average Test acc for chameleon: 0.7332, Val acc: 0.7420
```

## Modify the hyper-paramters for FEGNN

In FEGNN, there are mainly SEVEN hyper-parameters that can make adjustment to compare:

``` python
# activate the adjustment
--no_use_best_args # [ACTION] !!!Please use this if you want to costomize the hyper-parameters, or only the saved params (best for FE-GNN) in FEGNN_params.csv can be used. This argument will undo the function 'set_best_train_args()' from utils.py


# main hyper-parameters
--nl # please set this to -1 to get the best performance by using all principal components of the structure matrix, e.g. adjacency. 
--lr # [FLOAT] learning rate
--weight_decay # [FLOAT]weight decay
--nhid # [INT] hidden dimension
--nl # [INT] number of principal components chosed; using all please set to -1, using none of them set to 0
--poly # [INT] polynomials for generate the first part feature subspaces, NOT THE COMPARED BASELINE!!
--K # [INT] polynomial order
```


Note: this package only implements FEGNN and its variants. For other compared baselines in our script, your may try the original code repository of the papers of the hyper-parameters we provide in the Appendix.

<!-- The code of the addictively added experiments will be sooning appended. Thanks! -->
