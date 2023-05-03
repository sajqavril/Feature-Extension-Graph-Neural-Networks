import os, sys
import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time

from FEGNN_model import FEGNN
from utils import get_data, set_best_train_args
from torch.utils.tensorboard import SummaryWriter


def one_run(args, seed, run, bar):
    
    torch.manual_seed(seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(seed)

    args.seed = seed
    data = get_data(args) 
    eye = torch.eye(args.nhid)
    model = FEGNN(ninput=data.x.shape[1], nclass=data.y.max()+1, args=args)

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        data = data.cuda(args.cuda)
        model = model.cuda(args.cuda)  
        eye = eye.cuda(args.cuda)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if not args.no_earlystop:
        best_epoch = 0
        best_val_acc = 0.
        bad_epochs = 0
        best_test_acc = 0.
        best_val_loss = 9999
        val_loss_history = []
        val_acc_history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        bar.set_description('Run:{:2d}, epoch:{:4d}'.format(run, epoch))
        model.train()
        output = model(data)  
        
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
 
        output_eval = model(data)  
        pred_eval = output_eval.argmax(dim=1)
        correct_eval = (pred_eval == data.y)

        val_loss = F.nll_loss(output_eval[data.val_mask], data.y[data.val_mask])
        val_acc = correct_eval[data.val_mask].sum() / data.val_mask.sum()
        test_acc = correct_eval[data.test_mask].sum() / data.test_mask.sum()

        bar.set_postfix(train_loss='{:.4f}'.format(loss.item()), 
                        val_loss='{:.4f}'.format(val_loss.item()),
                        val_acc='{:.4f}'.format(val_acc.item()))

        if epoch > args.warmup: # warm up = 50

            val_loss_history.append(val_loss.item())
            val_acc_history.append(val_acc.item())

            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_epoch = epoch

            if not args.no_earlystop and epoch > args.patience + args.warmup:
                tmp_loss = torch.tensor(
                    val_loss_history[-(args.patience + 1):-1])
                if val_loss > tmp_loss.mean().item():
                    runtime = time.time() - start_time
                    epoch_time = runtime / (epoch + 1)
                    print(epoch_time)
                    print('Best epoch %d for run %d: train loss: %.4f, val loss: %.4f, val acc: %.4f, test acc %.4f'%(
                        best_epoch, run, loss, best_val_loss, best_val_acc, best_test_acc))
                    break
    
    return best_test_acc.item(), best_val_acc.item()


def main():
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--cuda', type=int, default=2, help='Cuda device.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (no use).')
    
    parser.add_argument('--dataset', type=str, default='cora', help='Data set.')
    
    parser.add_argument('--dropout', type=float, default=0.5)


    parser.add_argument("--poly", type=str, default='gpr', choices=['gpr', 'cheb', 'cheb2', 'bern', 'gcn', 'ours'])
    parser.add_argument('--K', type=int, default=2)

    parser.add_argument('--attn_nhid', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--xb', action='store_true', default=False)

    # training parameters
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for linear')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--no_earlystop', action='store_true', default=False, help='Set to voyage the whole epochs.')
    parser.add_argument('--patience', type=int, default=200, help='Heads of distribution attention.')
    parser.add_argument('--runs', type=int, default=10, help='Runs to train.')

    # split parameters
    parser.add_argument('--split', type=str, default='random', choices=['random', 'set', 'grand'])
    parser.add_argument('--train_proportion', type=float, default=0.6, help='Train proportion')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='Valid proportion')
    parser.add_argument('--idx', type=int, default=0, help='For multiple graphs, e.g. ppi has 20 graphs')

    parser.add_argument('--d', type=int, default=0, help='random dicts')
    parser.add_argument('--base', type=int, default=-1, help='random dicts')

    # # reg
    # parser.add_argument('--ortho', type=float, default=0., help='Dictionary matrix othogonal regularization')
    # parser.add_argument('--sp1', type=float, default=0., help='lin1 sparsity regularization')
    # parser.add_argument('--sp2', type=float, default=0., help='lin2 sparsity regularization')

    # FEGNN
    parser.add_argument('--nx', type=int, default=-1, help='hidden size for the node feature subdictionary, default -1 for use the feature\'s size')
    parser.add_argument('--nlx', type=int, default=-1, help='hidden size for the interaction subdictionary, default -1 for use the feature\'s size')
    parser.add_argument('--nl', type=int, default=0, help='hidden size for the sturcture subdictionary, default 0 for not using this subdictionary') # chameleon 700, squirrel 2000
    parser.add_argument('--share_lx', action='store_true', default=False, help='share the same w1 for different hops of lx')
    parser.add_argument('--warmup', type=int, default=50, help='random dicts')
    parser.add_argument('--no_use_best_args', action='store_true', default=False)

    args = parser.parse_args()
    if args.dataset.lower() in ['cs', 'physics']:
        args.split = 'grand'
    elif args.dataset.lower() in ['computers', 'photo', 'chameleon', 'squirrel', 'actor', 'texas', 'cornell']:
        args.split = 'random'

    if not args.no_use_best_args: # if use the best params
        args = set_best_train_args(args)
    print(args)

    for time in range(1):
        seeds=[0,1,2,3,4,5,6,7,8,9]

        pbar = tqdm.tqdm(range(args.runs))


        test_accs = []
        val_accs = []

        for idx in pbar:
            test_acc, val_acc = one_run(args, seed=seeds[idx], run=idx, bar=pbar)
            test_accs.append(test_acc)
            val_accs.append(val_acc)

        test_acc_mean = torch.Tensor(test_accs).mean().item()
        val_acc_mean = torch.Tensor(val_accs).mean().item()

        print('Average Test acc for {:s}: {:.4f}, Val acc: {:.4f}'.format(args.dataset, test_acc_mean, val_acc_mean))


if __name__ == '__main__':
    main()
    



    
    
    

