import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import suppress
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler

from timm.utils import AverageMeter
from timm.models import model_parameters

from dataloader import *
from modules import compl
from utils import *

def evaluate(model, loader, device, criterion, loss, is_training=False):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The neural network model to be evaluated.
        loader: DataLoader instance to load the dataset.
        device: Device (CPU or GPU) on which the model and data will reside.
        criterion: Loss function used for evaluation.
        loss: Type of loss function ('ce' for CrossEntropyLoss, 'bce' for BCEWithLogitsLoss).
        is_training: Boolean flag indicating whether the evaluation is during training (used for tracking metrics).

    Returns:
        bag_id: List of bag identifiers.
        bag_logit: List of model predictions.
        bag_labels: List of true labels for each data point.
        test_loss_log: Total loss for the evaluation.
        loss_meter: Average loss if training; None otherwise.
    """
    model.eval()  # Set the model to evaluation mode
    bag_logit, bag_labels, bag_id = [], [], []  # Initialize containers for logits, labels, and bag ids
    loss_meter = AverageMeter() if is_training else None  # For tracking loss during training
    test_loss_log = 0.  # Initialize total loss

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for i, data in enumerate(tqdm(loader)):
            bag_id.append(data[2][0])  # Add bag ID to the list

            # Handle labels based on the shape of data[1]
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            # Move inputs to device (GPU or CPU)
            if isinstance(data[0], (list, tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag = data[0]
                batch_size = data[0][0].size(0)
            else:
                bag = data[0].to(device)
                batch_size = bag.size(0)
            
            label = data[1].to(device)  # Move label to device
            test_logits = model(bag)  # Forward pass through the model

            # Compute loss based on specified type
            if loss == 'ce':
                test_loss = criterion(test_logits.view(batch_size, -1), label)
                if batch_size > 1:
                    bag_logit.extend(torch.softmax(test_logits, dim=-1)[:, 1].cpu().squeeze().numpy())
                else:
                    bag_logit.append(torch.softmax(test_logits, dim=-1)[:, 1].cpu().squeeze().numpy())
            elif loss == 'bce':
                test_loss = criterion(test_logits.view(batch_size, -1), label.view(1, -1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            # Update the loss meter if in training mode
            if is_training:
                loss_meter.update(test_loss, 1)
            else:
                test_loss_log += test_loss.item()

    return bag_id, bag_logit, bag_labels, test_loss_log, loss_meter.avg if is_training else None

def main(args):
    """
    Main entry point for training and evaluating the model.

    Args:
        args: Argument parser object containing all the configuration and hyperparameters.
    """
    seed_torch(args.seed)  # Set the seed for reproducibility

    # Load dataset based on selected option
    if args.datasets.lower() == 'pdac':
        train_p, train_l, test_p, test_l, val_p, val_l = load_pdac_data(args.dataset_root, args.cv_fold)

    # Initialize metric containers
    acs, pre, rec, fs, auc, spec, neg_acc, pos_acc, te_auc, te_fs = [], [], [], [], [], [], [], [], [], []
    ckc_metric = [acs, pre, rec, fs, auc, spec, neg_acc, pos_acc, te_auc, te_fs]

    # Prepare log file path
    if not args.test_only:
        log_file_path = os.path.join(args.model_path, 'log.txt')
    else:
        log_file_path = os.path.join(args.model_path, 'log_test.txt')

    # Log dataset being used
    with open(log_file_path, 'a') as log_file:
        log_file.write('Dataset: ' + args.datasets + '\n')

    # Cross-validation loop
    for k in range(args.fold_start, args.cv_fold):
        with open(log_file_path, 'a') as log_file:
            log_file.write('\n Start %d-fold cross validation: fold %d' % (args.cv_fold, k))
        ckc_metric = one_fold(args, k, ckc_metric, train_p, train_l, test_p, test_l, val_p, val_l)
    
    # Log the final metrics after cross-validation
    with open(log_file_path, 'a') as log_file:
        log_file.write('\n Cross validation accuracy mean: %.4f, std %.4f \n' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        log_file.write('Cross validation auc mean: %.4f, std %.4f \n' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        log_file.write('Cross validation precision mean: %.4f, std %.4f \n' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        log_file.write('Cross validation recall mean: %.4f, std %.4f \n' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        log_file.write('Cross validation fscore mean: %.4f, std %.4f \n' % (np.mean(np.array(fs)), np.std(np.array(fs))))
        log_file.write('Cross validation specificity mean: %.4f, std %.4f \n' % (np.mean(np.array(spec)), np.std(np.array(spec))))
        log_file.write('Cross validation negative_accuracy mean: %.4f, std %.4f \n' % (np.mean(np.array(neg_acc)), np.std(np.array(neg_acc))))
        log_file.write('Cross validation positive_accuracy mean: %.4f, std %.4f \n' % (np.mean(np.array(pos_acc)), np.std(np.array(pos_acc))))

    process_five_fold_results(args.model_path)
    
def one_fold(args, k, ckc_metric, train_p, train_l, test_p, test_l, val_p, val_l):
    """
    Performs a single fold of training and evaluation for a model.

    Args:
        args: Command line arguments containing configuration settings.
        k: The fold index for cross-validation.
        ckc_metric: List containing initial metrics for evaluation.
        train_p: List of paths to training images.
        train_l: List of labels for training images.
        test_p: List of paths to testing images.
        test_l: List of labels for testing images.
        val_p: List of paths to validation images.
        val_l: List of labels for validation images.

    Returns:
        List of updated evaluation metrics including accuracy, precision, recall, fscore, AUC, specificity, negative accuracy, and positive accuracy.
    """

    # Set random seed for reproducibility
    seed_torch(args.seed)

    # Initialize loss scaler for mixed-precision training if enabled
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress

    # Set device to CUDA if available, otherwise use CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Unpack evaluation metrics
    acs, pre, rec, fs, auc, spec, neg_acc, pos_acc, te_auc, te_fs = ckc_metric

    # Log file setup
    if not args.test_only:
        log_file_path = os.path.join(args.model_path, 'log.txt')
    else:
        log_file_path = os.path.join(args.model_path, 'log_test.txt')

    # Dataset loading for PDAC dataset
    if args.datasets.lower() == 'pdac':
        train_set = PDACDataset(train_p[k], train_l[k], root=args.dataset_root, persistence=args.persistence, keep_same_psize=args.same_psize, is_train=True)
        test_set = PDACDataset(test_p[k], test_l[k], root=args.dataset_root, persistence=args.persistence, keep_same_psize=args.same_psize)

        if args.val_ratio != 0.:
            val_set = PDACDataset(val_p[k], val_l[k], root=args.dataset_root, persistence=args.persistence, keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    # DataLoader for training, validation, and test sets
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model initialization based on selected model type
    if args.model == 'compl':
        model_params = {
            'input_dim': args.input_dim,
            'mlp_dim': args.mlp_dim,
            'n_classes': args.n_classes,
            'multi_scale': args.multi_scale,
            'embed_weights': args.embed_weights,
            'dropout': args.dropout,
            'act': args.act,
            'region_num': args.region_num,
            'pos': args.pos,
            'pos_pos': args.pos_pos,
            'pool': args.pool,
            'peg_k': args.peg_k,
            'drop_path': args.drop_path,
            'n_layers': args.n_trans_layers,
            'n_heads': args.n_heads,
            'attn': args.attn,
            'da_act': args.da_act,
            'trans_dropout': args.trans_drop_out,
            'ffn': args.ffn,
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'min_region_num': args.min_region_num,
            'qkv_bias': args.qkv_bias,
            'epeg_k': args.epeg_k,
            'epeg_2d': args.epeg_2d,
            'epeg_bias': args.epeg_bias,
            'epeg_type': args.epeg_type,
            'region_attn': args.region_attn,
            'peg_1d': args.peg_1d,
            'cr_msa': args.cr_msa,
            'crmsa_k': args.crmsa_k,
            'all_shortcut': args.all_shortcut,
            'crmsa_mlp': args.crmsa_mlp,
            'crmsa_heads': args.crmsa_heads,
        }
        model = compl.COMPL(**model_params).to(device)

    # Loss function setup based on selected type
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Optimizer setup based on selected type
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler setup
    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch * len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    # Early stopping setup
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=70, save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    # Training loop, evaluation, and model checkpointing
    if not args.test_only:
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch, opt_spec, opt_neg_acc, opt_pos_acc = 0, 0, 0, 0, 0, 0, 0, 0, 0
        opt_te_auc, opt_tea_auc, opt_te_fs, opt_te_tea_auc, opt_te_tea_fs = 0., 0., 0., 0., 0.

        train_time_meter = AverageMeter()
        for epoch in range(args.num_epoch):
            train_loss, start, end = train_loop(args, model, train_loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, k, epoch)
            train_time_meter.update(end - start)
            stop, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy, test_loss = val_loop(args, model, val_loader, device, criterion, early_stopping, epoch)

            # Log training and evaluation metrics
            with open(log_file_path, 'a') as log_file:
                log_file.write('\r Epoch [%d/%d] train loss: %.4f, test loss: %.4f, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f, specificity: %.4f, negative_accuracy: %.4f, positive_accuracy: %.4f, time: %.4f(%.4f)' % 
                (epoch + 1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy, train_time_meter.val, train_time_meter.avg))

            # Save best model based on AUC value
            if auc_value > opt_auc and epoch >= args.save_best_model_stage * args.num_epoch:
                optimal_ac = accuracy
                opt_pre = precision
                opt_re = recall
                opt_fs = fscore
                opt_auc = auc_value
                opt_epoch = epoch
                opt_spec = specificity
                opt_neg_acc = negative_accuracy
                opt_pos_acc = positive_accuracy

                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)

                best_pt = {
                    'model': model.state_dict(),
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))

            # Save checkpoint
            random_state = {
                'np': np.random.get_state(),
                'torch': torch.random.get_rng_state(),
                'py': random.getstate()
            }
            ckp = {
                'model': model.state_dict(),
                'lr_sche': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'k': k,
                'early_stop': early_stopping.state_dict(),
                'random': random_state,
                'ckc_metric': [acs, pre, rec, fs, auc, spec, neg_acc, pos_acc, te_auc, te_fs],
                'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch, opt_spec, opt_neg_acc, opt_pos_acc],
                'te_best_metric': [opt_te_auc, opt_te_fs, opt_te_tea_auc, opt_te_tea_fs]
            }
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

            # Stop early if needed
            if stop:
                break

        # Log optimal metrics after training
        with open(log_file_path, 'a') as log_file:
            log_file.write('\n Optimal accuracy: %.4f, Optimal auc: %.4f, Optimal precision: %.4f, Optimal recall: %.4f, Optimal fscore: %.4f, Optimal specificity: %.4f, Optimal negative_accuracy: %.4f, Optimal positive_accuracy: %.4f' % (optimal_ac, opt_auc, opt_pre, opt_re, opt_fs, opt_spec, opt_neg_acc, opt_pos_acc))

    # Load best model for testing
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
    info = model.load_state_dict(best_std['model'])
    print(info)

    # Run final test and save predictions
    accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy, test_loss_log, pred_df = test(args, model, test_loader, device, criterion)
    pred_df.to_csv(os.path.join(args.model_path, 'pred_{fold}.csv'.format(fold=k)))

    # Log test metrics
    with open(log_file_path, 'a') as log_file:
        log_file.write('\n Test accuracy: %.4f, Test auc: %.4f, Test precision: %.4f, Test recall: %.4f, Test fscore: %.4f, Test specificity: %.4f, Test negative_accuracy: %.4f, Test positive_accuracy: %.4f \n' % 
        (accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy))

    # Append metrics to lists
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)
    spec.append(specificity)
    neg_acc.append(negative_accuracy)
    pos_acc.append(positive_accuracy)

    return [acs, pre, rec, fs, auc, spec, neg_acc, pos_acc, te_auc, te_fs]


def train_loop(args, model, loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, k, epoch):
    """
    Train loop for the model during one epoch.
    
    Args:
        args: Command-line arguments containing various configurations for the model.
        model: The model to be trained.
        loader: DataLoader for the training data.
        optimizer: Optimizer used for updating model weights.
        device: Device (CPU or GPU) where computations will take place.
        amp_autocast: Automatic mixed precision context manager.
        criterion: Loss function for the model.
        loss_scaler: Scaler used for handling mixed precision loss scaling.
        scheduler: Learning rate scheduler.
        k: Current fold (for cross-validation).
        epoch: Current epoch number.

    Returns:
        train_loss_log: Average training loss for this epoch.
        start: Time when the training starts.
        end: Time when the training ends.
    """
    start = time.time()

    train_loss_log = 0.
    model.train()

    log_file_path = os.path.join(args.model_path, 'log.txt')

    # Iterate over the batches in the training loader
    for i, data in enumerate(tqdm(loader)):
        optimizer.zero_grad()

        # Handle the data input, either a list or a single tensor
        if isinstance(data[0], (list, tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag = data[0]
            batch_size = data[0][0].size(0)
        else:
            bag = data[0].to(device)  # Input data (b*n*1024)
            batch_size = bag.size(0)
        
        label = data[1].to(device)
        
        # Forward pass with mixed precision
        with amp_autocast():
            # Shuffle the input patches if specified
            if args.patch_shuffle:
                bag = patch_shuffle(bag, args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag, args.shuffle_group)

            # Model inference
            train_logits = model(bag)

            # Compute the loss
            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size, -1), label)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size, -1), one_hot(label.view(batch_size, -1).float(), num_classes=2))
        
        # Apply dynamic weighting if specified
        if args.dynamic_weight:
            probabilities = F.softmax(train_logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            label_one_hot = F.one_hot(label, num_classes=probabilities.size(-1)).float()
            true_label_prob = torch.sum(probabilities * label_one_hot, dim=-1)
            
            # Adjust weight based on prediction accuracy and label probability
            weight = torch.where(predicted_labels == label, torch.tensor(1.0, device='cuda:0'), 10 * (1 - true_label_prob))
            logit_loss = logit_loss * weight
        
        else:
            # Adjust the loss based on the label if not using dynamic weighting
            if label == 0:
                logit_loss = logit_loss * args.label_weights[0]
            elif label == 1:
                logit_loss = logit_loss * args.label_weights[1]

        train_loss = logit_loss

        # Backpropagation and optimizer step
        train_loss.backward()
        optimizer.step()

        # Adjust the learning rate scheduler if specified
        if args.lr_supi and scheduler is not None:
            scheduler.step()

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    # Compute the average training loss for the epoch
    train_loss_log = train_loss_log / len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()

    return train_loss_log, start, end


def val_loop(args, model, loader, device, criterion, early_stopping, epoch):
    """
    Validation loop to evaluate model performance after each training epoch.
    
    Args:
        args: Command-line arguments containing various configurations for the model.
        model: The model to be validated.
        loader: DataLoader for the validation data.
        device: Device (CPU or GPU) where computations will take place.
        criterion: Loss function for the model.
        early_stopping: EarlyStopping instance to monitor validation performance.
        epoch: Current epoch number.

    Returns:
        stop: Whether early stopping should be triggered.
        accuracy: Validation accuracy.
        auc_value: Area under the curve (AUC) value for validation.
        precision: Precision score for validation.
        recall: Recall score for validation.
        fscore: F1 score for validation.
        specificity: Specificity score for validation.
        negative_accuracy: Negative accuracy for validation.
        positive_accuracy: Positive accuracy for validation.
        val_loss: Average validation loss for this epoch.
    """
    bag_id, bag_logit, bag_labels, _, val_loss = evaluate(model, loader, device, criterion, args.loss, is_training=True)
    
    # Compute various metrics for validation
    bag_predictions, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy = five_scores(bag_labels, bag_logit, args.n_classes == 2)
    
    # Apply early stopping if applicable
    if early_stopping is not None:
        early_stopping(epoch, -auc_value, model)
        stop = early_stopping.early_stop
    else:
        stop = False
    
    return stop, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy, val_loss


def test(args, model, loader, device, criterion):
    """
    Test loop to evaluate the model on the test set.
    
    Args:
        args: Command-line arguments containing various configurations for the model.
        model: The model to be tested.
        loader: DataLoader for the test data.
        device: Device (CPU or GPU) where computations will take place.
        criterion: Loss function for the model.

    Returns:
        accuracy: Test accuracy.
        auc_value: Area under the curve (AUC) value for testing.
        precision: Precision score for testing.
        recall: Recall score for testing.
        fscore: F1 score for testing.
        specificity: Specificity score for testing.
        negative_accuracy: Negative accuracy for testing.
        positive_accuracy: Positive accuracy for testing.
        test_loss_log: Average test loss.
        pred_df: DataFrame containing predictions and labels.
    """
    bag_id, bag_logit, bag_labels, test_loss_log, _ = evaluate(model, loader, device, criterion, args.loss)
    
    # Compute various metrics for testing
    bag_predictions, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy = five_scores(bag_labels, bag_logit, args.n_classes == 2)
    test_loss_log = test_loss_log / len(loader)

    # Prepare predictions and labels for saving to CSV
    pred_df = pd.DataFrame({
        'Slide_id': bag_id,
        'Label': bag_labels,
        'Prediction': bag_logit,
        'Pred_Label': bag_predictions
    })

    return accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy, test_loss_log, pred_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset Parameters
    parser.add_argument('--datasets', default='pdac', type=str, help='[pdac, other]')
    parser.add_argument('--dataset_root', default='/data/COMPL/WSI/FT', type=str, help='Dataset root path')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation fold [5]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory')
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')

    # Training Parameters
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Max training epochs in the early stopping [130]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--dynamic_weight', action='store_true', help='Dynamic weight')

    # Model Parameters
    parser.add_argument('--input_dim', default=1024, type=int, help='Dim of input features.')
    parser.add_argument('--mlp_dim', default=256, type=int, help='Dim of output features.')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--label_weights', nargs='+', type=float, default=[1.0, 1.0], help='Label weight')

    # Optimization Parameters
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Decay of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [1e-5]')

    # Model Architecture Parameters
    parser.add_argument('--model', default='compl', type=str, help='Model name')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')

    # Transformer Architecture Parameters
    parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    parser.add_argument('--pool', default='attn', type=str, help='Classification pooling. Use abmil.')
    parser.add_argument('--ffn', action='store_true', help='Feed-forward network. Only for ablation')
    parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layers in the transformer')
    parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    parser.add_argument('--qkv_bias', action='store_false')
    parser.add_argument('--all_shortcut', action='store_true', help='x = x + rrt(x)')

    # Multi-scale Transformer Parameters
    parser.add_argument('--multi_scale', default=3, type=int, help='Multi scale layers')
    parser.add_argument('--embed_weights', type=float, nargs=3, default=[0.3333, 0.3333, 0.3333], help='Embedding weights for the model.')

    # Cross-regional Transformer Parameters (R-MSA, CR-MSA, DAttention, etc.)
    parser.add_argument('--region_attn', default='native', type=str, help='Only for ablation')
    parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    parser.add_argument('--region_num', default=8, type=int, help='Number of regions [8,12,16,...]')
    parser.add_argument('--trans_dim', default=64, type=int, help='Only for ablation')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of heads in the R-MSA')
    parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
    parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    parser.add_argument('--crmsa_k', default=5, type=int, help='K of the CR-MSA. [1,3,5]')
    parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')
    parser.add_argument('--da_act', default='tanh', type=str, help='Activation func in the DAttention [gelu,relu,tanh]')

    # Position Embedding Parameters
    parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    parser.add_argument('--epeg', action='store_false', help='Enable EPEG')
    parser.add_argument('--epeg_bias', action='store_false', help='Enable conv bias')
    parser.add_argument('--epeg_2d', action='store_true', help='Enable 2d conv. Only for ablation')
    parser.add_argument('--epeg_k', default=9, type=int, help='K of the EPEG [9,15,21,...]')
    parser.add_argument('--epeg_type', default='attn', type=str, help='Only for ablation')

    # Shuffle Parameters
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of shuffle groups')

    # Experimental Parameters
    parser.add_argument('--title', default='default', type=str, help='Title of experiment')
    parser.add_argument('--project', default='PDAC', type=str, help='Project name of experiment')

    # Miscellaneous Settings
    parser.add_argument('--seed', default=2024, type=int, help='Random number seed [2024]')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--model_path', default='/data/COMPL/output', type=str, help='Output path')
    parser.add_argument('--test_only', action='store_true', help='Test only')

    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.model_path, args.project)):
        os.mkdir(os.path.join(args.model_path, args.project))
    args.model_path = os.path.join(args.model_path, args.project, args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    print(args)

    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    main(args=args)

