import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, auc
import torch
from timm.utils import AverageMeter
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_five_fold_results(base_dir):
    """
    Process the results of five-fold cross-validation.
    Args:
        base_dir (str): The directory where the fold result files are located.
    """
    all_true_labels = []
    all_predict_probs = []
    all_fprs, all_tprs = [], []

    best_results_list = []

    for fold in range(0, 5):
        # Load predictions from each fold
        file_path = os.path.join(base_dir, f"pred_{fold}.csv")
        print(file_path)
        df = pd.read_csv(file_path)
        true_labels = torch.tensor(df['Label'].values)
        predict_probs = torch.tensor(df['Prediction'].values)

        # Compute ROC curve for each fold
        fpr, tpr, thresholds = roc_curve(true_labels.numpy(), predict_probs.numpy())
        roc_auc = auc(fpr, tpr)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        all_true_labels.append(true_labels)
        all_predict_probs.append(predict_probs)

    # Concatenate all true labels and predicted probabilities
    all_true_labels = torch.cat(all_true_labels)
    all_predict_probs = torch.cat(all_predict_probs)

    # Compute overall ROC curve
    overall_fpr, overall_tpr, _ = roc_curve(all_true_labels.numpy(), all_predict_probs.numpy())

    # Interpolate AUROC range
    min_fpr = np.linspace(0, 1, 100)
    interpolated_tprs = [np.interp(min_fpr, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)]
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)

    # Plot the AUROC range with a shaded area
    plt.fill_between(min_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='lightblue', alpha=0.3, label='AUROC range')

    # Plot the overall AUROC curve
    plt.plot(min_fpr, mean_tpr, color='blue', label='Overall AUROC')

    # Plot the diagonal line representing random classification
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # Set labels and title for the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC Curves')
    plt.legend(loc='lower right')

    # Save the ROC curve plot to the disk
    roc_curve_path = os.path.join(base_dir, "curve_test.png")
    plt.savefig(roc_curve_path)
    plt.close()

    print(f"ROC curve saved to: {roc_curve_path}")

def seed_torch(seed=2024):
    """
    Set random seeds for reproducibility in PyTorch and other libraries.
    Args:
        seed (int): The seed value to use for random number generation.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model, targ_model, mm=0.9999):
    """
    Update the target model's weights using a momentum-based method.
    Args:
        model (torch.nn.Module): The current model to use for updating.
        targ_model (torch.nn.Module): The target model to be updated.
        mm (float): Momentum factor for the update. Should be between 0 and 1.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)  # Update the target model's parameters

def patch_shuffle(x, group=0, g_idx=None, return_g_idx=False):
    """
    Shuffle patches in the input tensor.
    Args:
        x (torch.Tensor): The input tensor to shuffle.
        group (int): Number of groups to divide the patches into.
        g_idx (torch.Tensor, optional): Index tensor to shuffle patches.
        return_g_idx (bool, optional): Whether to return the group indices.
    """
    b, p, n = x.size()
    ps = torch.tensor(list(range(p)))

    # Padding to match group size
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group <= 0:
        return group_shuffle(x, group)
    _n = -H % group
    H, W = H + _n, W + _n
    add_length = H * W - p
    ps = torch.cat([ps, torch.tensor([-1 for i in range(add_length)])])

    # Reshape patches and shuffle
    ps = ps.reshape(shape=(group, H//group, group, W//group))
    ps = torch.einsum('hpwq->hwpq', ps)
    ps = ps.reshape(shape=(group**2, H//group, W//group))

    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    
    ps = ps.reshape(shape=(group, group, H//group, W//group))
    ps = torch.einsum('hwpq->hpwq', ps)
    ps = ps.reshape(shape=(H, W))
    idx = ps[ps >= 0].view(p)
    
    if return_g_idx:
        return x[:, idx.long()], g_idx
    else:
        return x[:, idx.long()]

def group_shuffle(x, group=0):
    """
    Shuffle the patches in the input tensor by group.
    Args:
        x (torch.Tensor): The input tensor to shuffle.
        group (int): Number of groups to divide the patches into.
    """
    b, p, n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps, torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group, -1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps >= 0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:, idx.long()]

def optimal_thresh(fpr, tpr, thresholds, p=0):
    """
    Calculate the optimal threshold based on the false positive rate and true positive rate.
    Args:
        fpr (numpy.ndarray): False positive rates for ROC curve.
        tpr (numpy.ndarray): True positive rates for ROC curve.
        thresholds (numpy.ndarray): Threshold values corresponding to fpr and tpr.
        p (float): Weighting factor for the loss function (default 0).
    """
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def make_weights_for_balanced_classes_split(dataset):
    """
    Compute class weights for a balanced dataset.
    Args:
        dataset: The dataset containing labels.
    """
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N / len(labels[labels == c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def five_scores(bag_labels, bag_predictions, sub_typing=True):
    """
    Calculate various classification metrics.
    Args:
        bag_labels (array): True labels.
        bag_predictions (array): Predicted probabilities.
        sub_typing (bool): Whether to use binary or macro averaging.
    """
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label = np.where(bag_predictions >= threshold_optimal, 1, 0)
    bag_predictions = this_class_label
    avg = 'binary' if sub_typing else 'macro'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)

    bag_labels_array = np.array(bag_labels)

    tp = np.sum((bag_predictions == 1) & (bag_labels_array == 1))
    tn = np.sum((bag_predictions == 0) & (bag_labels_array == 0))
    fp = np.sum((bag_predictions == 1) & (bag_labels_array == 0))
    fn = np.sum((bag_predictions == 0) & (bag_labels_array == 1))
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    negative_accuracy = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    positive_accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return bag_predictions, accuracy, auc_value, precision, recall, fscore, specificity, negative_accuracy, positive_accuracy

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Create a cosine learning rate scheduler.
    Args:
        base_value (float): Initial learning rate.
        final_value (float): Final learning rate.
        epochs (int): Number of epochs.
        niter_per_ep (int): Number of iterations per epoch.
        warmup_epochs (int, optional): Number of warmup epochs.
        start_warmup_value (float, optional): Starting value for warmup.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=20, stop_epoch=50, verbose=False, save_best_model_stage=0.):
        """
        Args:
            patience (int): The number of epochs to wait after the last validation loss improvement before stopping the training.
                            Default: 20
            stop_epoch (int): The earliest epoch at which training can stop.
            verbose (bool): If True, prints a message each time validation loss improves. 
                            Default: False
            save_best_model_stage (float): The epoch stage after which the best model should be saved based on the validation loss. Default: 0.
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0  # Counter for the number of epochs without improvement in validation loss
        self.best_score = None  # Best validation loss score so far
        self.early_stop = False  # Flag to indicate if early stopping should be triggered
        self.val_loss_min = np.Inf  # Initialize with an infinite value as the minimum validation loss
        self.save_best_model_stage = save_best_model_stage  # Store the epoch stage for saving the best model

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        """
        Checks whether the validation loss has improved and applies early stopping criteria.
        
        Args:
            epoch (int): The current epoch number.
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to be saved if the validation loss improves.
            ckpt_name (str): The filename for saving the model checkpoint. Default: 'checkpoint.pt'
        """
        # Score is set to the negative validation loss if the epoch is beyond the stage to save the best model
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:  # First epoch, initialize best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:  # Validation loss has not improved
            self.counter += 1  # Increment counter for consecutive epochs without improvement
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:  # Early stopping condition met
                self.early_stop = True
        else:  # Validation loss improved
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model, ckpt_name)  # Save the model
            self.counter = 0  # Reset the counter

    def state_dict(self):
        """Returns the current state of the EarlyStopping object for checkpointing."""
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }

    def load_state_dict(self, dict):
        """Loads the state of the EarlyStopping object from a saved checkpoint."""
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves the model when the validation loss has decreased."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), ckpt_name)  # Uncomment to save the model
        self.val_loss_min = val_loss  # Update the minimum validation loss

