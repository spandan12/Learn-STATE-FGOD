import torch
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt

def top_n_accuracy(y_probs, truths, n=1):
    # y_prob: (num_images, num_classes)
    # truth: (num_images, num_classes) multi/one-hot encoding
    best_n = np.argsort(y_probs, axis=1)[:, -n:]
    if isinstance(truths, np.ndarray) and truths.shape == y_probs.shape:
        ts = np.argmax(truths, axis=1)
    else:
        # a list of GT class idx
        ts = truths

    num_input = y_probs.shape[0]
    successes = 0
    for i in range(num_input):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / num_input

def accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_true, y_preds)
    error = 1.0 - accuracy
    return accuracy, error

def multi_accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    if isinstance(y_true[0], list):  # multi-target
        successes = sum(1 for i, targets in enumerate(y_true) if y_preds[i] in targets)
        accuracy = successes / len(y_true)
    else:  # single-target
        accuracy = accuracy_score(y_true, y_preds)
    error = 1.0 - accuracy
    return accuracy, error

def compute_acc_auc(y_probs, y_true_ids):
    onehot_tgts = np.zeros_like(y_probs)
    for idx, t in enumerate(y_true_ids):
        onehot_tgts[idx, t] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = accuracy(y_probs, y_true_ids)
        # so precision can set all to 2
        try:
            auc = roc_auc_score(onehot_tgts, y_probs, average='macro')
        except ValueError as e:
            print(f"value error encountered {e}, set auc sccore to -1.")
            auc = -1
        return {"top1": top1, "rocauc": auc}

    top1, _ = accuracy(y_probs, y_true_ids)
    k = min([2, num_classes])  # if number of labels < 5, use the total class
    top5 = top_n_accuracy(y_probs, y_true_ids, k)
    return {"top1": top1, f"top{k}": top5}

def multi_compute_acc_auc(y_probs, y_true_ids):
    # Convert y_true_ids to one-hot encoding for AUC calculation
    onehot_tgts = np.zeros_like(y_probs)
    for idx, targets in enumerate(y_true_ids):
        if isinstance(targets, list):  # multiple targets
            for t in targets:
                onehot_tgts[idx, t] = 1.
        else:
            onehot_tgts[idx, targets] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = multi_accuracy(y_probs, y_true_ids)
        try:
            auc = roc_auc_score(onehot_tgts, y_probs, average='macro')
        except ValueError as e:
            print(f"Value error encountered: {e}, setting AUC score to -1.")
            auc = -1
        return {"top1": top1, "rocauc": auc}

    top1, _ = multi_accuracy(y_probs, y_true_ids)
    k = min([2, num_classes])  # if number of labels < 5, use the total class
    top5 = top_n_accuracy(y_probs, y_true_ids, k)
    return {"top1": top1, f"top{k}": top5}

def get_taxonomy():
    taxonomy = np.load({taxonomy_path}, allow_pickle=True).item()

    return taxonomy

def get_targets_logits(data):
    return data['targets'], data['joint_logits']

def get_known_and_novel_only_accuracy(known_logits, known_targets, novel_logits, novel_targets, num_leaf_nodes):
    new_novel_targets = [novel_target-num_leaf_nodes for novel_target in novel_targets]
    known_acc_dict = compute_acc_auc(known_logits[:, :num_leaf_nodes], known_targets)
    novel_acc_dict = compute_acc_auc(novel_logits[:, num_leaf_nodes:], new_novel_targets)

    return known_acc_dict, novel_acc_dict

def results_to_guarantee(known_accuracies, novel_accuracies, pts, kg):

    # novel bias increasing -> known acc decreasing
    loc = np.abs(known_accuracies - kg).argmin()
    closest = known_accuracies[loc]
    if closest < kg:
        locs = np.array([max(loc-1, 0), loc])
    elif closest == kg:
        locs = np.array([loc, loc])
    else:
        locs = np.array([loc, min(loc+1, known_accuracies.shape[0]-1)])
    
    x = known_accuracies[locs]

    y = known_accuracies[locs]
    guarantee_known = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
    y = novel_accuracies[locs]
    guarantee_novel = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
    
    return guarantee_known, guarantee_novel

def harmonic_mean(a, b): return 2.*a*b/(a+b) if a > 0. and b > 0. else 0.


def main(base_path, color, label):
    ee = 1e-8
    
    known_data = torch.load(f'{path_to_known_logits}')
    novel_data = torch.load(f'{path_to_novel_logits}')

    known_targets, known_logits = get_targets_logits(known_data)
    novel_targets_temp, novel_logits = get_targets_logits(novel_data)

    T = get_taxonomy()
    index_map = T['index_map']
    label_hnd = T['label_hnd']
    
    novel_targets = [index_map[target] for target in novel_targets_temp]
    novel_biases =  np.arange(-4.55,4.55,0.5)

    all_known_accuracies = np.zeros(novel_biases.shape[0])
    all_novel_accuracies = np.zeros(novel_biases.shape[0])

    all_known_accuracies_top2 = np.zeros(novel_biases.shape[0])
    all_novel_accuracies_top2 = np.zeros(novel_biases.shape[0])

    novel_classes = np.arange(len(T['wnids_leaf']), len(T['wnids']))
    
    for b, novel_bias in enumerate(novel_biases):
        known_logits_biased = torch.tensor(known_logits).clone()
        novel_logits_biased = torch.tensor(novel_logits).clone()

        known_logits_biased[:, novel_classes] += novel_bias
        novel_logits_biased[:, novel_classes] += novel_bias

        known_acc_dict = compute_acc_auc(known_logits_biased.numpy(), known_targets)
        novel_acc_dict = compute_acc_auc(novel_logits_biased.numpy(), novel_targets)
        

        all_known_accuracies[b] = known_acc_dict['top1']
        all_novel_accuracies[b] = novel_acc_dict['top1']
        all_novel_accuracies[b] = novel_acc_dict
    
    gknown, gnovel = results_to_guarantee(all_known_accuracies, all_novel_accuracies, novel_biases, 0.5)
    gknown *= 100
    gnovel *= 100
    auc = -np.trapz(all_novel_accuracies,all_known_accuracies) * 100
    plt.plot(all_known_accuracies, all_novel_accuracies, color=color, label=label)
    plt.scatter(all_known_accuracies, all_novel_accuracies, color=color)
    plt.title(f'AUC:{auc:.3f} OA@50:{gnovel:.3f}')
    plt.grid()
    plt.xlabel('Closed-set class accuracy')
    plt.ylabel('Open-set class accuracy')
    
    

    hmean = harmonic_mean(gknown, gnovel)

    print(f'AUC:{auc:.2f} Harmonic Mean:{hmean:.2f} OA@50:{gnovel:.2f} ')
    return 
    

if __name__ == '__main__':
    
    
    main({path_to_logits}, 'red', 'state')

    plt.legend()
    plt.savefig(f'{path_to_logits}/top1_auc_curve.png')
    