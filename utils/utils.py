import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

# from torch.utils.data.dataloader import default_collate
# import torch_geometric
# from torch_geometric.data import Batch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

def collate_MIL_survival_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    event_time = np.array([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]

def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if mode == 'coattn':
        collate = collate_MIL_survival_sig
    elif mode == 'cluster':
        collate = collate_MIL_survival_cluster
    else:
        collate = collate_MIL_survival

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    pdb.set_trace()
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = 0 if T_cont \in (-inf, 0), Y = 1 if T_cont \in [0, a_1),  Y = 2 if T_cont in [a_1, a_2), ..., Y = k if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = 0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=0) = 0
# h(0) = 0 ---> do not need to model
# S(0) = P(Y > 0 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 1,2,...,k
corresponding Y = 1, ..., k. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
# def neg_likelihood_loss(hazards, Y, c):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#   # without padding, S(1) = S[0], h(1) = h[0]
#   S_padded = torch.cat([torch.ones_like(c), S], 1) #S(0) = 1, all patients are alive from (-inf, 0) by definition
#   # after padding, S(0) = S[0], S(1) = S[1], etc, h(1) = h[0]
#   #h[y] = h(1)
#   #S[1] = S(1)
#   neg_l = - c * torch.log(torch.gather(S_padded, 1, Y)) - (1 - c) * (torch.log(torch.gather(S_padded, 1, Y-1)) + torch.log(hazards[:, Y-1]))
#   neg_l = neg_l.mean()
#   return neg_l


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

# def nll_loss(hazards, Y, c, S=None, alpha=0.4, eps=1e-8):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   if S is None:
#       S = 1 - torch.cumsum(hazards, dim=1) # surival is cumulative product of 1 - hazards
#   uncensored_loss = -(1 - c) * (torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#   censored_loss = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps))
#   loss = censored_loss + uncensored_loss
#   loss = loss.mean()
#   return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    #reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg

def get_custom_exp_code_old(args):
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    ### Model Type
    # gene
    if args.model_type == 'max_net':
      param_code += 'SNN'
    elif args.model_type == 'SNN':
      param_code += 'SNN'
    elif args.model_type == 'SNNTrans':
      param_code += 'SNNTrans'
    elif args.model_type == 'MLP':
      param_code += 'MLP'
    # patch
    elif args.model_type == 'MIL_Cluster': # 没调通
      param_code += 'MIFCN'
    elif args.model_type == 'AMIL':
      param_code += 'AMIL'
    elif args.model_type == 'DeepSets':
      param_code += 'DS'
    elif args.model_type == 'CLAM_SB':
      param_code += 'CLAM_SB'
    elif args.model_type == 'CLAM_MB':
      param_code += 'CLAM_MB'
    elif args.model_type == 'TransMIL':
      param_code += 'TransMIL'
    # multimodal
    elif args.model_type == 'MCAT':
      param_code += 'MCAT'
    elif args.model_type == 'Porpoise':
      param_code += 'PORPOISE'
    elif args.model_type == 'MOTCat':
      param_code += 'MOTCat'
    elif args.model_type == 'CMAT':
      param_code += 'CMTA'
    elif args.model_type == 'SurvPath':
        if args.use_nystrom:
            param_code += 'SurvPath_with_nystrom'
        else:
            param_code += 'SurvPath'
    elif args.model_type == 'PathOmics':
      param_code += 'PathOmics'
    elif args.model_type == 'LD_CAVE':
      param_code += 'LD_CAVE'
    elif args.model_type == 'HSFSurv':
      param_code += 'HSFSurv'
    else:
      raise NotImplementedError

    ### Loss Function porpoise
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
      param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
      param_code += '_gc%s' % str(args.gc)

    ### Applying Which Features
    if args.apply_sigfeats:
      param_code += '_sig'
      dataset_path += '_sig'


    ### Fusion Operation
    if args.fusion != "None":
      param_code += '_' + args.fusion

    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args

def get_custom_exp_code(args):
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    ### Loss Function porpoise
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
      param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
      param_code += '_gc%s' % str(args.gc)

    ### Applying Which Features
    if args.apply_sigfeats:
      param_code += '_sig'
      dataset_path += '_sig'


    ### Fusion Operation
    # if args.fusion != None:
    # args.fusion = 'None' if args.fusion == None else args.fusion
    # param_code += '_' + args.fusion
    param_code += f"_{args.fusion}"

    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args

### HSFSurv utils
def lable_entropy(pro1, pro2, smooth = 1e-6):

    index1_of_max = torch.argmax(pro1)
    index2_of_max = torch.argmax(pro2)
    class_num =  pro1.shape
    smooth = torch.tensor(smooth).to(pro1.device)
    if index1_of_max == index2_of_max:
        p_mean = (pro1 + pro2) / torch.tensor(2.0).to(pro1.device)
        res = torch.sum(-(p_mean*torch.log(p_mean+smooth) / torch.tensor(np.log(class_num)).to(pro1.device)))

    else:

        p_mean = (pro1 + pro2) / torch.tensor(2.0).to(pro1.device)
        p1 = pro1[index1_of_max]
        p2 = pro2[index2_of_max]
        q = -(p1*torch.log(p1+smooth) + p2*torch.log(p2+smooth)) / torch.tensor(np.log(class_num)).to(pro1.device)

        res = torch.sum(-(p_mean * torch.log(p_mean + smooth) / torch.tensor(np.log(class_num)).to(pro1.device))) + q

    return res

class Queue_con(nn.Module):
    def __init__(self, class_num=4, dim=256):
        super().__init__()
        self.class_num = class_num
        self.dim = dim
        self.que = torch.randn(class_num,  dim)
        self.que_pa = torch.randn(class_num,  dim)

    @torch.no_grad()
    def update_queue(self, keys, index, m=0.9):
        keys = keys.to(self.que.device)
        if isinstance(index, torch.Tensor):
            # 如果是 PyTorch 张量，移到 CPU 再转换为 numpy
            index = index.cpu().numpy()
        else:
            # 如果已经是 numpy 数组，直接使用
            index = index
        temp = self.que[index, :] * torch.tensor(m, dtype=float, device=self.que.device)
        self.que[index, :] = torch.tensor(1-m, dtype=float) * keys.unsqueeze(0).cpu() + temp
        return self.que

    @torch.no_grad()
    def update_queue_pa(self, keys, index, m=0.9):
        keys = keys.to(self.que.device)
        if isinstance(index, torch.Tensor):
            # 如果是 PyTorch 张量，移到 CPU 再转换为 numpy
            index = index.cpu().numpy()
        else:
            # 如果已经是 numpy 数组，直接使用
            index = index
        temp = self.que_pa[index, :] * torch.tensor(m, dtype=float, device=self.que.device)
        self.que_pa[index, :] = torch.tensor(1-m, dtype=float, device=self.que.device) * keys.unsqueeze(0) + temp
        return self.que_pa

    @torch.no_grad()
    def get_que(self):
        return self.que

    @torch.no_grad()
    def get_que_pa(self):
        return self.que_pa

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def InfoNce(query, positive_key, label=0, c=0, que=None, epoch=25, temperature=0.1, reduction='mean', class_num=4,
            dim=256):

    query, positive_key = normalize(query, positive_key)
    positive_logit = query * positive_key

    if c == 1:
        ori_arr = np.array([i for i in range(label, class_num)])
        length = len(ori_arr)
        pro_arr = [1, 0.5, 0.25, 0.125]
        pro_arr = np.array(pro_arr[:length])
        pro_arr = pro_arr / pro_arr.sum()
        label = np.random.choice(ori_arr, size=1, p=pro_arr)

    temp_neg = torch.zeros(class_num - 1, dim).to(query.device)
    temp_neg_pa = torch.zeros(class_num - 1, dim).to(query.device)
    step = 0
    for i in range(class_num):
        if label == i:
            step -= 1
            continue
        else:
            temp_neg[step, :] = que.get_que()[i, :]
            temp_neg_pa[step, :] = que.get_que_pa()[i, :]
        step += 1
    mo = min((1 - 0.5 ** epoch), 0.9)
    que.update_queue(query, label, mo)  # update query=path_embedding,
    que.update_queue_pa(positive_key, label, mo) # update positive_key=omic_embedding

    negative_logits = (temp_neg*query).to(query.device)
    negative_logits_pa = (temp_neg_pa*positive_key).to(query.device)

    # First index in last dimension are the positive samples
    logits = torch.cat([positive_logit, negative_logits], dim=0)
    logits1 = torch.cat([positive_logit, negative_logits_pa], dim=0)
    labels = torch.arange(len(logits), device=query.device)
    labels1 = torch.arange(len(logits1), device=query.device)
    loss = 0.5 * F.cross_entropy(logits / temperature, labels, reduction=reduction) + 0.5 * F.cross_entropy(
        logits1 / temperature, labels1, reduction=reduction)
    return loss


class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, class_num=4,  dim=256, K=8, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.K = K
        self.class_num = class_num
        self.dim = dim

    def forward(self, query, positive_key,  label = 0, c=0, que=None, temperature=0.1, reduction='mean'):

        query, positive_key = self.normalize(query, positive_key)

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True).T

        # Cosine between all query-negative combinations
        if c==1:
            ori_arr = np.array([i for i in range(label, self.class_num)])
            length = len(ori_arr)
            pro_arr = [1, 0.5, 0.25, 0.125]
            pro_arr = np.array(pro_arr[:length])
            pro_arr = pro_arr / pro_arr.sum()
            label = np.random.choice(ori_arr, size=1, p=pro_arr)

        temp_neg = torch.zeros(self.class_num-1, self.K, self.dim).to(query.device)
        step = 0
        for i in range(self.class_num):
            if label == i:
                step -= 1
                continue
            else:
                temp_neg[step,:,:] = que.get_queue()[i,:,:]
            step += 1
        que.update_queue(positive_key, label)

        negative_logits = torch.einsum('n k d, t d -> n t', temp_neg, query).to(query.device)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=0)
        labels = torch.arange(len(logits), device=query.device)
        loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
        return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


### CCL utils
class CohortLoss(object):
    def __init__(self, temperature=2, alpha=10):
        """
        Args:
            temperature: 温度参数，用于对比学习
            alpha: 内部损失的权重参数
        """
        self.temperature = temperature
        self.alpha = alpha
        # 创建mask矩阵
        self.mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]])

    def _setup_device(self, tensor):
        """设置设备，确保mask在与输入数据相同的设备上"""
        if tensor.is_cuda and not self.mask.is_cuda:
            self.mask = self.mask.cuda()
        elif not tensor.is_cuda and self.mask.is_cuda:
            self.mask = self.mask.cpu()

    def _compute_intra_loss(self, indiv, origs):
        """计算内部相似性损失"""
        indiv_know = indiv.view(4, 1, -1)  # common, synergy, g_spec, p_spec
        orig = torch.cat(origs, dim=1).detach()  # gene, path

        sim = F.cosine_similarity(indiv_know, orig, dim=-1)
        intra_loss = torch.mean(torch.abs(sim) * (1 - self.mask) - self.mask * sim) + 1
        return intra_loss

    def _compute_inter_loss(self, indiv_know, cohort, gt_label, c):
        """计算样本间对比损失"""
        if int(c) == 0:
            # 未删失样本：选择不同标签的样本作为负样本
            neg_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort)
                                  if int(gt_label) != j], dim=0).detach()
            pos_feat = cohort[int(gt_label)][:-1].detach()
        else:
            # 删失样本：根据标签大小选择正负样本
            if int(gt_label) != 0:
                neg_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort)
                                      if int(gt_label) > j], dim=0).detach()
                pos_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort)
                                      if int(gt_label) <= j], dim=0).detach()
            else:
                return 0  # 特殊情况直接返回0

        if neg_feat.shape[0] < 1 or pos_feat.shape[0] < 1:
            return 0

        # 计算对比损失
        neg_dis = indiv_know.squeeze(1) * neg_feat / self.temperature
        pos_dis = indiv_know.squeeze(1) * pos_feat / self.temperature

        inter_loss = -torch.log(
            torch.exp(pos_dis).mean() /
            (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)
        )
        return inter_loss

    def __call__(self, out, gt):
        """
        Args:
            out: 模型输出字典，包含 'cohort' 和 'decompose'
            gt: 真实标签字典，包含 'label' 和 'c'
        """
        loss = 0
        if 'cohort' in out.keys():
            # 解构输出
            indiv, origs = out['decompose']
            cohort, c = out['cohort']

            # 设置设备
            self._setup_device(indiv)

            # 计算内部损失
            intra_loss = self._compute_intra_loss(indiv, origs)

            if c is None:
                return 0

            # 计算样本间对比损失
            indiv_know = indiv.view(4, 1, -1)
            inter_loss = self._compute_inter_loss(indiv_know, cohort, gt['label'], c)

            # 总损失
            loss = intra_loss.mean() + inter_loss

        return loss



###