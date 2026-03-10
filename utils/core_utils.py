from argparse import Namespace
from collections import OrderedDict
import os
import pickle
import random
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch

from datasets.dataset_generic import save_splits
from models.model_genomic import SNN
from models.model_MCAT import MCAT_Surv
from models.model_MCFD import MCFD

from utils.utils import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

    def state_dict(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "val_loss_min": self.val_loss_min,
        }

    def load_state_dict(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.early_stop = state["early_stop"]
        self.val_loss_min = state["val_loss_min"]


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)

def _train_val(datasets: tuple, cur: int, args: Namespace):
    # ---> directories & logger
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    # ---> resume checkpoint path (per fold)
    resume_ckpt_path = os.path.join(writer_dir, "resume_checkpoint.pt")
    final_ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    # ---> init dataloaders
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                    weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')



    # ----> init loss function
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----> init model
    # gene
    if args.model_type == 'SNN':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic,
                      'n_classes': args.n_classes}
        model = SNN(**model_dict)
    # patch
    elif args.model_type == 'MCFD':
        model_dict = {
            'omic_input_dim': args.omic_input_dim,
            'wsi_input_dim': args.wsi_encoding_dim,
            'proj_dim': args.proj_dim,
            'num_classes': args.n_classes,
            'proxy_num': args.proxy_num,
            'topk_wsi': args.topk_wsi,
            'depth': args.depth,
            'latent_num': args.latent_num,
            'cross_head': args.cross_head,
            'latent_heads': args.latent_heads,
            'alpha': args.alpha,
            'beta': args.beta,
            'seed': args.seed,
        }
        model = MCFD(**model_dict)

    # multimodal
    elif args.model_type == 'MCAT':
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)

    else:
        raise NotImplementedError

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    if cur == 0:
        print_network(model)

    # ---> lr scheduler
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose=True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    def safe_validation(model, loader, description):
        try:
            print(f"\nAttempting {description}...")
            results_dict, val_cidx = _summary(args, model, loader)
            print(f'{description} Success! Val c-Index: {val_cidx:.4f}')
            return results_dict, val_cidx
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[WARNING] OOM during {description}!")
                print("Skipping validation for this fold, but model is saved.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None, -1.0
            else:
                raise e

    if os.path.isfile(final_ckpt_path):
        print(f"\n==> Fold {cur} model detected at {final_ckpt_path}.")
        print("==> Skipping training, attempting validation only (Recovery Mode).")

        model.load_state_dict(torch.load(final_ckpt_path, map_location=device))

        results_val_dict, val_cindex = safe_validation(model, val_loader, "Recovery Validation")

        if writer: writer.close()
        return results_val_dict, val_cindex

    # ---> resume training
    start_epoch = 0
    best_val_cindex = -1.0

    if os.path.isfile(resume_ckpt_path):
        print(f"\n==> Resume training from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location="cpu")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if early_stopping and checkpoint.get("early_stopping") is not None:
            early_stopping.load_state_dict(checkpoint["early_stopping"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_cindex = checkpoint.get("best_val_cindex", -1.0)
        print(f"==> Resumed at epoch {start_epoch}, current best Val C-Index: {best_val_cindex:.4f}")

    # ---> do train val
    for epoch in range(start_epoch, args.max_epochs):
        # ---> do train
        _train_loop_survival(
            args=args,
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            writer=writer,
            loss_fn=loss_fn, reg_fn=reg_fn,
        )

        if (epoch + 1) % 5 == 0:
            results_val_dict, val_cindex = safe_validation(model, val_loader, f"Epoch {epoch + 1} Validation")

            if val_cindex > best_val_cindex:
                print(
                    f"*** Val c-Index improved from {best_val_cindex:.4f} to {val_cindex:.4f}. Saving best model... ***")
                best_val_cindex = val_cindex
                torch.save(model.state_dict(), final_ckpt_path)
            else:
                print(f"Val c-Index ({val_cindex:.4f}) did not improve from {best_val_cindex:.4f}.")

        # ---> save resume checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "early_stopping": early_stopping.state_dict() if early_stopping else None,
                "best_val_cindex": best_val_cindex
            },
            resume_ckpt_path
        )

    print(f"\nTraining completed for Fold {cur}.")

    if os.path.isfile(final_ckpt_path):
        print(f"Loading best model for Final Validation...")
        model.load_state_dict(torch.load(final_ckpt_path, map_location=device))
    else:
        print(f"No best model found (perhaps training epochs < 5). Saving and using final state...")
        torch.save(model.state_dict(), final_ckpt_path)

    # ---> do final test / validation on the best weights
    results_val_dict, val_cindex = safe_validation(model, val_loader, "Final Best Model Validation")

    print('Final Val c-Index: {:.4f}'.format(val_cindex))
    if writer:
        writer.close()

    return results_val_dict, val_cindex


def _train_loop_survival(args, epoch, model, train_loader, optimizer, writer=None, loss_fn=None, reg_fn=None,
                        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(train_loader)))
    all_censorships = np.zeros((len(train_loader)))
    all_event_times = np.zeros((len(train_loader)))

    for batch_idx, data in enumerate(train_loader):
        data_dict = _unpack_data(args, data, device)
        data_WSI = data_dict['data_WSI']
        data_omic = data_dict['data_omic']
        label = data_dict['label']
        event_time = data_dict['event_time']
        censor = data_dict['censor']

        if args.model_type in ['SNN']:
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI,
                                            x_omic=data_omic)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=censor)
            loss_value = loss.item()
        elif args.model_type in ['MCFD']:
            hazards, S, Y_hat, logits, loss_wsi_IB = model(
                x_path=data_WSI, x_omic=data_omic,
                label=label, censor=censor,
                training = True,
            )
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=censor)
            loss = loss + loss_wsi_IB
            loss_value = loss.item()
        elif args.model_type in ['MCAT']:
            hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic[0], x_omic2=data_omic[1], x_omic3=data_omic[2],
                                         x_omic4=data_omic[3], x_omic5=data_omic[4], x_omic6=data_omic[5])
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=censor)
            loss_value = loss.item()
        else:
            print("model train NotImplementedError")
            raise NotImplementedError


        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * args.lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, censor: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(
                batch_idx,
                 loss_value + loss_reg,
                 label.item(),
                 censor.item(),
                 float(
                     event_time),
                 float(
                     risk),
                 data_WSI.size(
                     0)))

        # backward pass
        loss = loss / args.gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % args.gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(train_loader)
    train_loss /= len(train_loader)

    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv,
                                                                                                 train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

def _summary(args, model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_risk_scores = np.zeros((len(val_loader)))
    all_censorships = np.zeros((len(val_loader)))
    all_event_times = np.zeros((len(val_loader)))

    slide_ids = val_loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, data in enumerate(val_loader):
        data_dict = _unpack_data(args, data, device)
        data_WSI = data_dict['data_WSI']
        data_omic = data_dict['data_omic']
        label = data_dict['label']
        event_time = data_dict['event_time']
        censor = data_dict['censor']
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.model_type in ['SNN']:
                hazards, S, Y_hat, _, _ = model(x_path=data_WSI,
                                                x_omic=data_omic)
            elif args.model_type in ['MCFD']:
                hazards, S, Y_hat, logits, loss_wsi_IB = model(
                    x_path=data_WSI, x_omic=data_omic,
                    label=None, censor=None,
                    training=False,
                )
            elif args.model_type in ['MCAT']:
                hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic[0], x_omic2=data_omic[1],
                                             x_omic3=data_omic[2],
                                             x_omic4=data_omic[3], x_omic5=data_omic[4], x_omic6=data_omic[5])
            else:
                print("model train NotImplementedError")
                raise NotImplementedError

        risk = (-torch.sum(S, dim=1).cpu().numpy()).item()
        event_time = event_time.item()
        censor = censor.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': censor}})

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                         tied_tol=1e-08)[0]

    return patient_results, c_index


def _unpack_data(args, data, device):
    r"""
    Depending on the model type, unpack the data and put it on the correct device

    Args:
        - modality : String
        - device : torch.device
        - data : tuple

    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor

    """

    if args.mode in ["omic"]:
        data_WSI = data[0].to(device)
        mask = None
        data_omic = data[1].to(device)
        label, event_time, censor = data[2], data[3], data[4]
        # clinical_data_list = data[5]

    elif args.mode in ["path_and_geno"]:
        data_WSI = data[0].to(device)
        if args.model_type == 'ProSurv':
            data_omics = data[1].to(device)
            data_omic = data_omics.unsqueeze(0).unsqueeze(0).to(device)
        else:
            data_omic = data[1].to(device)

        label, event_time, censor = data[2], data[3], data[4]

    elif args.mode in ["coattn"]:

        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omic = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        label, event_time, censor = data[7], data[8], data[9]

    else:
        raise ValueError('Unsupported modality:', args.mode)

    label, event_time, censor = label.to(device), event_time, censor.to(device)

    return {
        'data_WSI': data_WSI,
        'data_omic': data_omic,
        'label': label,
        'event_time': event_time,
        'censor': censor,
    }

def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list

