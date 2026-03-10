from __future__ import print_function

import argparse
import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import _train_val
from utils.utils import get_custom_exp_code
import torch

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
parser.add_argument('--cancer_style',   type=str,
					default='BLCA',
					choices=['BLCA', 'BRCA', 'UCEC', 'GBMLGG', 'LUAD'])
parser.add_argument('--backbone',   type=str,
					default='resnet50_trunc',
					help='WSI features extracted via',)
### Model Parameters.
parser.add_argument('--model_type', type=str,
					default='MCFD', help='method')
parser.add_argument('--fusion', type=str,
					default=None,
					help='Type of fusion: None/concat/bilinear.')
parser.add_argument('--mod', type=str,
					default='path_and_geno',
					help='Type of genomic loader: omic/path_and_geno/coattn.')
parser.add_argument('--apply_sig',		 action='store_true', default=True, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')
parser.add_argument('--seed', 			 type=int, default=7, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')
### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'],
					default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs', type=int,
					default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--n_classes',				 type=int, default=4, help='num_classes')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
					default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
					default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--sample',  type=bool,  default=False, help='Enable sampling (default: False)')
parser.add_argument('--num_patches',  type=int,  default=4096, help='Number of patches (default: 4096)')
### MCFD Parameters
parser.add_argument('--proxy_num', type=int, default=50, help='hyperparameters')
parser.add_argument('--topk_wsi', type=int, default=256, help='hyperparameters')
parser.add_argument('--depth', type=int, default=2, help='hyperparameters')
parser.add_argument('--latent_num', type=int, default=17, help='hyperparameters')
parser.add_argument('--cross_head', type=int, default=4, help='hyperparameters')
parser.add_argument('--latent_heads', type=int, default=4, help='hyperparameters')
parser.add_argument('--alpha', type=int, default=0.1, help="hyperparameters of loss function")
parser.add_argument('--beta', type=int, default=0.01, help="hyperparameters of loss function")
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.device = device

args.fusion = None if args.fusion == 'None' else args.fusion

args.split_dir = f"tcga_{args.cancer_style.lower()}"
### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k,
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'max_epochs': args.max_epochs,
			'results_dir': args.results_dir,
			'lr': args.lr,
			'experiment': args.exp_code,
			'reg': args.reg,
			'label_frac': args.label_frac,
			# 'inst_loss': args.inst_loss,
			'bag_loss': args.bag_loss,
			'bag_weight': args.bag_weight,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size_wsi': args.model_size_wsi,
			'model_size_omic': args.model_size_omic,
			"use_drop_out": args.drop_out,
			'weighted_sample': args.weighted_sample,
			'gc': args.gc,
			'opt': args.opt}
print('\nLoad Dataset')

study = '_'.join(args.task.split('_')[:2])
if study == 'tcga_kirc' or study == 'tcga_kirp':
	combined_study = 'tcga_kidney'
elif study == 'tcga_luad' or study == 'tcga_lusc':
	combined_study = 'tcga_luad'
else:
	combined_study = study
study_dir = '%s_20x_features' % combined_study

args.data_root_dir = '/mnt/data/TCGA/Processed/TCGA-%s/features/%s' % (
		args.cancer_style, args.backbone)

### dataloder
dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study),
									   mode = args.mode,
									   apply_sig = args.apply_sig,
									   # data_dir= os.path.join(args.data_root_dir, study_dir),
									   data_dir= args.data_root_dir,
									   shuffle = False,
									   seed = args.seed,
									   print_info = True,
									   patient_strat= False,
									   n_bins=4,
									   label_col = 'survival_months',
									   ignore=[],
									   sample=args.sample,
									   num_patches=args.num_patches,
									   )

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

args.results_dir = os.path.join(
	args.results_dir, args.model_type,
	f"epoch_{args.max_epochs}_fusion_{args.fusion}",
	str(args.exp_code) + '_s{}'.format(args.seed))

if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
if not os.path.exists(args.split_dir):
	os.makedirs(args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
	print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))


def main(args):
	#### Create Results Directory
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	latest_val_cindex = []
	folds = np.arange(start, end)

	summary_csv_path = os.path.join(args.results_dir, 'summary_latest.csv')

	if not os.path.isfile(summary_csv_path):
		summary_df = pd.DataFrame(columns=['folds', 'val_cindex'])
		summary_df.to_csv(summary_csv_path, index=False)
		print(f"Initialized summary file at {summary_csv_path}")
	else:
		summary_df = pd.read_csv(summary_csv_path)
		print(f"Resuming summary file from {summary_csv_path}")

	latest_val_cindex = summary_df[
		summary_df['folds'] != 'mean±std'
		]['val_cindex'].astype(float).tolist()

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)

		if str(i) in summary_df['folds'].astype(str).values:
			final_model_path = os.path.join(args.results_dir, f"s_{i}_checkpoint.pt")
			if os.path.exists(final_model_path):
				print(f"Skipping Fold {i}, already in summary and model exists.")
				continue
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))

		### Gets the Train + Val Dataset Loader.
		train_dataset, val_dataset = dataset.return_splits(from_id=False,
														   csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
		datasets = (train_dataset, val_dataset)

		### Specify the input dimension size if using genomic features.
		if args.mode in ['omic', 'path_and_geno', 'path_and_geno']:
			args.omic_input_dim = train_dataset.genomic_features.shape[1]
			print("Genomic Dimension", args.omic_input_dim)
		elif 'coattn' in args.mode:
			args.omic_sizes = train_dataset.omic_sizes
			print('Genomic Dimensions', args.omic_sizes)
		else:
			args.omic_input_dim = 0


		print("################# begin train ###################")
		val_latest, cindex_latest = _train_val(datasets, i, args)
		latest_val_cindex.append(cindex_latest)

		save_pkl(results_pkl_path, val_latest)

		new_row = pd.DataFrame({
			'folds': [i],
			'val_cindex': [round(cindex_latest, 4)]
		})

		summary_df = pd.concat([summary_df, new_row], ignore_index=True)
		summary_df.to_csv(summary_csv_path, index=False)

		print(f"Updated summary with fold {i}: c-index={cindex_latest:.4f}")

		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	if len(latest_val_cindex) > 0:
		mean_val = round(np.mean(latest_val_cindex), 4)
		std_val = round(np.std(latest_val_cindex), 4)

		summary_df = summary_df[summary_df['folds'] != 'mean±std']

		summary_row = pd.DataFrame({
			'folds': ['mean±std'],
			'val_cindex': [f'{mean_val}±{std_val}']
		})

		summary_df = pd.concat([summary_df, summary_row], ignore_index=True)
		summary_df.to_csv(summary_csv_path, index=False)

		print(f"Final CV result: {mean_val} ± {std_val}")

if __name__ == "__main__":
	import warnings
	warnings.filterwarnings('ignore', category=FutureWarning)

	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
