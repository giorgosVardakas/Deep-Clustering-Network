import torch
import argparse
import numpy as np
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import Subset
from datasets.datasets import get_dermatology_dataloader
from typing import Optional
from scipy.optimize import linear_sum_assignment as linear_assignment
import pdb

def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
	"""
	Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
	determine reassignments.

	:param y_true: list of true cluster numbers, an integer array 0-indexed
	:param y_predicted: list of predicted cluster numbers, an integer array 0-indexed
	:param cluster_number: number of clusters, if None then calculated from input
	:return: reassignment dictionary, clustering accuracy
	"""
	if cluster_number is None:
		# assume labels are 0-indexed
		cluster_number = (max(y_predicted.max(), y_true.max()) + 1)
	count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
	for i in range(y_predicted.size):
		count_matrix[y_predicted[i], y_true[i]] += 1

	row_ind, col_ind = linear_assignment(count_matrix.max() - count_matrix)
	reassignment = dict(zip(row_ind, col_ind))
	accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
	return reassignment, accuracy

def transform_clusters_to_labels(clusters, labels):
	# Find the cluster ids (labels)
	c_ids = np.unique(clusters)

	# Dictionary to transform cluster label to real label
	dict_clusters_to_labels = dict()

	# For every cluster find the most frequent data label
	for c_id in c_ids:
		indexes_of_cluster_i = np.where(c_id == clusters)
		elements, frequency = np.unique(labels[indexes_of_cluster_i], return_counts=True)
		true_label_index = np.argmax(frequency)
		true_label = elements[true_label_index]
		dict_clusters_to_labels[c_id] = true_label

	# Change the cluster labels to real labels
	for i, element in enumerate(clusters):
		clusters[i] = dict_clusters_to_labels[element]

	return clusters

def evaluate(model, train_loader):
	labels = []
	clusters = []
	for data, target in train_loader:
		batch_size = data.size()[0]
		data = data.view(batch_size, -1).to(model.device)
		latent_X = model.autoencoder(data, latent=True)
		latent_X = latent_X.detach().cpu().numpy()

		labels.append(target.view(-1, 1).numpy())
		clusters.append(model.clustering.update_assign(latent_X).reshape(-1, 1))

	labels = np.vstack(labels).reshape(-1)
	clusters = np.vstack(clusters).reshape(-1)
	predicted_labels = transform_clusters_to_labels(clusters, labels)

	labels = labels.astype("int")
	ACC = cluster_accuracy(labels, clusters)[1]
	PURITY = accuracy_score(labels, predicted_labels)
	NMI = normalized_mutual_info_score(labels, clusters)
	ARI = adjusted_rand_score(labels, clusters)
	return (ACC, PURITY, NMI, ARI)

def solver(args, model, train_loader):
	rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
	pur_list = []
	acc_list = []
	nmi_list = []
	ari_list = []

	for e in range(args.epoch):
		model.train()
		model.fit(e, train_loader)

		model.eval()
		ACC, PURITY, NMI, ARI = evaluate(model, train_loader)  # evaluation on test_loader
		acc_list.append(ACC)
		pur_list.append(PURITY)
		nmi_list.append(NMI)
		ari_list.append(ARI)

		print('Epoch: {:02d} | ACC: {:.2f} | PURITY: {:.2f} | NMI: {:.2f} | ARI: {:.2f}\n'.format(e, ACC, PURITY, NMI, ARI))

	return rec_loss_list, acc_list, pur_list, nmi_list, ari_list


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Deep Clustering Network')

	# Dataset parameters
	parser.add_argument('--input-dim', type=int, default=34, help='input dimension')

	# Training parameters
	parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 1e-4)')
	parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
	parser.add_argument('--batch-size', type=int, default=61, help='input batch size for training')
	parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
	parser.add_argument('--pre-epoch', type=int, default=300, help='number of pre-train epochs')
	parser.add_argument('--pretrain', type=bool, default=True, help='whether use pre-training')
	
	# Model parameters
	parser.add_argument('--lamda', type=float, default=0.005, help='coefficient of the reconstruction loss')
	parser.add_argument('--beta', type=float, default=1, help='coefficient of the regularization term on clustering')
	parser.add_argument('--hidden-dims', default=[60], help='learning rate (default: 1e-4)')
	parser.add_argument('--latent-dim', type=int, default=5, help='latent space dimension')
	parser.add_argument('--n-clusters', type=int, default=6, help='number of clusters in the latent space')
	parser.add_argument('--clustering', type=str, default='kmeans', help='choose a clustering method (default: kmeans) meanshift, tba')

	# Utility parameters
	parser.add_argument('--n-jobs', type=int, default=1, help='number of jobs to run in parallel')
	parser.add_argument('--device', type=str, default='cuda', help='device for computation (default: cuda)')

	args = parser.parse_args()

	# Load data
	train_loader, datashape = get_dermatology_dataloader(args.batch_size)
	
	# Main body
	model = DCN(args)    
	rec_loss_list, acc_list, pur_list, nmi_list, ari_list = solver(args, model, train_loader)