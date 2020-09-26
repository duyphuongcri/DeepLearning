"""
Implementation of comparison of  UMAP and t-SNE manifold learning algorithms 
( Figure 5.21 in textbook: Computer Vision: Algorithms and Applications 2nd Edition).
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import umap 
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

import pandas as pd 
from chainer.dataset import download as dl
import zipfile
import numpy as np
import tempfile, os, shutil, cv2, re
import pickle

def download_coil20(dataset_type='unprocessed'):
	if dataset_type=='unprocessed':
		url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip"
	elif dataset_type=='processed':
		url = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
	else:
		raise ValueError("dataset_type should be either unprocessed or processed")

	archive_path = dl.cached_download(url)
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()

	cache_root = "./temp/" 
	try:
		os.makedirs(cache_root)
	except OSError:
		if not os.path.isdir(cache_root):
			raise
	cache_path = tempfile.mkdtemp(dir=cache_root)

	data, label = [], []

	try:
		for name in names:
				path = cache_path+name
				if bool(re.search('obj', name)):
					img = cv2.imread(fileOb.extract(name, path=path))
					data.append(img)
					label.append(int(name.split("__")[0].split("/obj")[1]))
	finally:
		shutil.rmtree(cache_root)


	data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
	label = np.array(label).astype(np.uint8)
	return data, label

X_datasets, Y_datasets = [], []
# load data 
# X_coil20, y_coil20 = download_coil20(dataset_type='processed')
# with open("./data/data_coil20.txt", "wb") as data:
#     pickle.dump((X_coil20,y_coil20), data)

with open("./data/data_coil20.txt", "rb") as data:
    X_coil20, y_coil20 = pickle.load(data) 
X_coil20 = X_coil20.reshape((X_coil20.shape[0], -1))
X_datasets.append(X_coil20)
Y_datasets.append(y_coil20)

X_digits, y_digits = datasets.load_digits(n_class=10, return_X_y=True)
X_datasets.append(X_digits)
Y_datasets.append(y_digits)

dft = pd.read_csv('./data/fashion-mnist_test.csv', dtype=int) # read test data
X_fashion = dft.drop('label', axis=1)
y_fashion = dft['label']
X_datasets.append(X_fashion)
Y_datasets.append(y_fashion)

# Set up algorithms
methods = OrderedDict()
methods['umap'] = umap.UMAP()
methods['t-SNE'] = manifold.TSNE(n_components=2, init='pca',
                                 random_state=0)

fig = plt.figure(figsize=(15, 8))
# Plot results
labels = ['COIL20', 'MNIST', 'FASHION MNIST']
for i, (label, method) in enumerate(methods.items()):
    for j in range(len(X_datasets)):
        print(X_datasets[j].shape)
        Y = method.fit_transform(X_datasets[j])
        ax = fig.add_subplot(2, 3, i*3 + j+1)
        ax.scatter(Y[:, 0], Y[:, 1], c=Y_datasets[j], cmap=plt.cm.Spectral)
        ax.set_title(labels[j])
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
plt.show()