import numpy as np

from sklearn.cluster import KMeans, SpectralBiclustering
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.manifold import TSNE
from .data import load_train_test

x_train, x_test, y_train, y_test = load_train_test()

tsne = TSNE(n_components=2, init='random',random_state=0)
x_train = tsne.fit_transform(x_train)


kmeans = KMeans(n_clusters=2,random_state=0)
cluster = kmeans.fit(x_train)

labels = np.zeros_like(cluster)

for i in range(2):
    mask = (cluster == i)
    labels[mask] = mode(y_train[mask])[0]

print(accuracy_score(y_train),labels)


# mat = confusion_matrix(digits.target, labels)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=digits.target_names,
#             yticklabels=digits.target_names)

# plt.xlabel('true label')
# plt.ylabel('predicted label')