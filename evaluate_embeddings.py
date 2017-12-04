# Built-in
import random
import argparse

# Libs
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans, AffinityPropagation
import numpy as np

# Custom
import utils

random.seed(7123)


def evaluateEmbeddingsKMeans(feature_data):
  X, y = feature_data['features'], feature_data['labels']
  num_classes = len(set(list(y.flatten())))
  kmeans_preds = KMeans(n_clusters = num_classes, random_state = 7123).fit_predict(X)
  NMI = normalized_mutual_info_score(kmeans_preds, y)
  AMI = adjusted_mutual_info_score(kmeans_preds, y)
  print("KMeans (w/ oracle k) AMI Score: {}".format(AMI))
  print("KMeans (w/ oracle k) NMI Score: {}".format(NMI))

def evaluateEmbeddingsAffinityProp(feature_data):
  X, y = feature_data['features'], feature_data['labels']
  num_classes = len(set(list(y.flatten())))
  afp = AffinityPropagation().fit(X)
  afp_preds = afp.predict(X)
  NMI = normalized_mutual_info_score(afp_preds, y)
  AMI = adjusted_mutual_info_score(afp_preds, y)
  # Interesting metrics to think about: 
  #   http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py
  print("AffinityPropagation AMI Score: {}".format(AMI))
  print("AffinityPropagation NMI Score: {}".format(NMI))
  num_pred_clusters = len(afp.cluster_centers_indices_)
  print("  Estimated {} clusters with {} true classes".format(num_pred_clusters, num_classes))
  # https://github.com/alitouka/spark_dbscan/wiki/Choosing-parameters-of-DBSCAN-algorithm
  # use the min freuqency of classes? for minpts
  # plot the histogram for the nearest neighbor distances?

def evaluateEmbeddingsKNN(feature_data, k = 7):
  X, y = feature_data['features'], feature_data['labels'].reshape(-1, 1)
  nbrs = NearestNeighbors(n_neighbors = k, algorithm='kd_tree').fit(X)
  distances, indices = nbrs.kneighbors(X)
  classes = (y.flatten())[indices]
  classes = classes[:, 1:]
  preds = (classes.squeeze() == y)
  recall_k = [1,3,5,7]
  recall = []
  for k in recall_k:
    r = np.mean(np.any(preds[:, :k], axis = 1).flatten())
    recall.append(r)
    print("Recall@{} : {:.2f}".format(k, r*100))

def setupArgs():
  parser = argparse.ArgumentParser(description='Calculate cluster accuracy of embeddings w/ oracle known number of classes.')
  parser.add_argument('embeddings', help='path to embeddings file')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  feature_data, labels_dict = utils.loadEmbeddings(args.embeddings)
  evaluateEmbeddingsKNN(feature_data)
  evaluateEmbeddingsKMeans(feature_data)
  evaluateEmbeddingsAffinityProp(feature_data)
  
if __name__ == '__main__':
  main()