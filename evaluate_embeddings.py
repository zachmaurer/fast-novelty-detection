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
  X, y = feature_data['features'], feature_data['labels'].flatten()
  split_idx = X.shape[0] // 2
  X_train = X[0:split_idx, :]
  X_test, y_test = X[split_idx:, :], y[split_idx:]
  num_classes = len(set(list(y)))
  nbrs = NearestNeighbors(n_neighbors = k, algorithm='ball_tree').fit(X_train)
  distances, indices = nbrs.kneighbors(X_test)
  classes = y_test[indices]
  classes = np.hstack((classes, (num_classes-1)*np.ones((classes.shape[0],1))))
  classes = classes.astype(np.int32)
  freqs = np.apply_along_axis(np.bincount, axis = 1, arr = classes)
  freqs[:, -1] -= 1
  predicted_classes = np.argmax(freqs, axis = 1)
  accuracy = np.sum(predicted_classes == y_test) / predicted_classes.shape[0]
  print("KNN Derived Accuracy (k = {}): {:.4f}".format(k, accuracy))

def setupArgs():
  parser = argparse.ArgumentParser(description='Calculate cluster accuracy of embeddings w/ oracle known number of classes.')
  parser.add_argument('embeddings', help='path to embeddings file')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  feature_data, labels_dict = utils.loadEmbeddings(args.embeddings)
  for k in range(4, 8):
    evaluateEmbeddingsKNN(feature_data, k = k)
  evaluateEmbeddingsKMeans(feature_data)
  evaluateEmbeddingsAffinityProp(feature_data)
  
if __name__ == '__main__':
  main()