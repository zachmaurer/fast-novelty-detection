# Built-in
from os import path
import pickle
import random

# Libs
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cosine

# Custom
import utils
import constants


random.seed(7123)

class ADModel:
  """ Abstract Superclass for ADModel  """
  def __init__(self):
    self.seed = 731
    self.n_jobs = 4
    self.classes = None
    self.n_classes = None
    self.ANOMALY = 1
    self.NOT_ANOMALY = 0

  def train(self):
    raise NotImplemented('Implement concrete subclass.')
  
  def predict(self):
    raise NotImplemented('Implement concrete subclass.')

  def _hypothesis(self):
    raise NotImplemented('Implement concrete subclass.')

class SoftmaxThresholdAD(ADModel):
  def __init__(self, threshold = 0.3):
    super().__init__()
    pass

  def train(self, X, y):
    pass

  def predict(self, X):
    pass

  def _hypothesis(self, X):
    pass





class DistanceAD(ADModel):
  """ 
  Concrete subclass for Euclidean Distance based AD
  
  For each of the known classes,
    takes the mean of all embeddings
    creates a single variable normal distirbution
    of the distances from the samples to the mean

  For each test class (anomaly),
    tests whether this embedding is sufficiently
    far from the mean of all clusters
  """
  def __init__(self, threshold = 0.3):
    super().__init__()
    self.distributions = None # class -> (mu, std)
    self.means = None # class -> np.array mean embedding
    self.threshold = threshold # standard deviations away from mean

  def train(self, X, y):
    self.distributions = dict()
    self.means = dict()
    self.classes = sorted(list(set(y)))
    self.n_classes = len(self.classes)
    for c in self.classes:
      mask = y == c
      X_class = X[mask, :]
      X_mean = np.mean(X_class, axis = 0)
      distances = self.distance(X_class, X_mean)
      distribution = norm.fit(distances)
      #print(distribution)
      self.distributions[c] = distribution
      self.means[c] = X_mean

  def predict(self, X, mode = 'all'):
    if self.distributions is None:
      raise ValueError('Must train distributions first')
    sigma_distance = np.empty((X.shape[0], 0), np.float32)
    for c in self.classes:
      mean = self.means[c]
      dists = self.distance(X, mean).reshape(-1, 1)
      mu, sigma = self.distributions[c]
      d = np.abs((dists - mu) / sigma)
      sigma_distance = np.hstack((sigma_distance, d))
    preds = self._hypothesis(sigma_distance, mode)
    return preds

  def _hypothesis(self, X, mode):
    if mode == 'all':
      return np.all(X > self.threshold, axis = 1)
    if mode == 'average':
      return np.mean(X, axis = 1, keepdims = True) > self.threshold

  @staticmethod
  def distance(A, b):
    return np.sum((A - b)**2, axis = 1).flatten()


class CosineAD(DistanceAD):
  """ 
  Concrete subclass for Euclidean Distance based AD
  
  For each of the known classes,
    takes the mean of all embeddings
    creates a single variable normal distirbution
    of the distances from the samples to the mean

  For each test class (anomaly),
    tests whether this embedding is sufficiently
    far from the mean of all clusters
  """
  def __init__(self, threshold = 0.3):
    super().__init__()

  @staticmethod
  def distance(A, b):
    dot_prod = A @ b
    dot_prod /= np.linalg.norm(A, axis = 1) * np.linalg.norm(b)
    return dot_prod



class KMeansAD(ADModel):
  """ Concrete subclass for KMeans based AD """
  def __init__(self):
    super().__init__()
    self.clf = None
    self.centroids = None

  def train(self, X, y):
    k = len(set(y))
    self.clf = KMeans(n_clusters=k, random_state=self.seed, n_jobs=self.n_jobs).fit(X)
    self.centroids = self.clf.cluster_centers_



# Helper functions

def removeTestClassProbs(X_train, y_train, X_test):
  classes = np.array(sorted(list(set(y_train))))
  X_train = X_train[:, classes]
  X_test = X_test[:, classes]
  return X_train, X_test