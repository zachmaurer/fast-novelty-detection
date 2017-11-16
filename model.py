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

  def config(self, hyperparam_tuple):
    raise NotImplemented('Implement concrete subclass.')

  def train(self):
    raise NotImplemented('Implement concrete subclass.')
  
  def predict(self, **kwargs):
    raise NotImplemented('Implement concrete subclass.')

  def _hypothesis(self):
    raise NotImplemented('Implement concrete subclass.')

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
  def __init__(self):
    super().__init__()
    self.distributions = None # class -> (mu, std)
    self.means = None # class -> np.array mean embedding

    # Model Hyperparams
    self.params = np.arange(0.01, 3, 0.01)
    self.threshold = None

  def config(self, hyperparam_tuple):
    self.threshold = hyperparam_tuple

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
      self.distributions[c] = distribution
      self.means[c] = X_mean

  def predict(self, X, **kwargs):
    mode = kwargs.pop('mode')
    if mode is None:
      raise ValueError('DistanceAD requires a prediction `mode`.')
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
  Concrete subclass for Cosine Distance based AD
  """
  def __init__(self):
    super().__init__()

  @staticmethod
  def distance(A, b):
    cos_sim = A @ b
    cos_sim /= np.linalg.norm(A, axis = 1) * np.linalg.norm(b)
    cos_dist = 1.0 - cos_sim
    return cos_dist

class MahalanobisAD(ADModel):
  """ 
  Concrete subclass for Mahalanbois Distance based AD
  """
  def __init__(self):
    super().__init__()
    self.distributions = None # class -> (mu, cov)
    # Model Hyperparams
    self.params = np.arange(0.01, 3, 0.01)
    self.threshold = None

  def config(self, hyperparam_tuple):
    self.threshold = hyperparam_tuple

  def train(self, X, y):
    self.distributions = dict()
    self.classes = sorted(list(set(y)))
    self.n_classes = len(self.classes)
    for c in self.classes:
      mask = y == c
      X_class = X[mask, :]
      print(X_class.shape)
      X_mean = np.mean(X_class, axis = 0)
      X_cov = np.cov(X_class, rowvar = False)
      self.distributions[c] = (X_mean, X_cov)

  def predict(self, X, **kwargs):
    mode = kwargs.pop('mode')
    if mode is None: raise ValueError('MahalanobisAD requires a prediction `mode`.')
    if self.distributions is None: raise ValueError('Must train distributions first')
    z_scores = np.empty((X.shape[0], 0), np.float32)
    for c in self.classes:
      mu, cov = self.distributions[c]
      dists = self.distance(X, mu, cov).reshape(-1, 1)
      z_scores = np.hstack((z_scores, dists))
    preds = self._hypothesis(z_scores, mode)
    return preds

  def _hypothesis(self, X, mode):
    if mode == 'all':
      return np.all(X > self.threshold, axis = 1)
    if mode == 'average':
      return np.mean(X, axis = 1, keepdims = True) > self.threshold

  @staticmethod
  def distance(X, mu, cov):
    if np.isclose(np.linalg.det(cov), 0.0):
      raise ValueError("Covariance matrix is singular!")
      print(X.shape[0])
    return np.sqrt(np.sum((X - mu) @ np.linalg.pinv(cov) @ (X - mu).T, axis = 1)).flatten()



