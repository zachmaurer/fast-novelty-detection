# Built-in
from os import path
import pickle
import random
import time


# Libs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from scipy.special import erf

# Custom
import utils
import constants


random.seed(7123)

def startTimer():
  return time.time()

def endTimer(start, message = ''):
  end = time.time()
  elapsed = end - start
  print("{} operation took: {:.4f}s".format(message, elapsed))


class ADModel:
  """ Abstract Superclass for ADModel  """
  def __init__(self):
    self.seed = 731
    self.n_jobs = 2
    self.classes = None
    self.n_classes = None
    self.ANOMALY = 1
    self.NOT_ANOMALY = 0
    self.verbose = False

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


class OneClassSvmAD(ADModel):
  def __init__(self):
    super().__init__()  
    self.clf = None
    self.scaler = None

    # Model Hyperparams
    self.params = np.arange(0.1, 1, 0.1)
    self.nu = None

  def config(self, hyperparam_tuple):
    self.nu = hyperparam_tuple
  
  def train(self, X, y, verbose = False):
    # Scale features
    self.scaler = StandardScaler()
    self.scaler.fit(X)
    self.clf = OneClassSVM(nu=self.nu, random_state = self.seed, verbose = verbose)
    X_scaled = self.scaler.transform(X)
    self.clf.fit(X_scaled)

  def predict(self, X, **kwargs):
    preds = self.clf.predict(self.scaler.transform(X))
    preds = (-preds + 1) // 2
    return preds


class AffinityAD(ADModel):
  def __init__(self):
    super().__init__()
    self.cluster_centers = None
    self.scaler = StandardScaler()

    # Model Hyperparams
    self.params = np.arange(0.01, 0.26, 0.05)
    self.nu = None

  def config(self, hyperparam_tuple):
    self.nu = hyperparam_tuple

  def train(self, X, y):
    afp = AffinityPropagation().fit(X)
    self.cluster_centers = afp.cluster_centers_ # clusters x features
    dists = euclidean_distances(X, self.cluster_centers)
    X_scaled = self.scaler.fit_transform(dists)
    self.clf = OneClassSVM(nu=self.nu, random_state = self.seed, verbose = 0)
    self.clf.fit(X_scaled)

  def predict(self, X, **kwargs):
    dists = euclidean_distances(X, self.cluster_centers)
    X_scaled = self.scaler.transform(dists)
    preds = self.clf.predict(X_scaled)
    preds = (-preds + 1) // 2
    return preds

class LoOpAD(ADModel):
  # https://github.com/vc1492a/PyNomaly
  def __init__(self, k=8):
    super().__init__()
    self.X = None
    self.neighbors = None
    self.lmbda = 3.0
    self.k = k

    # Model Hyperparams
    self.params = [0.5, 0.6, 0.7]
    self.threshold = None

  def config(self, hyperparam_tuple):
    self.threshold = hyperparam_tuple

  def train(self, X, **kwargs):
    if X is not self.X: 
      self.X = X
      self.neighbors = NearestNeighbors(n_neighbors = self.k).fit(X)
    else:
      print("Skipping kNN fitting, because using same data X.")
    
  def predict(self, X, **kwargs):
    distances, indices = self._queryNearest(X)
    pdists = self._pdist(distances)
    print('indices, distance', indices.shape, distances.shape)
  
    context_distances, _ = self._queryNearest(self.X[indices.flatten(), :], remove_first = True)
    print('context dists', context_distances.shape)
    context_pdists = np.mean(self._pdist(context_distances).reshape(-1, self.k - 1), axis = 1)
    print('context pdists', context_pdists.shape)

    PLOF = pdists / context_pdists - 1
    nPLOF = self.lmbda * np.sqrt(np.mean(PLOF ** 2))
    erf_vec = np.vectorize(erf)
    LoOP = np.maximum(0, erf_vec(PLOF / (nPLOF * np.sqrt(2))))
    print(np.max(LoOP), np.min(LoOP), np.median(LoOP))
    preds = self._hypothesis(LoOP)
    return preds

  def _queryNearest(self, X, remove_first = False):
    start = startTimer()
    distances, indices = self.neighbors.kneighbors(X)
    print("Done querying.")
    if X is self.X or remove_first:
      distances = distances[:, 1:] # exclde first NN, because its itself
      indices = indices[:, 1:]
    else:
      distances = distances[:, :-1] # keep top-(k-1)
      indices = indices[:, :-1]
    endTimer(start, message = 'queryNearest ' + str(X.shape))
    return distances, indices


  def _pdist(self, distances):
    return self.lmbda * np.sqrt(np.mean(distances ** 2, axis = 1))

  def _hypothesis(self, X):
    return X > self.threshold

def testLoopAD():
  X_train = np.random.random((100, 40))
  X_test = np.random.random((45, 40))*123
  lpd = LoOpAD()
  lpd.config(0.5)
  lpd.train(X_train)
  train_preds = lpd.predict(X_train)
  test_preds = lpd.predict(X_test)
  print(np.sum(train_preds))
  print(np.sum(test_preds))

if __name__ == '__main__':
  testLoopAD()


####################################
########### Unused #################
####################################

class LocalOutlierFactorAD(ADModel):
  def __init__(self):
    super().__init__()  
    self.clf = None
    self.scaler = None
    # Model Hyperparams
    thresholds = np.arange(-0.5, 0.5, 0.25)
    contamination = [0.05]
    nn = [20]
    self.params = [(c, n, t) for n in nn for c in contamination for t in thresholds]
    self.contamination = None
    self.n_neighbors = None
    self.threshold = None

  def config(self, hyperparam_tuple):
    contamination, nearest_n, thresh = hyperparam_tuple
    self.contamination = contamination
    self.n_neighbors = nearest_n
    self.threshold = thresh
  
  def train(self, X, y, verbose = False):
    # Scale features
    self.scaler = StandardScaler()
    self.scaler.fit(X)
    self.clf = LocalOutlierFactor(contamination = self.contamination, n_neighbors = self.n_neighbors)
    X_scaled = self.scaler.transform(X)
    self.clf.fit(X_scaled)

  def predict(self, X, **kwargs):
    preds = self.clf._decision_function(self.scaler.transform(X))
    print(preds)
    preds = (preds < self.threshold).astype(np.int32).reshape(-1, 1)
    print(preds)
    return preds


class IsolationForestAD(ADModel):
  def __init__(self):
    super().__init__()  
    self.clf = None
    self.scaler = StandardScaler()
    # Model Hyperparams
    self.params = np.arange(100, 500, 100)
    self.n_estimators = None

  def config(self, hyperparam_tuple):
    self.n_estimators = hyperparam_tuple
  
  def train(self, X, y, verbose = False):
    # Scale features
    self.scaler.fit(X)
    self.clf = IsolationForest(contamination = 0.01, random_state = self.seed, n_jobs = self.n_jobs, \
       verbose = self.verbose, n_estimators = self.n_estimators)
    X_scaled = self.scaler.transform(X)
    self.clf.fit(X_scaled)

  def predict(self, X, **kwargs):
    preds = self.clf.predict(self.scaler.transform(X))
    preds = (-preds + 1) // 2
    return preds


# Cov is singular
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
      X_cov = np.linalg.inv(np.cov(X_class, rowvar = False))
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
    return np.sqrt(np.sum((X - mu) @ cov @ (X - mu).T, axis = 1)).flatten()



