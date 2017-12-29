import utils
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib

#import seaborn as sns
import numpy as np

def makeProjection(train_feature_data, test_feature_data):
  Xtrain, ytrain = train_feature_data['features'], train_feature_data['labels'].flatten()
  Xtest, ytest = test_feature_data['features'], test_feature_data['labels'].flatten()

  ytrain[:] = 0
  ytest[:] = 1

  X = np.vstack((Xtrain, Xtest))
  y = np.hstack((ytrain, ytest))

  (fig, subplots) = plt.subplots(2, 2, figsize=(15, 15))
  plt.title('TSNE')
  perplexities = [5, 30, 50, 100]
  plots = [(0,0), (0,1), (1,0), (1,1)]
  for i, p in enumerate(perplexities):
    print("Fitting TSNE...w/ {} perplexity".format(p))
    projection = TSNE(verbose = 1, perplexity= p).fit_transform(X)
    print("Done fitting TSNE.")
    row, col = plots[i]
    ax = subplots[row][col]
    cmap=matplotlib.colors.ListedColormap(['blue', 'orange'])
    ax.scatter(projection[:, 0], projection[:, 1], c=y, cmap=cmap, alpha = 0.1, s = 5)
    ax.set_title("Perplexity=%d" % p)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    # Visualizing TSNE
  plt.show()

def setupArgs():
  parser = argparse.ArgumentParser(description='Calculate cluster accuracy of embeddings w/ oracle known number of classes.')
  parser.add_argument('train', help='path to embeddings file')
  parser.add_argument('test', help='path to embeddings file')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  train_feature_data, train_labels_dict = utils.loadEmbeddings(args.train)
  test_feature_data, test_labels_dict = utils.loadEmbeddings(args.test)
  makeProjection(train_feature_data, test_feature_data)




if __name__ == '__main__':
  main()