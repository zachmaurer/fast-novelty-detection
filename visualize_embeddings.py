import utils
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

#import seaborn as sns
#import numpy as np

def makeProjection(feature_data):
  X, y = feature_data['features'], feature_data['labels'].flatten()
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
    ax.scatter(projection[:, 0], projection[:, 1], c=y, cmap=plt.cm.viridis)
    ax.set_title("Perplexity=%d" % p)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    # Visualizing TSNE
  plt.show()

def setupArgs():
  parser = argparse.ArgumentParser(description='Calculate cluster accuracy of embeddings w/ oracle known number of classes.')
  parser.add_argument('embeddings', help='path to embeddings file')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  feature_data, labels_dict = utils.loadEmbeddings(args.embeddings)
  makeProjection(feature_data)




if __name__ == '__main__':
  main()