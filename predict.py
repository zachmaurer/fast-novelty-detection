import pickle
import argparse
from os import path

import model
import numpy as np
import matplotlib.pyplot as plt

data_file = '../embeddings/dukanet_sf.pb_sm-tf-test_data.pkl'

def plotROC(false_positive_rates, true_positive_rates, description = None):
  lw = 2
  plt.figure()
  plt.plot(false_positive_rates, true_positive_rates, color='darkorange', lw=lw, label='ROC curve')
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic\n {}'.format(description or ""))
  plt.legend(loc="lower right")
  plt.show()

def loadData(data_path):
  with open(data_file, 'rb') as infile:
    data = pickle.load(infile)
  print("Loaded embeddings: {}".format(path.basename(data_path.rstrip('/'))))
  return data

def evaluateClassifier(data):
  X_train, y_train = data['train_examples'], data['train_labels']
  X_test, y_test = data['test_examples'], data['test_labels']
  #X_train, X_test = model.removeTestClassProbs(X_train, y_train, X_test)

  print("Embedding dimension: {}".format(X_train.shape[1]))
  print("Total train examples: {}  ({} classes)".format(X_train.shape[0], len(set(list(y_train)))))
  print("Total test examples: {}  ({} classes)".format(X_test.shape[0], len(set(list(y_test)))))
  print("")

  fpr, tpr = [],[]
  params = np.arange(0.01, 3, 0.01)
  for i, t in enumerate(params):
    clf = model.DistanceAD(threshold = t)
    clf.train(X_train, y_train)

    train_preds = clf.predict(X_train, mode='average')
    FP = sum(train_preds)
    FPR = 100*sum(train_preds)/train_preds.shape[0]
    #print("(Train error, false positive) Incorrect anomalies N={}, {:.2f}%  error".format(FP, FPR))

    test_preds = clf.predict(X_test, mode='average')
    TP = sum(test_preds)
    TPR = 100*TP/test_preds.shape[0]

    FN = test_preds.shape[0] - TP
    FNR = 100*FN/test_preds.shape[0]
    #print("(Test error, false negative) Missed anomalies N={}, {:.2f}%  error".format(FN, FNR))
    #print("{},{}".format(TPR/100, FPR/100))
    fpr.append(FPR/100)
    tpr.append(TPR/100)
    if (i+1) % 100 == 0:
      print("Finished evaluating {} of {} classifiers.".format(i+1, len(params)))
  return fpr, tpr

def setupArgs():
  parser = argparse.ArgumentParser(description='Run classification.')
  parser.add_argument('data_path', help='path to embeddings file')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  data = loadData(args.data_path)
  fpr, tpr = evaluateClassifier(data)
  plotROC(fpr, tpr)

if __name__ == '__main__':
  main()

