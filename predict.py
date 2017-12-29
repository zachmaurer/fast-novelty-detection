# Built-in
import argparse
from os import path

# Libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

#Custom
import model
import utils

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



def evaluateClassifier(train_data, val_data, test_data, clf, predict_kwargs, split_data = False):
  # Unpack the data
  X_train, y_train = train_data['features'], train_data['labels']
  X_val, y_val = val_data['features'], val_data['labels']
  X_test, y_test = test_data['features'], test_data['labels']

  # Print Stats
  print("Embedding dimension: {}".format(X_train.shape[1]))
  print("Total train examples: {}  ({} classes)".format(X_train.shape[0], len(set(list(y_train)))))
  print("Total val examples: {}  ({} classes)".format(X_val.shape[0], len(set(list(y_val)))))
  print("Total test examples: {}  ({} classes)".format(X_test.shape[0], len(set(list(y_test)))))
  print("")

  # Visualization data
  true_positive_history, false_positive_history = [], []

  # Train-eval loop
  for i, config in enumerate(clf.params):
    # Classify
    clf.config(config)
    clf.train(X_train, y_train)
    train_preds = clf.predict(X_val, **predict_kwargs)
    test_preds = clf.predict(X_test, **predict_kwargs)

    # train_preds should all be clf.NOT_ANOMALY == 0; false_positives
    false_positives = sum(train_preds)
    false_positive_rate = false_positives / train_preds.shape[0]
    false_positive_history.append(false_positive_rate)

    # test_preds should all be clf.ANOMALY == 1; true_positives
    true_positives = sum(test_preds)
    true_positive_rate = true_positives / test_preds.shape[0]
    true_positive_history.append(true_positive_rate)

    if clf.verbose or (i+1) % (len(clf.params) / 5) == 0:
      print("Finished evaluating {} of {} classifiers.".format(i+1, len(clf.params)))
      print("  True positive rate: {}".format(true_positive_rate))
      print("  False positive rate: {}".format(false_positive_rate))
  auc_score = metrics.auc(false_positive_history, true_positive_history)
  print("AUC Score: {:.4f}".format(auc_score))
  return false_positive_history, true_positive_history

def excludeTrainClasses(test_data, test_labels, train_labels):
  shared_classes = set(test_labels.keys()) & set(train_labels.keys())
  if len(shared_classes) == 0: return test_data, test_labels
  # If nonempty intersect, remove shared classes
  print("Excluding {} classes from the test data.".format(len(shared_classes)))
  unfiltered_X_test, unfiltered_y_test = test_data['features'], test_data['labels']
  shared_classes_idx = [int(test_labels[c]) for c in shared_classes]
  mask = np.isin(unfiltered_y_test, shared_classes_idx, invert = True)
  filtered_X_test = unfiltered_X_test[mask, :]
  filtered_y_test = unfiltered_y_test[mask]
  removals = unfiltered_X_test.shape[0] - filtered_X_test.shape[0]
  for c in shared_classes:
    del test_labels[c]
  print("Removed {} examples from test data".format(removals))
  return {'features' : filtered_X_test, 'labels' : filtered_y_test}, test_labels

def setupArgs():
  parser = argparse.ArgumentParser(description='Run classification.')
  parser.add_argument('train_data', help='path to embeddings file')
  parser.add_argument('val_data', help='path to embeddings file')
  parser.add_argument('test_data', help='path to embeddings file')
  parser.add_argument('--title', help='path to embeddings file', default = None)
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  train_data, train_labels = utils.loadEmbeddings(args.train_data)
  val_data, val_labels = utils.loadEmbeddings(args.val_data)
  test_data, test_labels = excludeTrainClasses(*utils.loadEmbeddings(args.test_data), train_labels)
  test_data, test_labels = utils.loadEmbeddings(args.test_data)
  clf = model.NearestCentroidSVM()
  clf.verbose = True
  predict_kwargs = {
    'mode' : 'average'
  }
  print("Training {} classifier.".format(clf.__class__))
  false_pos, true_pos = evaluateClassifier(train_data, val_data, test_data, clf, predict_kwargs)
  plotROC(false_pos, true_pos, description = args.title)

if __name__ == '__main__':
  main()

