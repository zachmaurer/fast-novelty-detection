import pickle

import model
import numpy as np
import matplotlib.pyplot as plt


data_file = '../embeddings/dukanet_sf.pb_sm-tf-test_data.pkl'
data = None
with open(data_file, 'rb') as infile:
  data = pickle.load(infile)

X_train, y_train = data['train_examples'], data['train_labels']
X_test, y_test = data['test_examples'], data['test_labels']
#X_train, X_test = model.removeTestClassProbs(X_train, y_train, X_test)


print("Total train examples: {}".format(X_train.shape[0]))
print("Total test examples: {}".format(X_test.shape[0]))

fpr, tpr = [],[]
for t in np.arange(0.01, 3, 0.01):
  clf = model.DistanceAD(threshold = t)
  clf.train(X_train, y_train)

  train_preds = clf.predict(X_train)
  FP = sum(train_preds)
  FPR = 100*sum(train_preds)/train_preds.shape[0]
  print("(Train error, false positive) Incorrect anomalies N={}, {:.2f}%  error".format(FP, FPR))

  test_preds = clf.predict(X_test)
  TP = sum(test_preds)
  TPR = 100*TP/test_preds.shape[0]

  FN = test_preds.shape[0] - TP
  FNR = 100*FN/test_preds.shape[0]
  print("(Test error, false negative) Missed anomalies N={}, {:.2f}%  error".format(FN, FNR))
  print("{},{}".format(TPR/100, FPR/100))
  fpr.append(FPR/100)
  tpr.append(TPR/100)

lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic\n for Normally Distributed Distance from Mean Classifier')
plt.legend(loc="lower right")
plt.show()