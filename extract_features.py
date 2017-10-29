# Built-in
import argparse
from os import path
import random
random.seed(7123)

# Libs
import tensorflow as tf
import numpy as np
import pickle

# Custom
import utils
import constants


DIM_SIZE = 29
target = 'dukanet_sf.pb/MobilenetV1/Predictions/Reshape:0'

target = 'dukanet_sf.pb/MobilenetV1/Logits/Dropout_1b/Identity:0'
DIM_SIZE = 512
#target = 'dukanet_sf.pb/MobilenetV1/Logits/SpatialSqueeze:0'

def splitDataByClass(features, labels, train_split):
  classes = list(set(labels))
  random.shuffle(classes)
  split_idx = int(len(classes)*train_split)
  train_classes = sorted(classes[0:split_idx]) #, sorted(labels[split_idx:])
  train_mask = np.isin(labels, train_classes)
  test_mask = ~train_mask
  data = {
    'train_examples' : features[train_mask, :],
    'train_labels' : labels[train_mask],
    'test_examples' : features[test_mask, :],
    'test_labels' : labels[test_mask]
  }
  print("Split dataset with {:.1f}% train classes (N = {}), {:.1f}% test classes (N = {})" \
    .format(train_split*100, split_idx, (1-train_split)*100, len(classes) - split_idx))
  print("Number of Train Examples: {} \nNumber of Test Examples: {}" \
    .format(len(data['train_labels']), len(data['test_labels'])))
  return data

def saveFeatures(features, labels, model_path, input_path, train_split=0.7):
  data = splitDataByClass(features, labels, train_split)
  outfile_name = "{}_{}_data.pkl".format(path.basename(model_path), path.basename(input_path))
  with open(path.join('..', 'embeddings', outfile_name), 'wb') as outf:
    pickle.dump(data, outf, protocol=pickle.HIGHEST_PROTOCOL)
  print("Saved {} features with dimension {} and {} labels.".format(features.shape[0], features.shape[1], labels.shape[0]))

def runInference(model_path, input_paths, data_type):
  with tf.Session() as sess:
    # Load model
    utils.loadModel(model_path, path.basename(model_path), print_layers=False)
    dataset_iterator = utils.loadData(input_paths, data_type)

    # Get layer outputs
    features_node = sess.graph.get_tensor_by_name(target)
    
    # Init storage arrays
    features = np.empty((0,DIM_SIZE), np.float32)
    labels = np.empty((0,), np.int32)
    n_processed = 0

    # Eval loop
    while True:
      try:
        # Evaluate batch
        X, y = sess.run(dataset_iterator.get_next())
        feed_dict = {'dukanet_sf.pb/input_images:0' : X}
        target_output = sess.run(features_node, feed_dict = feed_dict)
        target_output = np.squeeze(target_output)
        # Stack outputs
        features = np.vstack((features, target_output))
        labels = np.hstack((labels, y))
        
        # Logging
        n_processed += constants.BATCH_SIZE
        if n_processed % 10*constants.BATCH_SIZE == 0: 
          print("Processed {} records.".format(n_processed))
      except tf.errors.OutOfRangeError:
        break
    print("Completed extracting features. \n")
    return features, labels

def setupArgs():
  def validDataType(data_type):
    if data_type not in ['tfr', 'img', 'jpeg', 'jpg', 'png']: raise ValueError()
    return data_type
  parser = argparse.ArgumentParser(description='Extract activations from trained models.')
  parser.add_argument('model_path', help='path to model file')
  parser.add_argument('data_type', help='one of {img, jpeg, jpg, tfr}', type=validDataType)
  parser.add_argument('input_path', help='path to image files')
  args = parser.parse_args()
  return args

def main():
  args = setupArgs()
  features, labels = runInference(args.model_path, args.input_path, args.data_type)
  saveFeatures(features, labels, args.model_path, args.input_path)


if __name__ == '__main__':
  main()
