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

# Globals
INPUT_LAYER = '{}/input_images:0'

EXTRACT_CONFIG = {
  # model_name -> (dim_size, target_layer, input_dimension)
  'dukanet_sf.pb' : [
      ('dukanet_sf.pb/MobilenetV1/Predictions/Reshape:0', 29, 160),
      ('dukanet_sf.pb/MobilenetV1/Logits/Dropout_1b/Identity:0', 512, 160)
                    ],
  'structured_test.pb' : [
      ('structured_test.pb/MobilenetV1/Logits/SpatialSqueeze:0', 100, 224)
                    ]
}

# Helpers

def saveFeatures(features, labels, model_path, input_path, labels_dict):
  data = ({'features' : features, 'labels' : labels}, labels_dict)
  outfile_name = "{}_{}_data.pkl".format(path.basename(model_path.strip('/')), path.basename(input_path.strip('/')))
  outfile_name = path.join('..', 'embeddings', outfile_name)
  with open(outfile_name, 'wb') as outf:
    pickle.dump(data, outf, protocol=pickle.HIGHEST_PROTOCOL)
  print("Saved {} features with dimension {} and {} labels.".format(features.shape[0], features.shape[1], labels.shape[0]))
  print("Destination: {}".format(outfile_name))

def selectTarget(model_path):
  model = path.basename(model_path.rstrip('/'))
  options = EXTRACT_CONFIG[model]
  print("Select one of the following options [0-{}]:".format(len(options)-1))
  for i, o in enumerate(options):
    print("  [{}] Layer '{}' with dimension {}".format(i, *o))
  while True:
    selection = input('Selection: ')
    try:
      selection = int(selection)
      break
    except ValueError:
      print('Invalid selection, please try again.')
  print("")
  return options[selection]


# Main

def runInference(model_path, input_paths, data_type, args = None):
  model_name = path.basename(model_path)
  target, dim_size, image_size = selectTarget(model_path)
  dataset_iterator, labels_dict = utils.loadImageDataset(input_paths, data_type, image_size)

  with tf.Session() as sess:
    # Load model
    utils.loadModel(model_path, path.basename(model_path), print_layers=False)
    # Get layer outputs
    features_node = sess.graph.get_tensor_by_name(target)
    # Init storage arrays
    features = np.empty((0, dim_size), np.float32)
    labels = np.empty((0,), np.int32)
    n_processed = 0
    # Eval loop
    while True:
      try:
        # Evaluate batch
        X, y = sess.run(dataset_iterator.get_next())
        feed_dict = {INPUT_LAYER.format(model_name) : X}
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
    return features, labels, labels_dict

def setupArgs():
  def validDataType(data_type):
    if data_type not in ['tfr', 'img', 'jpeg', 'jpg', 'png']: raise ValueError()
    return data_type
  parser = argparse.ArgumentParser(description='Extract activations from trained models.')
  parser.add_argument('model_path', help='path to model file')
  parser.add_argument('data_type', help='one of {img, jpeg, jpg, tfr}', type=validDataType)
  parser.add_argument('input_path', help='path to image files')
  args = parser.parse_args()
  print("")
  return args

def main():
  args = setupArgs()
  features, labels, labels_dict = runInference(args.model_path, args.input_path, args.data_type, args)
  saveFeatures(features, labels, args.model_path, args.input_path, labels_dict)


if __name__ == '__main__':
  main()
