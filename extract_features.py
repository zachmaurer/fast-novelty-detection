# Built-in
import argparse
from os import path

# Libs
import tensorflow as tf

# Custom
import utils
import constants


def runInference(model_path, input_paths, data_type):
  with tf.Session() as sess:
    # Load model
    utils.loadModel(model_path, path.basename(model_path))
    #utils.printModelLayers()

    # Load data
    if data_type == 'tfr':
      dataset = utils.loadTfRecordDataset(input_paths)      
    else:
      dataset = utils.loadJpegDataset(input_paths)
    dataset = dataset.batch(constants.BATCH_SIZE)
    dataset_iterator = dataset.make_one_shot_iterator()

    # Get layer outputs
    target = 'dukanet_sf.pb/MobilenetV1/Predictions/Reshape:0'
    features_node = sess.graph.get_tensor_by_name(target)
    while True:
      try:
        feed_dict = {'dukanet_sf.pb/input_images:0' : sess.run(dataset_iterator.get_next())}
        features = sess.run(features_node, feed_dict = feed_dict)
        print(features)
      except tf.errors.OutOfRangeError:
        break

def setupArgs():
  def validDataType(data_type):
    if data_type not in ['tfr', 'img', 'jpeg', 'jpg', 'png']: raise ValueError()
    return data_type
  parser = argparse.ArgumentParser(description='Extract activations from trained models.')
  parser.add_argument('model_path', help='path to model file')
  parser.add_argument('data_type', help='one of {img, jpeg, jpg, tfr}', type=validDataType)
  parser.add_argument('input_path', help='path to image files')
  #parser.add_argument('output', help='output path and filename', default=None)
  args = parser.parse_args()
  return args

def main():
  args = setupArgs()
  runInference(args.model_path, args.input_path, args.data_type)

if __name__ == '__main__':
  main()
