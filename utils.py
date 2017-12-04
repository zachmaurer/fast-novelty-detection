# Built-in
import os
from os import path
import pickle

# Libs
import cv2
import numpy as np
import tensorflow as tf

# Custom
import constants

# -------------------------------------------


#######################
### Embedding Utils ###
#######################

def loadEmbeddings(data_path):
  with open(data_path, 'rb') as infile:
    data, labels = pickle.load(infile)
  print("Loaded embeddings: {}".format(path.basename(data_path.rstrip('/'))))
  return data, labels

# -------------------------------------------


###################
### Model Utils ###
###################

def loadModel(model_path, name, print_layers=False):
  print("\n\nLoading model: {}".format(name))
  with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name=name)
  if print_layers:
    printModelLayers()


def printModelLayers():
  for n in tf.get_default_graph().as_graph_def().node:
    print(n.name)

# -------------------------------------------


#####################
### Dataset Utils ###
#####################

def loadImageDataset(input_paths, data_type, image_size):
    if data_type == 'tfr':
      dataset = loadTfRecordDataset(input_paths, image_size, label_fn = True)      
    else:
      dataset = loadJpegDataset(input_paths, image_size)
    dataset = dataset.batch(constants.BATCH_SIZE)
    dataset_iterator = dataset.make_one_shot_iterator()
    with open(path.join(input_paths, 'labels.txt')) as labels_file:
      labels_dict = {}
      for l in labels_file:
        if len(l.strip()) > 0:
          label_idx, label_name  = l.strip().split(':')
          labels_dict[label_name] = int(label_idx)
    return dataset_iterator, labels_dict

def loadTfRecordDataset(image_paths, image_size, label_fn = False):
  """ Loads image dataset from a list of directories containing TFRecord files. """
  return loadDataset(image_paths, tf.contrib.data.TFRecordDataset, openTfRecordImage, label_fn, image_size)

def loadJpegDataset(image_paths, image_size, label_fn = None):
  """ Loads image dataset from a list of directories containing image files.  """
  return loadDataset(image_paths, tf.contrib.data.Dataset.from_tensor_slices, openImageTf, label_fn, image_size)

def loadDataset(image_paths, tf_dataset, file_opener, label_fn, image_size, excluded_classes = {}):
  """
    Generic dataset loader. 
    @tf_dataset : the tf.Dataset class
    @file_opener : function to open that type of datas
    @label_fn : Given path/fname, returns an int label. 
      If None, returns a dataset with no labels.
      If True/False, used ONLY to determine whether to return labels from a TFRecord Example.
  """
  if isinstance(image_paths, str): 
    image_paths = [image_paths]
  filenames = []
  for p in image_paths:
    filenames += [path.join(p, x)  for x in os.listdir(p) if not x.startswith('.') and not x.endswith('.txt')]
  input_imgs = tf.constant(filenames)
  if label_fn is not None and not isinstance(label_fn, bool):
    labels = tf.constant([label_fn(x) for x in filenames])
    dataset = tf_dataset((input_imgs, labels))
    dataset = dataset.map(lambda x, y: (file_opener(x, image_size = image_size), y))
  else:
    dataset = tf_dataset((input_imgs))
    dataset = dataset.map(lambda x: file_opener(x, return_labels = label_fn, image_size = image_size))
  return dataset

# -------------------------------------------


##################
### File Utils ###
##################

def openTfRecordImage(example_proto, image_size = constants.IMAGE_SIZE, **kwargs):
  return_labels = kwargs.pop('return_labels')
  if return_labels:
    features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
                'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image/encoded'] = tf.image.decode_jpeg(parsed_features['image/encoded'], channels = 3)
    parsed_features['image/encoded'] = tf.image.resize_images(parsed_features['image/encoded'], \
                                                                [image_size, image_size])
    return parsed_features['image/encoded'], parsed_features['image/class/label']
  else:
    features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
            }
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image/encoded'] = tf.image.decode_jpeg(parsed_features['image/encoded'], channels = 3)
    parsed_features['image/encoded'] = tf.image.resize_images(parsed_features['image/encoded'], \
                                                                [image_size, image_size])
    return parsed_features['image/encoded']


def openImageTf(image_path, image_size = constants.IMAGE_SIZE, **kwargs):
  """ Returns a uint8 Tensor w/ shape [height, width, channels] """
  image_string = tf.read_file(image_path)
  # decode_image doesn't work? -- https://github.com/tensorflow/tensorflow/issues/8551
  image_decoded = tf.image.decode_jpeg(image_string) 
  image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
  return image_resized

  # -------------------------------------------

  ################
  ### Unused  ###
  ################

def openImageCv(image_path, image_size = constants.IMAGE_SIZE):
  cv_img = cv2.imread(image_path)
  cv_img = cv2.resize(cv_img, (image_size, image_size))
  cv_img = np.expand_dims(cv_img, 0)
  return cv_img

def BGRtoRGB(img):
  img = img[...,::-1]
  return img

def normalizeRGB(img):
  return (img.astype(np.float32)/255.0-0.5)*2.0

# def splitDataByClass(features, labels, train_split):
#   classes = list(set(labels))
#   random.shuffle(classes)
#   split_idx = int(len(classes)*train_split)
#   train_classes = sorted(classes[0:split_idx]) #, sorted(labels[split_idx:])
#   train_mask = np.isin(labels, train_classes)
#   test_mask = ~train_mask
#   data = {
#     'train_examples' : features[train_mask, :],
#     'train_labels' : labels[train_mask],
#     'test_examples' : features[test_mask, :],
#     'test_labels' : labels[test_mask]
#   }
#   print(data['train_examples'].shape)
#   print("Split dataset with {:.1f}% train classes (N = {}), {:.1f}% test classes (N = {})" \
#     .format(train_split*100, split_idx, (1-train_split)*100, len(classes) - split_idx))
#   print("Number of Train Examples: {} \nNumber of Test Examples: {}" \
#     .format(len(data['train_labels']), len(data['test_labels'])))
#   return data

# def removeTestClassProbs(X_train, y_train, X_test):
#   classes = np.array(sorted(list(set(y_train))))
#   X_train = X_train[:, classes]
#   X_test = X_test[:, classes]
#   return X_train, X_test
