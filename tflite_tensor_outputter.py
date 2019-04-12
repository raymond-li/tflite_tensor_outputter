# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# RL: Original script modified to generate intermediate tensor outputs for TensorFlow Lite files.
# Original: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
"""Generates intermediate tensor outputs for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
# import pickle
# import pprint
import os

from PIL import Image

# from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper # for older versions of tensorflow
from tensorflow.lite.python import interpreter as interpreter_wrapper # lite moved from contrib

# flatbuffers generated Python functions. SubGraph::OutputsOffset() was added for below buffer_change_output_tensor_to()
# TODO: Can probably remove most of the other generated functions
import tflite.Model

DEFAULT_INPUT_IMAGE = "input/dog.jpg"
DEFAULT_INPUT_MODEL = "input/inception_v3/inception_v3_quant.tflite"
DEFAULT_INPUT_LABELS = "input/inception_v3/labels.txt"
DEFAULT_OUTPUT_DIR = "output/layer_outputs_dog_quant"

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

# Set subgraph 0's output(s) to new_tensor_i
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
  # Reads model_buffer as a proper flatbuffer file and gets the offset programatically
  # It might be much more efficient if Model.subgraphs[0].outputs[] was set to a list of all the tensor indices.
  fb_model_root = tflite.Model.Model.GetRootAsModel(model_buffer, 0)
  output_tensor_index_offset = fb_model_root.Subgraphs(0).OutputsOffset(0) # Custom added function to return the file offset to this vector
  # print("buffer_change_output_tensor_to. output_tensor_index_offset: ")
  # print(output_tensor_index_offset)
  # output_tensor_index_offset = 0x5ae07e0 # address offset specific to inception_v3.tflite
  # output_tensor_index_offset = 0x16C5A5c # address offset specific to inception_v3_quant.tflite
  # Flatbuffer scalars are stored in little-endian.
  new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
  ])
  # Replace the 4 bytes corresponding to the first output tensor index
  return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

if __name__ == "__main__":
  floating_model = False

  # Command line arguments and default values
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", \
    default=DEFAULT_INPUT_IMAGE, \
    help="image to be classified")
  parser.add_argument("-m", "--model_file", \
    default=DEFAULT_INPUT_MODEL, \
    help=".tflite model to be executed")
  parser.add_argument("-l", "--label_file", \
    default=DEFAULT_INPUT_LABELS, \
    help="name of file containing labels")
  parser.add_argument("-o", "--output_dir", \
    default=DEFAULT_OUTPUT_DIR, \
    help="directory to write the tensor files to")
  parser.add_argument("--input_mean", \
    default=128, \
    help="input_mean")
  parser.add_argument("--input_std", \
    default=128, \
    help="input standard deviation")
  args = parser.parse_args()

  # Read model into model_buffer for output tensor modification
  model_buffer = None
  with open(args.model_file, 'rb') as f:
    model_buffer = f.read()

  # interpreter = interpreter_wrapper.Interpreter(model_path=args.model_file)
  interpreter = interpreter_wrapper.Interpreter(model_content=model_buffer) # Read from buffer instead of file path
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()

  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  # Check the type of the input tensor
  if input_details[0]['dtype'] == np.float32:
    floating_model = True

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image)
  img = img.resize((width, height))

  # Add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    # Convert input data to float
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  print(input_data)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Probably don't need to invoke here.
  print("interpreter.invoke()")
  interpreter.invoke()

  # Generate all intermediate tensor outputs and write to file. Relies on an exception being raised when an invalid output tensor is requested.
  ind = 0
  subdir_name = DEFAULT_OUTPUT_DIR
  os.makedirs(subdir_name, exist_ok=True)
  try:
    while True:
      # Change output tensor
      model_buffer = buffer_change_output_tensor_to(model_buffer, ind)
      interpreter = interpreter_wrapper.Interpreter(model_content=model_buffer)
      interpreter.allocate_tensors()
      interpreter.set_tensor(input_details[0]['index'], input_data)
      # Run inference on the input data up until the output tensor is calculated
      interpreter.invoke()

      # f_tens = open("layer_outputs_dog/{}_tensor.txt".format(ind), "w")
      # Get the tensor data
      dets = interpreter._get_tensor_details(ind)
      tens = interpreter.get_tensor(ind)

      # Write out to files: details and tensor
      print("Writing {}/{}_details_{}.txt".format(subdir_name, ind, dets['name'].replace('/', '-')))
      print("Writing {}/{}_tensor.txt".format(subdir_name, ind))
      f_dets = open("{}/{}_details_{}.txt".format(subdir_name, ind, dets['name'].replace('/', '-')), "w")
      f_dets.write("tensor[{}]: {}\n".format(ind, dets))
      # np.save("layer_outputs_dog/{}_tensor.txt".format(ind), tens)
      tens.tofile("{}/{}_tensor.txt".format(subdir_name, ind), ",")
      # pickle.dump(tens, f)
      # f.write("\n")

      ind += 1
      f_dets.close()
      # f_tens.close()
  except:
    # TODO: Check what exception.
    print("except caught. passing")

  # Revert to the original output tensor and re-run for results
  model_buffer = buffer_change_output_tensor_to(model_buffer, output_details[0]['index'])
  interpreter = interpreter_wrapper.Interpreter(model_content=model_buffer)
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  # for k, v in interpreter.get_tensor_details():
  #   print("tensor[{}]: {}".format(k, v))
  # print(interpreter._interpreter.NumTensors())

  output_data = interpreter.get_tensor(output_details[0]['index'])
  print("output_data:")
  print(output_data)
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  for i in top_k:
    if floating_model:
      print('{0:08.6f}'.format(float(results[i]))+":", labels[i])
    else:
      print('{0:08.6f}'.format(float(results[i]/255.0))+":", labels[i])
