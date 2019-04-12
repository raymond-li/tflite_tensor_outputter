# TFLite Tensor Outputter Script

## Prerequisites
 - Python 3.6
 - Tensorflow
 - Flatbuffers

## Usage
Generating the Python Flatbuffer code for the FlatBuffers Schema (tflite.fbs):
```sh
flatc --python tflite.fbs
```

Running the tool to generate files for all the tensors in the model during inference of the input image:
```sh
# Using default arguments specified at the beginning of the file
python3.6 tflite_tensor_outputter.py

# Using command line
python3.6 tflite_tensor_outputter.py --image input/dog.jpg --model_file input/inception_v3/inception_v3_quant.tflite --label_file input/inception_v3/labels.txt --output_dir output/layer_outputs_dog_quant/
```