# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import cv2


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def classifyImage(image):
    file_name = image
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    # input_height, input_width = image.shape
    size = 224
    input_height = size
    input_width = size
    input_mean = size / 2
    input_std = size / 2
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)

    image = cv2.resize(image, dsize=(size, size),
                       interpolation=cv2.INTER_CUBIC)
    cv2.imshow('input', image)
    # input('wait')

    image = np.array(image).reshape(1, size, size, 3)

    t = image  # np.array(image)[:, :, 0:3]
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end = time.time()

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    eval_time = end - start
    index_max = np.argmax(results)

    return eval_time, labels[index_max], results[index_max]
