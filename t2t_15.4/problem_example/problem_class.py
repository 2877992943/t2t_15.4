# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""IMDB Sentiment Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import json
import tensorflow.compat.v1 as tf
labDic={"other": 0, "Value": 1,
        "NegativeRatio": 2, "DecreaseValue": 3,
        "IncreaseValue": 4, "PositiveRatio": 5, "Ratio": 6}


datapath='./corpus'

class encoder_abstr(object):
  def __init__(self):
    reader = open(os.path.join(datapath, 'vocabx.txt'))
    self.str2id = {}
    self.id2str = {}
    for line in reader.readlines():
      line = line.strip()
      if line:
        if line in self.str2id: continue
        self.str2id[line] = len(self.str2id)

    ###
    self.id2str = dict(zip(list(self.str2id.values()), list(self.str2id.keys())))
  def encode(self,s):
    ll=[]
    for w in s:
      ll.append(self.str2id[w])
    return ll

  def decode(self,ids):
    for id in ids:
      ll.append(self.id2str[id])
    return ll

  @property
  def vocab_size(self):
    return len(self.str2id)




@registry.register_problem
class class_yr(text_problems.Text2ClassProblem):
  """IMDB sentiment classification."""
  #URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  @property
  def num_classes(self):
    return 7

  def class_labels(self, data_dir):
    del data_dir
    return ["other", "Value",
        "NegativeRatio", "DecreaseValue",
        "IncreaseValue", "PositiveRatio", "Ratio"]

  def doc_generator(self, imdb_dir, dataset, include_label=False):
    dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
        imdb_dir, dataset, "neg"), False)]

    for d, label in dirs:
      for filename in os.listdir(d):
        with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
          doc = imdb_f.read().strip()
          if include_label:
            yield doc, label
          else:
            yield doc


  def line_generator(self,fname):
    fpath=os.path.join(datapath,fname)
    with tf.gfile.Open(fpath) as imdb_f:
      for line in imdb_f.readlines():
        line=line.strip()
        if line:
          xll,y=json.loads(line)
          yield xll,y

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    # """Generate examples."""
    # # Download and extract
    # compressed_filename = os.path.basename(self.URL)
    # download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
    #                                                self.URL)
    # imdb_dir = os.path.join(tmp_dir, "aclImdb")
    # if not tf.gfile.Exists(imdb_dir):
    #   with tarfile.open(download_path, "r:gz") as tar:
    #     tar.extractall(tmp_dir)
    #
    # # Generate examples
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset = "train.json" if train else "test.json"
    #for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
    for line, label in self.line_generator(dataset):
      yield {
          "inputs": line,
          "label": labDic[label],
      }

  #############
  # customized
  # def generate_text_for_vocab(self, data_dir, tmp_dir):
  #   for i, sample in enumerate(
  #       self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
  #     yield sample["inputs"]
      # if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
      #   break

  def get_or_create_vocab(self,data_dir=None, tmp_dir=None,force_get=True):
    self.encoder=encoder_abstr()
    return self.encoder



  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      #inputs.append(text_encoder.EOS_ID)
      label = sample["label"]
      yield {"inputs": inputs, "targets": [label]} #idlist

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

    return {
      "inputs": encoder,
      "targets": text_encoder.ClassLabelEncoder(self.class_labels(data_dir))
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.SYMBOL,
                  "targets": modalities.ModalityType.CLASS_LABEL}
    p.vocab_size = {#"inputs": self._encoders["inputs"].vocab_size,
                    "inputs":self.encoder.vocab_size,
                    "targets": self.num_classes}


