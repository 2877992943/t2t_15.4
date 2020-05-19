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

"""lm Problem."""

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
#import tensorflow.compat.v1 as tf


import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
datapath='./corpus'
#datapath=FLAGS.corpus_dir



class encoder_abstr(object):
  def __init__(self,dictpath='vocabx.txt'):
    reader = open(os.path.join(datapath, dictpath))
    self.str2id = {}
    self.id2str = {}
    for line in reader.readlines():
      line = line.strip()
      if line:
        if line in self.str2id: continue
        self.str2id[line] = len(self.str2id)

    ###
    self.id2str = dict(zip(list(self.str2id.values()), list(self.str2id.keys())))
  def encode(self,s): # list input
    ll=[]
    for w in s:
      w=w if w in self.str2id else 'UNK'
      ll.append(self.str2id[w])


    return ll

  def decode(self,ids):
    ll=[]
    for id in ids:
      id = id if id in self.id2str else 2
      ll.append(self.id2str[id])
    return ll

  @property
  def vocab_size(self):
    return len(self.str2id)



@registry.register_problem
class class_lm_douban(text_problems.Text2SelfProblem):
  @property
  def is_generate_per_split(self):
    return True



  def line_generator(self,fname): #
    fpath = os.path.join(datapath, fname)
    with tf.gfile.Open(fpath) as imdb_f:
        for line in imdb_f.readlines():
          line=line.strip()
          if line:
            yield line
  # def line_generator(self,fname):
  #   fpath=os.path.join(datapath,fname)
  #   with tf.gfile.Open(fpath) as imdb_f:
  #     for line in imdb_f.readlines():
  #       line=line.strip()
  #       if line:
  #         wll=json.loads(line)
  #         xll=[w['char'] for w in wll]
  #         yll=[w['y'] for w in wll]
  #         yield xll,yll

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
    dataset = "train.txt" if train else "test.txt"
    #for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
    for line in self.line_generator(dataset):
      yield {
          #"input": xll, #wordll
          "target": line,
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
    #self.input_encoder=encoder_abstr('vocabx.txt')
    self.target_encoder=encoder_abstr('vocab.txt')
    return self.target_encoder



  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:

      target = encoder.encode(sample["target"])+[1] # EOS

      print (' '.join(sample["target"]))
      yield {"targets": target}

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
    encoders = {"targets": encoder}
    if self.has_inputs:
      encoders["inputs"] = encoder
    return encoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)

    p.modality = {"targets": modalities.ModalityType.SYMBOL}
    p.vocab_size = {"targets": self.target_encoder.vocab_size}

    if self.has_inputs:
      p.modality["inputs"] = modalities.ModalityType.SYMBOL
      p.vocab_size["inputs"] = self.input_encoder.vocab_size

    # p.modality = {"inputs": modalities.ModalityType.SYMBOL,
    #               "targets": modalities.ModalityType.SYMBOL}
    # p.vocab_size = {#"inputs": self._encoders["inputs"].vocab_size,
    #                 "inputs":self.input_encoder.vocab_size,
    #                 "targets": self.input_encoder.vocab_size}  #  应该是target_encoder


