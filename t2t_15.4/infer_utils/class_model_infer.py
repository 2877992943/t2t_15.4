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

r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2tsrc-decoder \
      --data_dir ~/data \
      --problem=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.bin import t2t_trainer
import decoding_v04 as decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

from tensorflow.python.training import saver as saver_mod

import tensorflow as tf
import numpy as np
import time

import util_fea

flags = tf.flags
FLAGS = flags.FLAGS

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")
flags.DEFINE_bool("disable_grappler_optimizations", False,
                  "Disable Grappler if need be to avoid tensor format errors.")
##
flags.DEFINE_string("corpus_dir", None,
                    "Path to the corpus")


def create_hparams():
  hparams_path = None
  if FLAGS.output_dir:
    hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem,
      hparams_path=hparams_path)


def create_decode_hparams():
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = FLAGS.decode_to_file
  decode_hp.decode_reference = FLAGS.decode_reference
  return decode_hp

#
#
class class_model(object):
  def __init__(self):
      self.sess=None
      self.input_token=None
      self.target_token=None

  def init_env(self,

               t2t_usr_dir,
               problem,
               corpus_dir,
               model,
               hp_set,
               output_dir,
               dec_hp="beam_size=4,alpha=0.6",
               data_dir=None
               ):
      tf.logging.set_verbosity(tf.logging.INFO)
      FLAGS.data_dir = './data'#'./data' #随便写
      FLAGS.t2t_usr_dir =t2t_usr_dir #'./'
      FLAGS.problem = problem#'class_seq2seq'
      FLAGS.corpus_dir = corpus_dir#'./corpus'
      ##
      FLAGS.model =model #'transformer'

      FLAGS.hparams_set =hp_set #'transformer_l2'
      FLAGS.output_dir =output_dir #'./model'
      FLAGS.decode_hparams = dec_hp#"beam_size=4,alpha=0.6"
      FLAGS.decode_from_file = 'hasfile'

      ########## main

      tf.logging.set_verbosity(tf.logging.INFO)

      graph = tf.Graph()
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=graph, config=config)
      with graph.as_default():

          trainer_lib.set_random_seed(FLAGS.random_seed)
          usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

          hp = create_hparams()
          decode_hparams = create_decode_hparams()
          decode_hparams.batch_size = 32
          run_config = t2t_trainer.create_run_config(hp)
          if FLAGS.disable_grappler_optimizations:
              run_config.session_config.graph_options.rewrite_options.disable_meta_optimizer = True

          hp.add_hparam("model_dir", run_config.model_dir)

          hparams = hp

          self.input_token, self.target_token = decoding.get_input_target_vocab(hparams)

          ###########
          # placeholder 预测用
          inp_ph = tf.placeholder(shape=[None, None, 1], dtype=tf.int32)
          #target_ph = tf.placeholder(shape=[None, None, 1], dtype=tf.int32)
          #dec_len_ph = tf.placeholder(dtype=tf.int32)
          const=tf.constant(0)
          dec_len_ph=tf.constant(10)
          self.inp_ph=inp_ph
          self.dec_len_ph=dec_len_ph

          features_ph_pred = {'input_space_id':const,
                                  #hparams.problem_hparams.input_space_id,
                         'target_space_id':const,
                                #hparams.problem_hparams.target_space_id,
                        'decode_length': dec_len_ph,
                         'inputs': inp_ph}  # [batch step 1]

          ####### build model
          from tensor2tensor.models import transformer


          p_hparams = hparams.get("problem_hparams")

          model_cls = transformer.TransformerEncoder(hparams=hparams,
                                              # mode=mode,
                                              problem_hparams=p_hparams,
                                              #decode_hparams=decode_hparams
                                                     )

          ####   placeholder 训练用
          # variables=[],call model to make variables[xxxxx]
          featuresxy = util_fea.make_some_fea_ph()  # input target placeholder
          logits, _ = model_cls(featuresxy)
          ### predict mode
          mode_p = tf.estimator.ModeKeys.PREDICT
          model_cls.set_mode(mode_p)
          ###
          top_beams = 1  # 只返回一个
          with tf.variable_scope(tf.get_variable_scope(), reuse=True):
              self.infer_out = model_cls.infer(
                  features_ph_pred,
                  #beam_size=decode_hparams.beam_size,
                  #top_beams=top_beams,
                  #alpha=decode_hparams.alpha,
                  #decode_length=decode_hparams.extra_length,
                  use_tpu=False)
              self.infer_out=self.infer_out['outputs']

          self.sess.run(tf.global_variables_initializer())
          ckpt = saver_mod.get_checkpoint_state(FLAGS.output_dir)
          tf.logging.info("Start to restore the parameters from %s", ckpt.model_checkpoint_path)
          saver = tf.train.import_meta_graph( \
                    ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
          saver.restore(self.sess, ckpt.model_checkpoint_path)

  def infer(self,test):# 'xxxxxxx' 1 input
      inp_arr = decoding.get_input_str2id(self.input_token, test,False)

      feed = {self.inp_ph: inp_arr, self.dec_len_ph: 100}
      output = self.sess.run(fetches=self.infer_out, feed_dict=feed)
      output = decoding.get_output_id2str_cls(self.target_token, output.flatten())
      return output #str

