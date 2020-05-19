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
class seq2seq_model(object):
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
               data_dir=None,
               addition_hp=None,
               ):
      tf.logging.set_verbosity(tf.logging.INFO)
      FLAGS.data_dir = data_dir#'./data' #随便写
      FLAGS.t2t_usr_dir =t2t_usr_dir #'./'
      FLAGS.problem = problem#'class_seq2seq'
      FLAGS.corpus_dir = corpus_dir#'./corpus'
      if addition_hp:
        FLAGS.hparams = addition_hp #一些其他参数如"batch_size=4096,num_heads=16"
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
          dec_len_ph = tf.placeholder(dtype=tf.int32)
          self.inp_ph=inp_ph
          self.dec_len_ph=dec_len_ph

          features_ph_pred = {'input_space_id': hparams.problem_hparams.input_space_id,
                         'target_space_id': hparams.problem_hparams.target_space_id,
                         'decode_length': dec_len_ph,
                         'inputs': inp_ph}  # [batch step 1]

          ####### build model
          from tensor2tensor.models import transformer


          p_hparams = hparams.get("problem_hparams")

          model_cls = transformer.Transformer(hparams=hparams,
                                              # mode=mode,
                                              problem_hparams=p_hparams,
                                              decode_hparams=decode_hparams)

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
                  beam_size=decode_hparams.beam_size,
                  top_beams=top_beams,
                  alpha=decode_hparams.alpha,
                  decode_length=decode_hparams.extra_length,
                  use_tpu=False)

          self.sess.run(tf.global_variables_initializer())
          ckpt = saver_mod.get_checkpoint_state(FLAGS.output_dir)
          tf.logging.info("Start to restore the parameters from %s", ckpt.model_checkpoint_path)
          saver = tf.train.import_meta_graph( \
                    ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
          saver.restore(self.sess, ckpt.model_checkpoint_path)

  def infer(self,testll):# ["TIME" "," "实" "现"]
      inp_arr = decoding.get_input_str2id(self.input_token, testll)

      feed = {self.inp_ph: inp_arr, self.dec_len_ph: 100}
      output = self.sess.run(fetches=self.infer_out, feed_dict=feed)
      output = decoding.get_output_id2str(self.target_token, output['outputs'])
      return output


if __name__ == "__main__":
    ## init model
    model_inst=seq2seq_model()
    model_inst.init_env(
               t2t_usr_dir='../problem_src',
               problem ='class_seq2seq',
               corpus_dir='../corpus',
               model='transformer',
               hp_set='transformer_l2',
               output_dir='../model')





    inp_json={"ids": "xxxxxxxxxxx",
              "words": [{"val": "0.03亿元", "typ": "m", "sid": 0, "beg": 0, "end": 6},
                        {"val": ",", "typ": "w", "sid": 1, "beg": 6, "end": 7},
                        {"beg": 7, "end": 11, "typ": "trigger", "val": "利润总额",  "sid": 2},
                        {"beg": 11, "end": 17, "typ": "m", "val": "0.15亿元", "sid": 3},
                        {"val": ",", "typ": "w", "sid": 4, "beg": 17, "end": 18},
                        {"beg": 18, "end": 21, "typ": "trigger", "val": "净利润",  "sid": 5},
                        {"beg": 21, "end": 27, "typ": "m", "val": "0.15亿元",  "sid": 6},
                        {"val": "。", "typ": "w", "sid": 7, "beg": 27, "end": 28}],
              "pos": 5, "cls": "trigger"}


    inp_json={"ids": "md1576058554393_100|1177|2", "words": [{"val": "截至", "typ": "v", "sid": 0, "beg": 22, "end": 24}, {"beg": 24, "end": 35, "typ": "TIME", "val": "2015年12月31日", "lab": "Times", "sid": 1}, {"val": "、", "typ": "w", "sid": 2, "beg": 35, "end": 36}, {"beg": 36, "end": 47, "typ": "TIME", "val": "2016年12月31日", "lab": "Times", "sid": 3}, {"val": "和", "typ": "c", "sid": 4, "beg": 47, "end": 48}, {"beg": 48, "end": 59, "typ": "TIME", "val": "2017年12月31日", "lab": "Times", "sid": 5}, {"val": ",", "typ": "w", "sid": 6, "beg": 59, "end": 60}, {"beg": 60, "end": 63, "typ": "company", "val": "发行人", "lab": "Subject", "sid": 7}, {"beg": 63, "end": 71, "typ": "trigger", "val": "货币资金期末余额", "lab": "trigger", "sid": 8}, {"val": "分别", "typ": "d", "sid": 9, "beg": 71, "end": 73}, {"val": "为", "typ": "v", "sid": 10, "beg": 73, "end": 74}, {"beg": 74, "end": 88, "typ": "m", "val": "2,382,194.56万元", "lab": "Value", "sid": 11}, {"val": "、", "typ": "w", "sid": 12, "beg": 88, "end": 89}, {"beg": 89, "end": 103, "typ": "m", "val": "5,176,007.82万元", "lab": "Value", "sid": 13}, {"val": "和", "typ": "c", "sid": 14, "beg": 103, "end": 104}, {"beg": 104, "end": 118, "typ": "m", "val": "5,343,588.51万元", "lab": "Value", "sid": 15}, {"val": ",", "typ": "w", "sid": 16, "beg": 118, "end": 119}, {"val": "占", "typ": "v", "sid": 17, "beg": 119, "end": 120}, {"beg": 120, "end": 124, "typ": "trigger", "val": "资产总额", "lab": "WithIn", "sid": 18}, {"val": "的", "typ": "u", "sid": 19, "beg": 124, "end": 125}, {"val": "比重", "typ": "n", "sid": 20, "beg": 125, "end": 127}, {"val": "分别", "typ": "d", "sid": 21, "beg": 127, "end": 129}, {"val": "为", "typ": "v", "sid": 22, "beg": 129, "end": 130}, {"beg": 130, "end": 136, "typ": "m", "val": "31.47%", "lab": "Ratio", "sid": 23}, {"val": "、", "typ": "w", "sid": 24, "beg": 136, "end": 137}, {"beg": 137, "end": 143, "typ": "m", "val": "31.01%", "lab": "Ratio", "sid": 25}, {"val": "和", "typ": "c", "sid": 26, "beg": 143, "end": 144}, {"beg": 144, "end": 150, "typ": "m", "val": "25.67%", "lab": "Ratio", "sid": 27}, {"val": "。", "typ": "w", "sid": 28, "beg": 150, "end": 151}], "events": [{"trigger": 8, "Times": 1, "Times2": 3, "Times3": 5, "Subject": 7, "Value": 11, "Value2": 13, "Value3": 15, "WithIn": 18, "Ratio": 23, "Ratio2": 25, "Ratio3": 27}], "pos": 8, "cls": "trigger"}

    ##### json 数据转出如下格式
    testll,charId2cutId=util_fea.prepare_data_json2ll(inp_json)
    print (charId2cutId)
    print (testll)

    #test = "TIME , 实 现 triggerRoot mEntity , trigger mEntity 。"
    #testll = test.split(' ')


    #### 预测
    rst=model_inst.infer(testll) #['subject','time','trigger','value','value','u','u'...]

    ### 数据格式
    json_rst=util_fea.make_outputFormat(rst,charId2cutId)
    print (json_rst)


