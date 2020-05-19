from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10
event_mention_typ={'m','TIME','trigger','WithIn','company','ORG'}

def make_some_fea1(has_input=True):
  inputs = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")
  return features



def make_some_fea11(has_input=True):
  inputs = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1))
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")
  return features



def make_some_fea_ph(has_input=True):
  inputs = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1))
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")
  ###
  features['inputs']=tf.placeholder(shape=[None,None,1],dtype=tf.int32)
  features['targets']=tf.placeholder(shape=[None,None,1],dtype=tf.int32)
  return features


def make_cls_infer_ph(): #分类 预测 用
    const = tf.constant(0)
    dec_len_ph = tf.constant(10)
    inp_ph = tf.placeholder(shape=[None, None, 1], dtype=tf.int32)
    features_ph_pred = {'input_space_id': const,
                        # hparams.problem_hparams.input_space_id,
                        'target_space_id': const,
                        # hparams.problem_hparams.target_space_id,
                        'decode_length': dec_len_ph,
                        'inputs': inp_ph}



def prepare_data_json2ll(d):
    rootPos = d['pos'] # root position
    ### get rawsent
    wordll = [w['typ'] if w['typ'] in event_mention_typ else w['val'] for w in d['words']]
    wordll = ['mEntity' if w == 'm' else w for w in wordll]
    ### trigger ->trigger1 trigger2
    # wordll = number_the_triggers(wordll)
    whether_event_mention = [True if w['typ'] in event_mention_typ else False for w in d['words']]
    ###
    ######## combine x y -> [{x:x,y:y},{},,,]
    xydictll = [{'w': w, 'eventflag': flag} for w,  flag in
                zip(wordll, whether_event_mention)]
    xydictll[rootPos]['w'] = 'triggerRoot'
    #############
    # word -> char
    charll = []
    charId_to_cutId={}
    for did,dic in enumerate(xydictll):
        if dic['eventflag'] == False:  # 只把不是EVENT MENTION的单词拆成字
            for char in dic['w']:
                #charll.append({'char': char, 'y': 'unlabel'})
                charll.append(char)
                ##
                charId=len(charll)-1
                charId_to_cutId[charId]=did
        else:
            #charll.append({'char': dic['w'], 'y': dic['y']})
            charll.append(dic['w'])
            ##
            charId = len(charll) - 1
            charId_to_cutId[charId] = did
    return charll,charId_to_cutId


def make_outputFormat(wll,charId2cutId):#['subject','time','trigger','value','value','u','u'...]
    root,rootPos=None,None
    if len(wll)==len(charId2cutId): #预测结果和输入长度一致
        headRelaTail_ll=[] #[{"trigger":8, "relation": "Times", "pos": 6},,{},,,]
        for wi,w in enumerate(wll):
            if w=='trigger':
                root=w
                rootPos=wi
                break
        if not rootPos: # 没有trigger
            return []
        for wi,w in enumerate(wll):
            if w in ['unlabel','UNK','trigger']:continue
            else:
                headRelaTail_ll.append({root:charId2cutId[rootPos],
                                        'relation':w,
                                        'pos':charId2cutId[wi]})
        #######
        return headRelaTail_ll
    if root==None:
        return []



def make_relationTask_input(wll):
    return ['relationTask']+wll

def make_validTask_input(wll):
    return ['classificationTask']+wll

def remove_taskPrompt(result):#[taskxx,sxxx,wefsdf,...]->[sxxx,wefsdf...]
    return result[1:]


