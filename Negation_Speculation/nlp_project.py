# %% [code]
# %% [code]
# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false}}

'''
The code is taken from https://github.com/adityak6798/Transformers-For-Negation-and-Speculation/blob/master/Multitask_Learning_of_Negation_and_Speculation.ipynb,
then adapted to accomplish our goal.
'''


from transformers import BertForTokenClassification, RobertaForTokenClassification
# from model import RobertaForTokenClassification
import torch
from torch.optim import Adam
#from early_stopping import EarlyStopping
import numpy as np
#from metrics import f1_score, f1_cues, f1_scope, flat_accuracy, flat_accuracy_positive_cues, report_per_class_accuracy, scope_accuracy
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, ReLU
from transformers import pipeline, AutoTokenizer

# %% [code] {"id":"s8YfmznL4B-G","papermill":{"duration":0.110146,"end_time":"2022-11-26T23:58:04.696040","exception":false,"start_time":"2022-11-26T23:58:04.585894","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:57.683528Z","iopub.execute_input":"2022-11-27T16:04:57.683992Z","iopub.status.idle":"2022-11-27T16:04:57.706106Z","shell.execute_reply.started":"2022-11-27T16:04:57.683952Z","shell.execute_reply":"2022-11-27T16:04:57.704909Z"},"jupyter":{"outputs_hidden":false}}
import os, re, torch, html, tempfile, copy, json, math, shutil, tarfile, tempfile, sys, random, pickle, string
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, ReLU
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import RobertaTokenizer, BertForTokenClassification, BertTokenizer, BertConfig, BertModel, WordpieceTokenizer, XLNetTokenizer
from transformers.file_utils import cached_path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import stats
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['RoBERTa', 'AugNB', 'CueNB'], default='RoBERTa')
parser.add_argument('--subtask', choices=['cue_detection', 'scope_resolution'], default='cue_detection')
parser.add_argument('--train-datasets', nargs='+', default=['bioscope_full_papers'])
args = parser.parse_known_args()[0]
print("Input args:")
print(args)

# %% [code] {"papermill":{"duration":6.420061,"end_time":"2022-11-26T23:57:58.777323","exception":false,"start_time":"2022-11-26T23:57:52.357262","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:54.160440Z","iopub.execute_input":"2022-11-27T16:04:54.161049Z","iopub.status.idle":"2022-11-27T16:04:55.935993Z","shell.execute_reply.started":"2022-11-27T16:04:54.161001Z","shell.execute_reply":"2022-11-27T16:04:55.934916Z"},"jupyter":{"outputs_hidden":false}}
#PRETRAINED_PATH_augnb='../input/augnbzip/augnb'
PRETRAINED_PATH_augnb='../input/augnbzip/augnb'
#PRETRAINED_PATH_augnb='../input/d/kritibhattarai/augnbzip/augnb'

# %% [code] {"papermill":{"duration":0.536655,"end_time":"2022-11-26T23:57:59.370641","exception":false,"start_time":"2022-11-26T23:57:58.833986","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# augnb_model=RobertaForTokenClassification.from_pretrained(PRETRAINED_PATH_augnb)
# #Test if the model model is loaded correctly and can be used like traditional BERT model
# tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH_augnb)
# classifier  = pipeline("ner",model = augnb_model,tokenizer=tokenizer)
# res = classifier("I do not like it")
# print(res) #the result is not great as it is pretrained on negative focused dataset, not fined tuned for ner application. However, the point is we have successfully loaded the negative focused pretrained model of the corresponding paper (Truong et al.,2022).

# %% [code] {"papermill":{"duration":5.185676,"end_time":"2022-11-26T23:58:04.570875","exception":false,"start_time":"2022-11-26T23:57:59.385199","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:56.151950Z","iopub.execute_input":"2022-11-27T16:04:56.152826Z","iopub.status.idle":"2022-11-27T16:04:57.682070Z","shell.execute_reply.started":"2022-11-27T16:04:56.152789Z","shell.execute_reply":"2022-11-27T16:04:57.681177Z"},"jupyter":{"outputs_hidden":false}}
PRETRAINED_PATH_cuenb='../input/cuenbzip/cuenb'
#PRETRAINED_PATH_cuenb='../input/d/kritibhattarai/cuenbzip/cuenb'

# cuenb_model=RobertaForTokenClassification.from_pretrained(PRETRAINED_PATH_cuenb)

# %% [code] {"id":"pVhKBUwW4Hz2","papermill":{"duration":0.022865,"end_time":"2022-11-26T23:58:04.768673","exception":false,"start_time":"2022-11-26T23:58:04.745808","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:57.717896Z","iopub.execute_input":"2022-11-27T16:04:57.718319Z","iopub.status.idle":"2022-11-27T16:04:57.728902Z","shell.execute_reply.started":"2022-11-27T16:04:57.718283Z","shell.execute_reply":"2022-11-27T16:04:57.727936Z"},"jupyter":{"outputs_hidden":false}}
MAX_LEN = 128
bs = 8
EPOCHS = 60

PATIENCE = 6
INITIAL_LEARNING_RATE = 3e-5
NUM_RUNS = 3 #Number of times to run the training and evaluation code


if args.model == 'AugNB':
    CUE_MODEL = PRETRAINED_PATH_augnb #PRETRAINED_PATH_augnb  PRETRAINED_PATH_cuenb
    SCOPE_MODEL = PRETRAINED_PATH_augnb
elif args.model == 'CueNB':
    CUE_MODEL = PRETRAINED_PATH_cuenb
    SCOPE_MODEL = PRETRAINED_PATH_cuenb
elif args.model == 'RoBERTa':
    CUE_MODEL = 'roberta-base'
    SCOPE_MODEL = 'roberta-base'
else:
    raise ArgumentError(f"Unsupported model {parser.model}")
SCOPE_METHOD = 'local' # Options: global, local
EARLY_STOPPING_METHOD = 'combined' # Options: combined, separate
if args.subtask == 'cue_detection':
    ERROR_ANALYSIS_FOR_SCOPE = False # Options: True, False
    SUBTASK = 'cue_detection' # Options: cue_detection, scope_resolution
elif args.subtask == 'scope_resolution':
    ERROR_ANALYSIS_FOR_SCOPE = True # Options: True, False
    SUBTASK = 'scope_resolution' # Options: cue_detection, scope_resolution

TRAIN_DATASETS = args.train_datasets
TEST_DATASETS = ['bioscope_full_papers',
                 'bioscope_abstracts','sfu']

#TELEGRAM_CHAT_ID = #Replace with chat ID for telegram notifications
#TELEGRAM_TOKEN = #Replace with token for telegram notifications

# %% [code] {"id":"5usbT29U4Isg","papermill":{"duration":1.178679,"end_time":"2022-11-26T23:58:05.961105","exception":false,"start_time":"2022-11-26T23:58:04.782426","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:57.734818Z","iopub.execute_input":"2022-11-27T16:04:57.735156Z","iopub.status.idle":"2022-11-27T16:04:58.829032Z","shell.execute_reply.started":"2022-11-27T16:04:57.735130Z","shell.execute_reply":"2022-11-27T16:04:58.827869Z"},"jupyter":{"outputs_hidden":false}}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json"
}

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json"
}

XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
}

TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
print(device,n_gpu)
print(torch.cuda.is_available())

#nvidia-smi

import time
out_data = []
def store_result(res_dict, run_num):
    res_dict["Run Number"] = run_num+1
    res_dict["Timestamp"] = int(time.time())
    print(f"Storing results for run {run_num}:")
    print(res_dict)
    out_data.append(res_dict)

# Writes the current output data to a file
import csv
def save_result():
    with open("output.tsv", "a+") as f:
        keys = out_data[0].keys()
        writer = csv.DictWriter(f, keys, delimiter="\t")
        # Write headers if file is empty
        f.seek(0)
        if not f.read(1):
            writer.writeheader()
        writer.writerows(out_data)
    out_data = []



# %% [code] {"id":"SeaGV5XvJzwe","papermill":{"duration":0.025266,"end_time":"2022-11-26T23:58:06.001051","exception":false,"start_time":"2022-11-26T23:58:05.975785","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:58.830859Z","iopub.execute_input":"2022-11-27T16:04:58.831184Z","iopub.status.idle":"2022-11-27T16:04:58.840873Z","shell.execute_reply.started":"2022-11-27T16:04:58.831155Z","shell.execute_reply":"2022-11-27T16:04:58.839719Z"},"jupyter":{"outputs_hidden":false}}
class Cues:
    def __init__(self, data):
        self.sentences = data[0]
        self.negation_cues = data[1]
        self.speculation_cues = data[2]
        self.num_sentences = len(data[0])
class Scopes:
    def __init__(self, data):
        self.negation_sentences = data[0][0]
        self.speculation_sentences = data[1][0]
        self.negation_cues = data[0][1]
        self.speculation_cues = data[1][1]
        self.negation_scopes = data[0][2]
        self.speculation_scopes = data[1][2]
        self.num_sentences = len(data[0])

# %% [code] {"id":"Xe1kG3IZKBrD","papermill":{"duration":0.172561,"end_time":"2022-11-26T23:58:06.187530","exception":false,"start_time":"2022-11-26T23:58:06.014969","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:58.843620Z","iopub.execute_input":"2022-11-27T16:04:58.844034Z","iopub.status.idle":"2022-11-27T16:04:59.322914Z","shell.execute_reply.started":"2022-11-27T16:04:58.843992Z","shell.execute_reply":"2022-11-27T16:04:59.321948Z"},"jupyter":{"outputs_hidden":false}}
class Data:
    def __init__(self, file, dataset_name = 'bioscope', error_analysis = False):
        '''
        file: The path of the data file.
        dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, starsem.
        frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        '''

        def bioscope(f_path, cue_sents_only=False):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences+=re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            cue_only_data = []
            negation_cue_cues = []
            speculation_cue_cues = []
            negation_scope_cues = []
            speculation_scope_cues = []
            negation_scope_scopes = []
            speculation_scope_scopes = []
            negation_scope_sentence = []
            speculation_scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            s_idx = []
            cue_id_to_task = {}
            in_sentence = 0
            for token in sentences:
                if token == '':
                    continue
                elif '<sentence' in token:
                    in_sentence = 1
                elif '<cue' in token:
                    in_cue.append(str(re.split('(ref=".*?")',token)[1][4:]))
                    c_idx.append(str(re.split('(ref=".*?")',token)[1][4:]))
                    if c_idx[-1] not in cue.keys():
                        cue[c_idx[-1]] = []
                    if 'speculation' in token:
                        cue_id_to_task[in_cue[-1]] = 'speculation'
                    else:
                        cue_id_to_task[in_cue[-1]] = 'negation'
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    #print(re.split('(id=".*?")',token)[1][3:])
                    in_scope.append(str(re.split('(id=".*?")',token)[1][3:]))
                    s_idx.append(str(re.split('(id=".*?")',token)[1][3:]))
                    scope[s_idx[-1]] = []
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '</sentence' in token:
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence), [3]*len(sentence)]) # Sentence, Negation Cues, Speculation Cues
                    else:
                        cue_sentence.append(sentence)
                        negation_cue_cues.append([3]*len(sentence))
                        speculation_cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            if cue_id_to_task[i] == 'negation':
                                negation_scope_sentence.append(sentence)
                                negation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    negation_cue_cues[-1][cue[i][0]] = 1
                                    negation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        negation_cue_cues[-1][c] = 2
                                        negation_scope_cues[-1][c] = 2
                                negation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        negation_scope_scopes[-1][s] = 1
                            else:
                                speculation_scope_sentence.append(sentence)
                                speculation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    speculation_cue_cues[-1][cue[i][0]] = 1
                                    speculation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        speculation_cue_cues[-1][c] = 2
                                        speculation_scope_cues[-1][c] = 2
                                speculation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        speculation_scope_scopes[-1][s] = 1

                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_sentence = 0
                    c_idx = []
                    s_idx = []
                    cue_id_to_task = {}
                elif '<' not in token:
                    if in_sentence==1:
                        words = token.split()
                        sentence+=words
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i]+=[word_num+i for i in range(len(words))]
                        elif len(in_scope)!=0:
                            for i in in_scope:
                                scope[i]+=[word_num+i for i in range(len(words))]
                        word_num+=len(words)

            cue_only_sents = [i[0] for i in cue_only_data]
            negation_cue_only_cues = [i[1] for i in cue_only_data]
            speculation_cue_only_cues = [i[2] for i in cue_only_data]
            cue_train_data = (cue_sentence+cue_only_sents, negation_cue_cues+negation_cue_only_cues, speculation_cue_cues+speculation_cue_only_cues)
            scope_train_data = ([negation_scope_sentence, negation_scope_cues, negation_scope_scopes], [speculation_scope_sentence, speculation_scope_cues, speculation_scope_scopes])
            return [cue_train_data, scope_train_data]

        def sfu_review(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences+=re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            negation_cue_cues = []
            speculation_cue_cues = []
            negation_scope_cues = []
            speculation_scope_cues = []
            negation_scope_scopes = []
            speculation_scope_scopes = []
            negation_scope_sentence = []
            speculation_scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            cue_only_data = []
            s_idx = []
            in_word = 0
            cue_id_to_task = {}
            for token in sentences:
                if token == '':
                    continue
                elif token == '<W>':
                    in_word = 1
                elif token == '</W>':
                    in_word = 0
                    word_num += 1
                elif '<cue' in token:
                    in_cue.append(int(re.split('(ID=".*?")',token)[1][4:-1]))
                    c_idx.append(in_cue[-1])
                    if c_idx[-1] not in cue.keys():
                        cue[c_idx[-1]] = []
                    if 'speculation' in token:
                        cue_id_to_task[in_cue[-1]] = 'speculation'
                    else:
                        cue_id_to_task[in_cue[-1]] = 'negation'
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    continue
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '<ref' in token:
                    in_scope.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    s_idx.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    for i in s_idx[-1]:
                        scope[i] = []
                elif '</SENTENCE' in token:
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence), [3]*len(sentence)]) # Sentence, Negation Cues, Speculation Cues
                    else:
                        cue_sentence.append(sentence)
                        negation_cue_cues.append([3]*len(sentence))
                        speculation_cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            if cue_id_to_task[i] == 'negation':
                                negation_scope_sentence.append(sentence)
                                negation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    negation_cue_cues[-1][cue[i][0]] = 1
                                    negation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        negation_cue_cues[-1][c] = 2
                                        negation_scope_cues[-1][c] = 2
                                negation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        negation_scope_scopes[-1][s] = 1
                            else:
                                speculation_scope_sentence.append(sentence)
                                speculation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    speculation_cue_cues[-1][cue[i][0]] = 1
                                    speculation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        speculation_cue_cues[-1][c] = 2
                                        speculation_scope_cues[-1][c] = 2
                                speculation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        speculation_scope_scopes[-1][s] = 1
                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_word = 0
                    c_idx = []
                    s_idx = []
                    cue_id_to_task = {}
                elif '<' not in token:
                    if in_word == 1:
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i].append(word_num)
                        if len(in_scope)!=0:
                            for i in in_scope:
                                for j in i:
                                    scope[j].append(word_num)
                        sentence.append(token)
            cue_only_sents = [i[0] for i in cue_only_data]
            negation_cue_only_cues = [i[1] for i in cue_only_data]
            speculation_cue_only_cues = [i[2] for i in cue_only_data]
            cue_train_data = (cue_sentence+cue_only_sents, negation_cue_cues+negation_cue_only_cues, speculation_cue_cues+speculation_cue_only_cues)
            scope_train_data = ([negation_scope_sentence, negation_scope_cues, negation_scope_scopes], [speculation_scope_sentence, speculation_scope_cues, speculation_scope_scopes])
            return [cue_train_data, scope_train_data]

        if dataset_name == 'bioscope':
            ret_val = bioscope(file)
            cue_data_to_proc = ret_val[0]
            scope_data_to_proc = ret_val[1]
        elif dataset_name == 'sfu':
            sfu_cues = [[], [], []]
            sfu_scopes = [[[], [], []], [[], [], []]]
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(file+"//"+dir_name):
                        r_val = sfu_review(file+"//"+dir_name+'//'+f_name)
                        sfu_cues = [a+b for a,b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [[a+b for a,b in zip(i,j)] for i,j in zip(sfu_scopes, r_val[1])]

            cue_data_to_proc = sfu_cues
            scope_data_to_proc = sfu_scopes
        else:
            raise ValueError("Supported Dataset types are:\n\tbioscope\n\tsfu")
        if error_analysis == True:
            neg_punct, neg_no_punct = [[],[],[]], [[],[],[]]
            for sentence, scope_c, scope in zip(scope_data_to_proc[0][0], scope_data_to_proc[0][1], scope_data_to_proc[0][2]):
                c_ids = [idx for idx, x in enumerate(scope_c) if x != 3]
                min_c_id = min(c_ids)
                max_c_id = max(c_ids)
                scope_a = scope.copy()
                for c in c_ids:
                    scope_a[c] = 1
                punct_ids = set([idx for idx, x in enumerate(sentence) for sym in string.punctuation if sym in x])
                if len(punct_ids) == 0:
                    neg_no_punct[0].append(sentence)
                    neg_no_punct[1].append(scope_c)
                    neg_no_punct[2].append(scope)
                    continue
                min_p_id = [idx for idx in punct_ids if idx < min_c_id]
                if len(min_p_id) == 0:
                    min_p_id = -1
                else:
                    min_p_id = max(min_p_id)
                max_p_id = [idx for idx in punct_ids if idx > max_c_id]
                if len(max_p_id) == 0:
                    max_p_id = -1
                else:
                    max_p_id = min(max_p_id)
                s_ids = [idx for idx, s in enumerate(scope_a) if s==1]
                last_scope_id = max(s_ids)
                first_scope_id = min(s_ids)
                if (last_scope_id+1 == max_p_id or last_scope_id == max_p_id) or (first_scope_id-1 == min_p_id or first_scope_id == min_p_id): # or (last_scope_id in punct_ids)
                    neg_punct[0].append(sentence)
                    neg_punct[1].append(scope_c)
                    neg_punct[2].append(scope)
                else:
                    neg_no_punct[0].append(sentence)
                    neg_no_punct[1].append(scope_c)
                    neg_no_punct[2].append(scope)
            spec_punct, spec_no_punct = [[],[],[]], [[],[],[]]
            for sentence, scope_c, scope in zip(scope_data_to_proc[1][0], scope_data_to_proc[1][1], scope_data_to_proc[1][2]):
                c_ids = [idx for idx, x in enumerate(scope_c) if x != 3]
                min_c_id = min(c_ids)
                max_c_id = max(c_ids)
                scope_a = scope.copy()
                for c in c_ids:
                    scope_a[c] = 1
                punct_ids = set([idx for idx, x in enumerate(sentence) for sym in string.punctuation if sym in x])
                if len(punct_ids) == 0:
                    spec_no_punct[0].append(sentence)
                    spec_no_punct[1].append(scope_c)
                    spec_no_punct[2].append(scope)
                    continue
                min_p_id = [idx for idx in punct_ids if idx < min_c_id]
                if len(min_p_id) == 0:
                    min_p_id = -1
                else:
                    min_p_id = max(min_p_id)
                max_p_id = [idx for idx in punct_ids if idx > max_c_id]
                if len(max_p_id) == 0:
                    max_p_id = -1
                else:
                    max_p_id = min(max_p_id)
                s_ids = [idx for idx, s in enumerate(scope_a) if s==1]
                last_scope_id = max(s_ids)
                first_scope_id = min(s_ids)
                if (last_scope_id+1 == max_p_id or last_scope_id == max_p_id) or (first_scope_id-1 == min_p_id or first_scope_id == min_p_id): # or (last_scope_id in punct_ids)
                    spec_punct[0].append(sentence)
                    spec_punct[1].append(scope_c)
                    spec_punct[2].append(scope)
                else:
                    spec_no_punct[0].append(sentence)
                    spec_no_punct[1].append(scope_c)
                    spec_no_punct[2].append(scope)

            self.scope_data_punct = Scopes([neg_punct, spec_punct])
            self.scope_data_no_punct = Scopes([neg_no_punct, spec_no_punct])
        else:
            self.scope_data_punct =None
            self.scope_data_no_punct = None
        self.cue_data = Cues(cue_data_to_proc)
        self.scope_data = Scopes(scope_data_to_proc)

    def get_cue_dataloader(self, val_size = 0.15, test_size = 0.15, other_datasets = []):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        if 'xlnet' in CUE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in CUE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in CUE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        elif 'cuenb' in CUE_MODEL:  ######irf
            tokenizer = RobertaTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)
        elif 'augnb' in CUE_MODEL:  ######irf
            tokenizer = RobertaTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)
            #sys.exit(1)
            #tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)
        def preprocess_data(obj, tokenizer):
            dl_sents = obj.cue_data.sentences
            dl_negation_cues = obj.cue_data.negation_cues
            dl_speculation_cues = obj.cue_data.speculation_cues

            sentences = [" ".join(sent) for sent in dl_sents]

            mytexts = []
            myneglabels = []
            myspeclabels = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences
            for sent, neg_tags, spec_tags in zip(sentences_clean, dl_negation_cues, dl_speculation_cues):
                new_neg_tags = []
                new_spec_tags = []
                new_text = []
                new_masks = []
                for word, neg_tag, spec_tag in zip(sent.split(),neg_tags,spec_tags):
                    #print('splitting: ', word)
                    sub_words = tokenizer._tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_neg_tags.append(neg_tag)
                        new_spec_tags.append(spec_tag)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                myneglabels.append(new_neg_tags)
                myspeclabels.append(new_spec_tags)

            input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in mytexts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            neg_tags = pad_sequences(myneglabels,
                                maxlen=MAX_LEN, value=4, padding="post",
                                dtype="long", truncating="post").tolist()

            spec_tags = pad_sequences(myspeclabels,
                                maxlen=MAX_LEN, value=4, padding="post",
                                dtype="long", truncating="post").tolist()

            mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype='long', truncating='post').tolist()

            attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

            random_state = np.random.randint(1,2020)

            tra_inputs, test_inputs, tra_neg_tags, test_neg_tags = train_test_split(input_ids, neg_tags, test_size=test_size, random_state = random_state)
            _, _, tra_spec_tags, test_spec_tags = train_test_split(input_ids, spec_tags, test_size=test_size, random_state = random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size, random_state = random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(mymasks, input_ids, test_size=test_size, random_state = random_state)

            random_state_2 = np.random.randint(1,2020)

            tr_inputs, val_inputs, tr_neg_tags, val_neg_tags = train_test_split(tra_inputs, tra_neg_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            _, _, tr_spec_tags, val_spec_tags = train_test_split(tra_inputs, tra_spec_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

            return [tr_inputs, tr_neg_tags, tr_spec_tags, tr_masks, tr_mymasks], [val_inputs, val_neg_tags, val_spec_tags, val_masks, val_mymasks], [test_inputs, test_neg_tags, test_spec_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_neg_tags = []
        tr_spec_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets)+1)]
        test_inputs = [[] for i in range(len(other_datasets)+1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs+=train_ret_val[0]
        tr_neg_tags+=train_ret_val[1]
        tr_spec_tags+=train_ret_val[2]
        tr_masks+=train_ret_val[3]
        tr_mymasks+=train_ret_val[4]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        val_inputs[0].append(val_ret_val[4])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])
        test_inputs[0].append(test_ret_val[4])

        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs+=train_ret_val[0]
            tr_neg_tags+=train_ret_val[1]
            tr_spec_tags+=train_ret_val[2]
            tr_masks+=train_ret_val[3]
            tr_mymasks+=train_ret_val[4]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            val_inputs[idx].append(val_ret_val[4])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])
            test_inputs[idx].append(test_ret_val[4])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_neg_tags = torch.LongTensor(tr_neg_tags)
        tr_spec_tags = torch.LongTensor(tr_spec_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_neg_tags, tr_spec_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i,j,k,l,m in val_inputs:
            val_data = TensorDataset(i, l, j, k, m)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i,j,k,l,m in test_inputs:
            test_data = TensorDataset(i, l, j, k, m)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders

    def get_scope_dataloader(self, val_size = 0.15, test_size=0.15, other_datasets = [], error_analysis = False, punct_dl = False):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''

        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        if 'xlnet' in SCOPE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in SCOPE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in SCOPE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        elif 'cuenb' in SCOPE_MODEL:  ######irf
            tokenizer = RobertaTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case)
        elif 'augnb' in SCOPE_MODEL:  ######irf
            tokenizer = RobertaTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case)
            #sys.exit(1)
            #tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)
        def preprocess_data(obj, tokenizer_obj):
            if error_analysis == False:
                #print("obj.scope_data")
                #print(obj.scope_data)

                dl_neg_sents = obj.scope_data.negation_sentences
                dl_neg_cues = obj.scope_data.negation_cues
                dl_neg_scopes = obj.scope_data.negation_scopes
                dl_spec_sents = obj.scope_data.speculation_sentences
                dl_spec_cues = obj.scope_data.speculation_cues
                dl_spec_scopes = obj.scope_data.speculation_scopes
            else:
                if punct_dl == False:
                   # print("obj.scope_data_no_punct")
                   # print(obj.scope_data_no_punct)
                    dl_neg_sents = obj.scope_data_no_punct.negation_sentences
                    dl_neg_cues = obj.scope_data_no_punct.negation_cues
                    dl_neg_scopes = obj.scope_data_no_punct.negation_scopes
                    dl_spec_sents = obj.scope_data_no_punct.speculation_sentences
                    dl_spec_cues = obj.scope_data_no_punct.speculation_cues
                    dl_spec_scopes = obj.scope_data_no_punct.speculation_scopes
                else:
                    #print("---X---")
                    #print("obj.scope_data_punct")
                    #print("----X--")
                    #print(obj.scope_data_punct)
                    dl_neg_sents = obj.scope_data_punct.negation_sentences
                    dl_neg_cues = obj.scope_data_punct.negation_cues
                    dl_neg_scopes = obj.scope_data_punct.negation_scopes
                    dl_spec_sents = obj.scope_data_punct.speculation_sentences
                    dl_spec_cues = obj.scope_data_punct.speculation_cues
                    dl_spec_scopes = obj.scope_data_punct.speculation_scopes
            if SCOPE_METHOD == 'global':
                neg_sentences = [" ".join([s for s in sent+[' [SEP] Negation']]) for sent in dl_neg_sents]
                dl_neg_scopes = [scope_sent+[0,0] for scope_sent in dl_neg_scopes]
                dl_neg_cues = [cue_sent+[3,3] for cue_sent in dl_neg_cues]
                spec_sentences = [" ".join([s for s in sent+[' [SEP] Speculation']]) for sent in dl_spec_sents]
                dl_spec_scopes = [scope_sent+[0,0] for scope_sent in dl_spec_scopes]
                dl_spec_cues = [cue_sent+[3,3] for cue_sent in dl_spec_cues]
            else:
                neg_sentences = [" ".join([s for s in sent]) for sent in dl_neg_sents]
                spec_sentences = [" ".join([s for s in sent]) for sent in dl_spec_sents]

            neg_mytexts = []
            neg_mylabels = []
            neg_mycues = []
            neg_mymasks = []
            spec_mytexts = []
            spec_mylabels = []
            spec_mycues = []
            spec_mymasks = []

            if do_lower_case == True:
                neg_sentences_clean = [sent.lower() for sent in neg_sentences]
                spec_sentences_clean = [sent.lower() for sent in spec_sentences]
            else:
                neg_sentences_clean = neg_sentences
                spec_sentences_clean = spec_sentences

            for sent, tags, cues in zip(neg_sentences_clean, dl_neg_scopes, dl_neg_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(),tags,cues):
                    sub_words = tokenizer._tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                neg_mymasks.append(new_masks)
                neg_mytexts.append(new_text)
                neg_mylabels.append(new_tags)
                neg_mycues.append(new_cues)

            for sent, tags, cues in zip(spec_sentences_clean, dl_spec_scopes, dl_spec_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(),tags,cues):
                    sub_words = tokenizer._tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                spec_mymasks.append(new_masks)
                spec_mytexts.append(new_text)
                spec_mylabels.append(new_tags)
                spec_mycues.append(new_cues)

            final_negation_sentences = []
            final_negation_labels = []
            final_negation_masks = []
            final_speculation_sentences = []
            final_speculation_labels = []
            final_speculation_masks = []

            if SCOPE_METHOD == 'global':
                for sent,cues,labels,masks in zip(neg_mytexts, neg_mycues, neg_mylabels, neg_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(mask)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_negation_sentences.append(temp_sent)
                    final_negation_labels.append(temp_label)
                    final_negation_masks.append(temp_masks)

                for sent,cues,labels,masks in zip(spec_mytexts, spec_mycues, spec_mylabels, spec_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(mask)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(mask)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_speculation_sentences.append(temp_sent)
                    final_speculation_labels.append(temp_label)
                    final_speculation_masks.append(temp_masks)

            elif SCOPE_METHOD == 'local':

                for sent,cues,labels,masks in zip(neg_mytexts, neg_mycues, neg_mylabels, neg_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_negation_sentences.append(temp_sent)
                    final_negation_labels.append(temp_label)
                    final_negation_masks.append(temp_masks)

                for sent,cues,labels,masks in zip(spec_mytexts, spec_mycues, spec_mylabels, spec_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+6}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_speculation_sentences.append(temp_sent)
                    final_speculation_labels.append(temp_label)
                    final_speculation_masks.append(temp_masks)

            else:
                raise ValueError("Supported methods for scope detection are:\nrglobal\nlocal")

            neg_input_ids = pad_sequences([[tokenizer_obj._convert_token_to_id(word) for word in txt] for txt in final_negation_sentences],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            spec_input_ids = pad_sequences([[tokenizer_obj._convert_token_to_id(word) for word in txt] for txt in final_speculation_sentences],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            neg_tags = pad_sequences(final_negation_labels,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            spec_tags = pad_sequences(final_speculation_labels,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            neg_final_masks = pad_sequences(final_negation_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            spec_final_masks = pad_sequences(final_speculation_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            neg_attention_masks = [[float(i>0) for i in ii] for ii in neg_input_ids]

            spec_attention_masks = [[float(i>0) for i in ii] for ii in spec_input_ids]

            if test_size > 0.99:
                neg_tr_inputs, neg_tr_tags, neg_tr_masks, neg_tr_mymasks = [], [], [], []
                neg_val_inputs, neg_val_tags, neg_val_masks, neg_val_mymasks = [], [], [], []
                neg_test_inputs, neg_test_tags, neg_test_masks, neg_test_mymasks = neg_input_ids, neg_tags, neg_attention_masks, neg_final_masks
                spec_tr_inputs, spec_tr_tags, spec_tr_masks, spec_tr_mymasks = [], [], [], []
                spec_val_inputs, spec_val_tags, spec_val_masks, spec_val_mymasks = [], [], [], []
                spec_test_inputs, spec_test_tags, spec_test_masks, spec_test_mymasks = spec_input_ids, spec_tags, spec_attention_masks, spec_final_masks

            else:
                random_state = np.random.randint(1,2020)

                neg_tra_inputs, neg_test_inputs, neg_tra_tags, neg_test_tags = train_test_split(neg_input_ids, neg_tags, test_size=test_size, random_state = random_state)
                neg_tra_masks, neg_test_masks, _, _ = train_test_split(neg_attention_masks, neg_input_ids, test_size=test_size, random_state = random_state)
                neg_tra_mymasks, neg_test_mymasks, _, _ = train_test_split(neg_final_masks, neg_input_ids, test_size=test_size, random_state = random_state)

                spec_tra_inputs, spec_test_inputs, spec_tra_tags, spec_test_tags = train_test_split(spec_input_ids, spec_tags, test_size=test_size, random_state = random_state)
                spec_tra_masks, spec_test_masks, _, _ = train_test_split(spec_attention_masks, spec_input_ids, test_size=test_size, random_state = random_state)
                spec_tra_mymasks, spec_test_mymasks, _, _ = train_test_split(spec_final_masks, spec_input_ids, test_size=test_size, random_state = random_state)

                random_state_2 = np.random.randint(1,2020)

                neg_tr_inputs, neg_val_inputs, neg_tr_tags, neg_val_tags = train_test_split(neg_tra_inputs, neg_tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                neg_tr_masks, neg_val_masks, _, _ = train_test_split(neg_tra_masks, neg_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                neg_tr_mymasks, neg_val_mymasks, _, _ = train_test_split(neg_tra_mymasks, neg_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

                spec_tr_inputs, spec_val_inputs, spec_tr_tags, spec_val_tags = train_test_split(spec_tra_inputs, spec_tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                spec_tr_masks, spec_val_masks, _, _ = train_test_split(spec_tra_masks, spec_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                spec_tr_mymasks, spec_val_mymasks, _, _ = train_test_split(spec_tra_mymasks, spec_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

            return (([neg_tr_inputs, neg_tr_tags, neg_tr_masks, neg_tr_mymasks], [neg_val_inputs, neg_val_tags, neg_val_masks, neg_val_mymasks], [neg_test_inputs, neg_test_tags, neg_test_masks, neg_test_mymasks]), ([spec_tr_inputs, spec_tr_tags, spec_tr_masks, spec_tr_mymasks], [spec_val_inputs, spec_val_tags, spec_val_masks, spec_val_mymasks], [spec_test_inputs, spec_test_tags, spec_test_masks, spec_test_mymasks]))

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        neg_val_inputs = []
        neg_val_tags = []
        neg_val_masks = []
        neg_val_mymasks = []
        spec_val_inputs = []
        spec_val_tags = []
        spec_val_masks = []
        spec_val_mymasks = []
        neg_test_inputs = [[] for i in range(len(other_datasets)+1)]
        spec_test_inputs = [[] for i in range(len(other_datasets)+1)]

        r_val = preprocess_data(self, tokenizer)
        [neg_train_ret_val, neg_val_ret_val, neg_test_ret_val] = r_val[0]
        [spec_train_ret_val, spec_val_ret_val, spec_test_ret_val] = r_val[1]
        tr_inputs += neg_train_ret_val[0]
        tr_tags += neg_train_ret_val[1]
        tr_masks += neg_train_ret_val[2]
        tr_mymasks += neg_train_ret_val[3]
        tr_inputs += spec_train_ret_val[0]
        tr_tags += spec_train_ret_val[1]
        tr_masks += spec_train_ret_val[2]
        tr_mymasks += spec_train_ret_val[3]

        neg_val_inputs += neg_val_ret_val[0]
        neg_val_tags += neg_val_ret_val[1]
        neg_val_masks += neg_val_ret_val[2]
        neg_val_mymasks += neg_val_ret_val[3]
        spec_val_inputs += spec_val_ret_val[0]
        spec_val_tags += spec_val_ret_val[1]
        spec_val_masks += spec_val_ret_val[2]
        spec_val_mymasks += spec_val_ret_val[3]

        neg_test_inputs[0].append(neg_test_ret_val[0])
        neg_test_inputs[0].append(neg_test_ret_val[1])
        neg_test_inputs[0].append(neg_test_ret_val[2])
        neg_test_inputs[0].append(neg_test_ret_val[3])

        spec_test_inputs[0].append(spec_test_ret_val[0])
        spec_test_inputs[0].append(spec_test_ret_val[1])
        spec_test_inputs[0].append(spec_test_ret_val[2])
        spec_test_inputs[0].append(spec_test_ret_val[3])

        for idx, arg in enumerate(other_datasets, 1):
            [neg_train_ret_val, neg_val_ret_val, neg_test_ret_val], [spec_train_ret_val, spec_val_ret_val, spec_test_ret_val] = preprocess_data(arg, tokenizer)
            tr_inputs += neg_train_ret_val[0]
            tr_tags += neg_train_ret_val[1]
            tr_masks += neg_train_ret_val[2]
            tr_mymasks += neg_train_ret_val[3]
            tr_inputs += spec_train_ret_val[0]
            tr_tags += spec_train_ret_val[1]
            tr_masks += spec_train_ret_val[2]
            tr_mymasks += spec_train_ret_val[3]

            neg_val_inputs += neg_val_ret_val[0]
            neg_val_tags += neg_val_ret_val[1]
            neg_val_masks += neg_val_ret_val[2]
            neg_val_mymasks += neg_val_ret_val[3]
            spec_val_inputs += spec_val_ret_val[0]
            spec_val_tags += spec_val_ret_val[1]
            spec_val_masks += spec_val_ret_val[2]
            spec_val_mymasks += spec_val_ret_val[3]

            neg_test_inputs[idx].append(neg_test_ret_val[0])
            neg_test_inputs[idx].append(neg_test_ret_val[1])
            neg_test_inputs[idx].append(neg_test_ret_val[2])
            neg_test_inputs[idx].append(neg_test_ret_val[3])

            spec_test_inputs[idx].append(spec_test_ret_val[0])
            spec_test_inputs[idx].append(spec_test_ret_val[1])
            spec_test_inputs[idx].append(spec_test_ret_val[2])
            spec_test_inputs[idx].append(spec_test_ret_val[3])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        neg_val_inputs = torch.LongTensor(neg_val_inputs)
        neg_val_tags = torch.LongTensor(neg_val_tags)
        neg_val_masks = torch.LongTensor(neg_val_masks)
        neg_val_mymasks = torch.LongTensor(neg_val_mymasks)
        spec_val_inputs = torch.LongTensor(spec_val_inputs)
        spec_val_tags = torch.LongTensor(spec_val_tags)
        spec_val_masks = torch.LongTensor(spec_val_masks)
        spec_val_mymasks = torch.LongTensor(spec_val_mymasks)
        neg_test_inputs = [[torch.LongTensor(i) for i in j] for j in neg_test_inputs]
        spec_test_inputs = [[torch.LongTensor(i) for i in j] for j in spec_test_inputs]

        if test_size < 0.99:
            train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

            neg_val_data = TensorDataset(neg_val_inputs, neg_val_masks, neg_val_tags, neg_val_mymasks)
            neg_val_sampler = RandomSampler(neg_val_data)
            neg_val_dataloader = DataLoader(neg_val_data, sampler=neg_val_sampler, batch_size=bs)

            spec_val_data = TensorDataset(spec_val_inputs, spec_val_masks, spec_val_tags, spec_val_mymasks)
            spec_val_sampler = RandomSampler(spec_val_data)
            spec_val_dataloader = DataLoader(spec_val_data, sampler=spec_val_sampler, batch_size=bs)

        else:
            train_data = []
            train_sampler = []
            train_dataloader = []

            neg_val_data = []
            neg_val_sampler = []
            neg_val_dataloader = []

            spec_val_data = []
            spec_val_sampler = []
            spec_val_dataloader = []

        neg_test_dataloaders = []
        for i,j,k,l in neg_test_inputs:
            neg_test_data = TensorDataset(i, k, j, l)
            neg_test_sampler = RandomSampler(neg_test_data)
            neg_test_dataloaders.append(DataLoader(neg_test_data, sampler=neg_test_sampler, batch_size=bs))

        spec_test_dataloaders = []
        for i,j,k,l in spec_test_inputs:
            spec_test_data = TensorDataset(i, k, j, l)
            spec_test_sampler = RandomSampler(spec_test_data)
            spec_test_dataloaders.append(DataLoader(spec_test_data, sampler=spec_test_sampler, batch_size=bs))

        return train_dataloader, [neg_val_dataloader, spec_val_dataloader], [neg_test_dataloaders, spec_test_dataloaders]

# %% [code] {"papermill":{"duration":0.013899,"end_time":"2022-11-26T23:58:06.215685","exception":false,"start_time":"2022-11-26T23:58:06.201786","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"u8-ZbcD9G52Y","papermill":{"duration":0.034895,"end_time":"2022-11-26T23:58:06.264427","exception":false,"start_time":"2022-11-26T23:58:06.229532","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.324573Z","iopub.execute_input":"2022-11-27T16:04:59.324944Z","iopub.status.idle":"2022-11-27T16:04:59.343861Z","shell.execute_reply.started":"2022-11-27T16:04:59.324909Z","shell.execute_reply":"2022-11-27T16:04:59.342884Z"},"jupyter":{"outputs_hidden":false}}
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    import re
    import tensorflow as tf
    tf_path = os.path.abspath(tf_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)
    return model

def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    import numpy as np
    import tensorflow as tf
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        if name not in tf_weights:
            continue
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
            array = np.transpose(array)
        if isinstance(pointer, list):
            # Here we will split the TF weigths
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)

    return model

# %% [code] {"id":"N4_nO6K6HW1_","papermill":{"duration":0.029893,"end_time":"2022-11-26T23:58:06.308321","exception":false,"start_time":"2022-11-26T23:58:06.278428","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.345436Z","iopub.execute_input":"2022-11-27T16:04:59.346264Z","iopub.status.idle":"2022-11-27T16:04:59.354868Z","shell.execute_reply.started":"2022-11-27T16:04:59.346228Z","shell.execute_reply":"2022-11-27T16:04:59.353972Z"},"jupyter":{"outputs_hidden":false}}
def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

# %% [code] {"id":"oEJoaNyLHAq_","papermill":{"duration":0.03733,"end_time":"2022-11-26T23:58:06.360052","exception":false,"start_time":"2022-11-26T23:58:06.322722","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.356511Z","iopub.execute_input":"2022-11-27T16:04:59.356916Z","iopub.status.idle":"2022-11-27T16:04:59.379965Z","shell.execute_reply.started":"2022-11-27T16:04:59.356883Z","shell.execute_reply":"2022-11-27T16:04:59.379088Z"},"jupyter":{"outputs_hidden":false}}
class PretrainedConfig(object):
    r""" Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.
        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.
        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained model configurations as values.
        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.output_past = kwargs.pop('output_past', True)  # Not used by all models
        self.torchscript = kwargs.pop('torchscript', False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.
        Parameters:
            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            return_unused_kwargs: (`optional`) bool:
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.
        Examples::
            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}
        """
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file)
            else:
                msg = "Model name '{}' was not found in model name list ({}). " \
                      "We assumed '{}' was a path or url to a configuration file named {} or " \
                      "a directory containing such a file but couldn't find any such file at this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file, CONFIG_NAME)
            raise EnvironmentError(msg)


        # Load config
        config = cls.from_json_file(resolved_config_file)

        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

# %% [code] {"papermill":{"duration":0.01363,"end_time":"2022-11-26T23:58:06.387517","exception":false,"start_time":"2022-11-26T23:58:06.373887","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"S83axCRbHwa5","papermill":{"duration":0.038698,"end_time":"2022-11-26T23:58:06.440664","exception":false,"start_time":"2022-11-26T23:58:06.401966","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.382598Z","iopub.execute_input":"2022-11-27T16:04:59.383210Z","iopub.status.idle":"2022-11-27T16:04:59.406024Z","shell.execute_reply.started":"2022-11-27T16:04:59.383175Z","shell.execute_reply":"2022-11-27T16:04:59.405109Z"},"jupyter":{"outputs_hidden":false}}
class BertConfig(PretrainedConfig):
    r"""
        :class:`~transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")

class RobertaConfig(BertConfig):
    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

class XLNetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a ``XLNetModel``.
    Args:
        vocab_size_or_config_json_file: Vocabulary size of ``inputs_ids`` in ``XLNetModel``.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for XLNet, 'uni' for Transformer-XL
        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.
        dropout: float, dropout rate.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
        finetuning_task: name of the glue task on which the model was fine-tuned if any
    """
    pretrained_config_archive_map = XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=32000,
                 d_model=1024,
                 n_layer=24,
                 n_head=16,
                 d_inner=4096,
                 max_position_embeddings=512,
                 ff_activation="gelu",
                 untie_r=True,
                 attn_type="bi",

                 initializer_range=0.02,
                 layer_norm_eps=1e-12,

                 dropout=0.1,
                 mem_len=None,
                 reuse_len=None,
                 bi_data=False,
                 clamp_len=-1,
                 same_length=False,

                 finetuning_task=None,
                 num_labels=2,
                 summary_type='last',
                 summary_use_proj=True,
                 summary_activation='tanh',
                 summary_last_dropout=0.1,
                 start_n_top=5,
                 end_n_top=5,
                 **kwargs):
        """Constructs XLNetConfig.
        """
        super(XLNetConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                setattr(config, key, value)
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.d_model = d_model
            self.n_layer = n_layer
            self.n_head = n_head
            assert d_model % n_head == 0
            self.d_head = d_model // n_head
            self.ff_activation = ff_activation
            self.d_inner = d_inner
            self.untie_r = untie_r
            self.attn_type = attn_type

            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps

            self.dropout = dropout
            self.mem_len = mem_len
            self.reuse_len = reuse_len
            self.bi_data = bi_data
            self.clamp_len = clamp_len
            self.same_length = same_length

            self.finetuning_task = finetuning_task
            self.num_labels = num_labels
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_last_dropout = summary_last_dropout
            self.start_n_top = start_n_top
            self.end_n_top = end_n_top
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def vocab_size(self):
        return self.n_token

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_token = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

# %% [code] {"papermill":{"duration":0.013677,"end_time":"2022-11-26T23:58:06.468551","exception":false,"start_time":"2022-11-26T23:58:06.454874","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"cDe20E7uIO1w","papermill":{"duration":0.056579,"end_time":"2022-11-26T23:58:06.539177","exception":false,"start_time":"2022-11-26T23:58:06.482598","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.407793Z","iopub.execute_input":"2022-11-27T16:04:59.408223Z","iopub.status.idle":"2022-11-27T16:04:59.451189Z","shell.execute_reply.started":"2022-11-27T16:04:59.408190Z","shell.execute_reply":"2022-11-27T16:04:59.450263Z"},"jupyter":{"outputs_hidden":false}}
class PreTrainedModel(nn.Module):
    r""" Base class for all models.
        :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods commons to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.
        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:
                - ``model``: an instance of the relevant subclass of :class:`~transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.
            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end
        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.
        Arguments:
            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, 'tie_weights'):
            self.tie_weights()

        return model_embeds

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.
            Arguments:
                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed

        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.
        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``
        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.
        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.
        Parameters:
            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.
            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:
                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.
        Examples::
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                force_download=force_download,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError("Error no file named {} found in directory {} or `from_tf` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                        pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                assert from_tf, "Error finding file {}, no file or TF 1.X checkpoint found".format(pretrained_model_name_or_path)
                archive_file = pretrained_model_name_or_path + ".index"

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                            archive_file)
                else:
                    msg = "Model name '{}' was not found in model name list ({}). " \
                        "We assumed '{}' was a path or url to model weight files named one of {} but " \
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
                raise EnvironmentError(msg)


        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
            if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model

# %% [code] {"papermill":{"duration":0.013683,"end_time":"2022-11-26T23:58:06.566608","exception":false,"start_time":"2022-11-26T23:58:06.552925","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"owbESENIIXji","papermill":{"duration":0.083957,"end_time":"2022-11-26T23:58:06.664655","exception":false,"start_time":"2022-11-26T23:58:06.580698","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.454086Z","iopub.execute_input":"2022-11-27T16:04:59.454375Z","iopub.status.idle":"2022-11-27T16:04:59.527267Z","shell.execute_reply.started":"2022-11-27T16:04:59.454350Z","shell.execute_reply":"2022-11-27T16:04:59.526105Z"},"jupyter":{"outputs_hidden":false}}
class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

BertLayerNorm = torch.nn.LayerNorm

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids)

class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        return super(RobertaModel, self).forward(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask)

## Our implementation of RobertaForTokenClassification
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        x = outputs[0]
        x = self.dropout(x)
        logits = self.classifier(x)
        return (logits, )

class MultiHeadRobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(MultiHeadRobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_neg = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_spec = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        x = outputs[0]
        x = self.dropout(x)
        logits_neg = self.classifier_neg(x)
        logits_spec = self.classifier_spec(x)
        return ((logits_neg, logits_spec), )

class MultiHeadBertForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(MultiHeadBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_neg = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_spec = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        x = outputs[0]
        x = self.dropout(x)
        logits_neg = self.classifier_neg(x)
        logits_spec = self.classifier_spec(x)
        return ((logits_neg, logits_spec), )

# %% [code] {"papermill":{"duration":0.01357,"end_time":"2022-11-26T23:58:06.692355","exception":false,"start_time":"2022-11-26T23:58:06.678785","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"1cKZOE2CKq32","papermill":{"duration":0.105391,"end_time":"2022-11-26T23:58:06.812107","exception":false,"start_time":"2022-11-26T23:58:06.706716","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.529332Z","iopub.execute_input":"2022-11-27T16:04:59.529847Z","iopub.status.idle":"2022-11-27T16:04:59.627948Z","shell.execute_reply.started":"2022-11-27T16:04:59.529809Z","shell.execute_reply":"2022-11-27T16:04:59.626865Z"},"jupyter":{"outputs_hidden":false}}
XLNetLayerNorm = nn.LayerNorm
class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head))

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3]-1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij', attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum('ijbn->bnij', attn_mask)

        # attention probability
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)

        # attention output
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)

        if self.output_attentions:
            return attn_vec, torch.einsum('bnij->ijbn', attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(self, h, g,
                      attn_mask_h, attn_mask_g,
                      r, seg_mat,
                      mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            ###### Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)

            # content-based value head
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # position-based key head
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            ##### h-stream
            # content-stream query head
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            ###### Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content heads
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # positional heads
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForward(nn.Module):
    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or \
                (sys.version_info[0] == 2 and isinstance(config.ff_activation, unicode)):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class XLNetLayer(nn.Module):
    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g,
                attn_mask_h, attn_mask_g,
                r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g,
                                r, seg_mat, mems=mems, target_mapping=target_mapping,
                                head_mask=head_mask)
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs


class XLNetPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLNetConfig
    pretrained_model_archive_map = XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, XLNetLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [module.q, module.k, module.v, module.o, module.r,
                          module.r_r_bias, module.r_s_bias, module.r_w_bias,
                          module.seg_embed]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel):
                module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)

class XLNetModel(XLNetPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **mems**: (`optional`, returned when ``config.mem_len > 0``)
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetModel.from_pretrained('xlnet-large-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    def __init__(self, config):
        super(XLNetModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_past = config.output_past

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.n_token, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        self.word_embedding = self._get_resized_embeddings(self.word_embedding, new_num_tokens)
        return self.word_embedding

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill
        ::
                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(next(self.parameters()))
        return ret

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[:self.reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-self.mem_len:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameters()))
        return pos_emb

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
        # else:  # We removed the inp_q input which was same as target mapping
        #     inp_q_ext = inp_q[:, :, None]
        #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len > 0 and self.output_past:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(),)

        if self.mem_len is not None and self.mem_len > 0 and self.output_past:
            outputs = outputs + (new_mems,)

        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, (new_mems), (hidden_states), (attentions)

## Our implementation of XLNetForTokenClassification
class XLNetForTokenClassification(XLNetPreTrainedModel):
  def __init__(self, config):
        super(XLNetForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)
        self.init_weights()

  def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask,
                                               head_mask=head_mask)

        output = transformer_outputs[0]
        output = self.dropout(output)
        logits = self.logits_proj(output)

        return (logits,)

class MultiHeadXLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super(MultiHeadXLNetForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.logits_proj_neg = nn.Linear(config.d_model, config.num_labels)
        self.logits_proj_spec = nn.Linear(config.d_model, config.num_labels)
        self.init_weights()
    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, labels=None):

        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask,
                                               head_mask=head_mask)

        output = transformer_outputs[0]
        output = self.dropout(output)
        logits_neg = self.logits_proj_neg(output)
        logits_spec = self.logits_proj_spec(output)
        return ((logits_neg, logits_spec),)

# %% [code] {"id":"ikdDG1vILFG6","papermill":{"duration":0.034077,"end_time":"2022-11-26T23:58:06.860444","exception":false,"start_time":"2022-11-26T23:58:06.826367","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.629450Z","iopub.execute_input":"2022-11-27T16:04:59.630476Z","iopub.status.idle":"2022-11-27T16:04:59.648369Z","shell.execute_reply.started":"2022-11-27T16:04:59.630439Z","shell.execute_reply":"2022-11-27T16:04:59.647425Z"},"jupyter":{"outputs_hidden":false}}
def f1_cues(y_true, y_pred):
    '''Needs flattened cues'''
    tp = sum([1 for i,j in zip(y_true, y_pred) if (i==j and i!=3)])
    fp = sum([1 for i,j in zip(y_true, y_pred) if (j!=3 and i==3)])
    fn = sum([1 for i,j in zip(y_true, y_pred) if (i!=3 and j==3)])
    if tp==0:
        prec = 0.0001
        rec = 0.0001
    else:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {2*prec*rec/(prec+rec)}")
    return prec, rec, 2*prec*rec/(prec+rec)


def f1_scope(y_true, y_pred, level = 'token'): #This is for gold cue annotation scope, thus the precision is always 1.
    if level == 'token':
        print(f1_score([i for i in j for j in y_true], [i for i in j for j in y_pred]))
    elif level == 'scope':
        tp = 0
        fn = 0
        fp = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                tp+=1
            else:
                fn+=1
        prec = 1
        rec = tp/(tp+fn)
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {2*prec*rec/(prec+rec)}")

def report_per_class_accuracy(y_true, y_pred):
    labels = list(np.unique(y_true))
    lab = list(np.unique(y_pred))
    labels = list(np.unique(labels+lab))
    n_labels = len(labels)
    data = pd.DataFrame(columns = labels, index = labels, data = np.zeros((n_labels, n_labels)))
    for i,j in zip(y_true, y_pred):
        data.at[i,j]+=1
    print(data)

def flat_accuracy(preds, labels, input_mask = None):
    pred_flat = [i for j in preds for i in j]
    labels_flat = [i for j in labels for i in j]
    return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)


def flat_accuracy_positive_cues(preds, labels, input_mask = None):
    pred_flat = [i for i,j in zip([i for j in preds for i in j],[i for j in labels for i in j]) if (j!=4 and j!=3)]
    labels_flat = [i for i in [i for j in labels for i in j] if (i!=4 and i!=3)]
    if len(labels_flat) != 0:
        return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    else:
        return None

def scope_accuracy(preds, labels):
    correct_count = 0
    count = 0
    for i,j in zip(preds, labels):
        if i==j:
            correct_count+=1
        count+=1
    return correct_count/count

# %% [code] {"id":"WFnVVirRY4mr","papermill":{"duration":0.025866,"end_time":"2022-11-26T23:58:06.900755","exception":false,"start_time":"2022-11-26T23:58:06.874889","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.649812Z","iopub.execute_input":"2022-11-27T16:04:59.650299Z","iopub.status.idle":"2022-11-27T16:04:59.663790Z","shell.execute_reply.started":"2022-11-27T16:04:59.650233Z","shell.execute_reply":"2022-11-27T16:04:59.662802Z"},"jupyter":{"outputs_hidden":false}}
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, save_path = 'Checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, score, model):

        #score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

# %% [code] {"id":"Mz9Wuug27JSc","papermill":{"duration":0.145953,"end_time":"2022-11-26T23:58:07.060702","exception":false,"start_time":"2022-11-26T23:58:06.914749","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.665902Z","iopub.execute_input":"2022-11-27T16:04:59.666465Z","iopub.status.idle":"2022-11-27T16:04:59.806260Z","shell.execute_reply.started":"2022-11-27T16:04:59.666430Z","shell.execute_reply":"2022-11-27T16:04:59.805321Z"},"jupyter":{"outputs_hidden":false}}
class CueModel_Combined:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = 'cuda', learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        if train == True:
            if 'xlnet' in CUE_MODEL:
                self.model = MultiHeadXLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')
                #self.model_2 = MultiHeadXLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')

            elif 'roberta' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')
                #self.model_2 = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')

            elif 'bert' in CUE_MODEL:
                self.model = MultiHeadBertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
                #self.model_2 = MultiHeadBertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
            elif 'cuenb' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)

            elif 'augnb' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)



            else:
                raise ValueError("Supported model types are: xlnet, roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        print("self.device==========",self.device)###########irf
        if device == 'cuda':
            print("yessssss CUDA")###########irf
            self.model.cuda()
            #self.model_2.cuda()
        else:
            self.model.cpu()
            #self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):

        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint.pt')
        #early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                loss = loss_neg + loss_spec
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions_neg , true_labels_neg, predictions_spec , true_labels_spec, ip_mask = [], [], [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(torch.from_numpy(np.asarray(t).astype('long')).to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

                    with torch.no_grad():
                        logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels_neg = b_labels_neg.view(-1)[active_loss]
                        active_labels_spec = b_labels_spec.view(-1)[active_loss]
                        tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                        tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                        tmp_eval_loss = (tmp_eval_loss_neg.mean().item()+tmp_eval_loss_spec.mean().item())/2

                    logits_neg = logits_neg.detach().cpu().numpy()
                    logits_spec = logits_spec.detach().cpu().numpy()
                    label_ids_neg = b_labels_neg.to('cpu').numpy()
                    label_ids_spec = b_labels_spec.to('cpu').numpy()

                    mymasks = b_mymasks.to('cpu').numpy()

                    logits_neg = [list(p) for p in logits_neg]
                    logits_spec = [list(p) for p in logits_spec]

                    actual_logits_neg = []
                    actual_label_ids_neg = []
                    actual_logits_spec = []
                    actual_label_ids_spec = []

                    for l_n,lid_n,l_s,lid_s,m in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks):

                        actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                        actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])

                        curr_preds_n = []
                        my_logits_n = []
                        curr_preds_s = []
                        my_logits_s = []
                        in_split = 0

                        for i_n, i_s, j in zip(l_n, l_s, m):
                            if j==1:
                                if in_split == 1:
                                    if len(my_logits_n)>0:
                                        curr_preds_n.append(my_logits_n[-1])
                                    mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                                    if len(my_logits_s)>0:
                                        curr_preds_s.append(my_logits_s[-1])
                                    mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                                    if len(my_logits_n)>0:
                                        my_logits_n[-1] = mode_pred_n
                                    else:
                                        my_logits_n.append(mode_pred_n)
                                    if len(my_logits_s)>0:
                                        my_logits_s[-1] = mode_pred_s
                                    else:
                                        my_logits_s.append(mode_pred_s)
                                    curr_preds_n = []
                                    curr_preds_s = []
                                    in_split = 0
                                my_logits_n.append(np.argmax(i_n))
                                my_logits_s.append(np.argmax(i_s))

                            if j==0:
                                curr_preds_n.append(i_n)
                                curr_preds_s.append(i_s)
                                in_split = 1
                        if in_split == 1:
                            if len(my_logits_n)>0:
                                curr_preds_n.append(my_logits_n[-1])
                            mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                            if len(my_logits_s)>0:
                                curr_preds_s.append(my_logits_s[-1])
                            mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                            if len(my_logits_n)>0:
                                my_logits_n[-1] = mode_pred_n
                            else:
                                my_logits_n.append(mode_pred_n)
                            if len(my_logits_s)>0:
                                my_logits_s[-1] = mode_pred_s
                            else:
                                my_logits_s.append(mode_pred_s)
                        actual_logits_neg.append(my_logits_n)
                        actual_logits_spec.append(my_logits_s)

                    logits_neg = actual_logits_neg
                    label_ids_neg = actual_label_ids_neg
                    logits_spec = actual_logits_spec
                    label_ids_spec = actual_label_ids_spec

                    predictions_neg.append(logits_neg)
                    true_labels_neg.append(label_ids_neg)
                    predictions_spec.append(logits_spec)
                    true_labels_spec.append(label_ids_spec)

                    tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
                    #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss
                    valid_loss.append(tmp_eval_loss)
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss/nb_eval_steps

            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
            pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
            labels_flat_neg = [l for l in labels_flat_neg if l!=4]
            labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
            pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
            labels_flat_spec = [l for l in labels_flat_spec if l!=4]
            report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
            report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
            print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
            labels_flat = labels_flat_neg + labels_flat_spec
            pred_flat = pred_flat_neg + pred_flat_spec
            p,r,f1 = f1_cues(labels_flat, pred_flat)
            #p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)

            if f1>return_dict['Best F1'] and early_stopping.early_stop == False:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            if early_stopping.early_stop == False:
                early_stopping(f1, self.model)
            else:
                print("Early stopping")
                break

            '''labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))'''

        self.model.load_state_dict(torch.load('checkpoint.pt'))
        #self.model_2.load_state_dict(torch.load('checkpoint2.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Negation - Precision": 0,
                       "Negation - Recall": 0,
                       "Negation - F1": 0,
                       "Speculation - Precision": 0,
                       "Speculation - Recall": 0,
                       "Speculation - F1": 0}
        self.model.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions_neg, true_labels_neg, predictions_spec, true_labels_spec, ip_mask = [], [], [], [], []
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

            with torch.no_grad():
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                #_, logits_spec = self.model_2(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                tmp_eval_loss = tmp_eval_loss_neg+tmp_eval_loss_spec
                logits_neg = logits_neg.detach().cpu().numpy()
                logits_spec = logits_spec.detach().cpu().numpy()

            label_ids_neg = b_labels_neg.to('cpu').numpy()
            label_ids_spec = b_labels_spec.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()
            logits_neg = [list(p) for p in logits_neg]
            logits_spec = [list(p) for p in logits_spec]

            actual_logits_neg = []
            actual_label_ids_neg = []
            actual_logits_spec = []
            actual_label_ids_spec = []

            for l_n,lid_n,l_s,lid_s,m in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks):

                actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])

                curr_preds_n = []
                my_logits_n = []
                curr_preds_s = []
                my_logits_s = []
                in_split = 0

                for i_n, i_s, j in zip(l_n, l_s, m):
                    if j==1:
                        if in_split == 1:
                            if len(my_logits_n)>0:
                                curr_preds_n.append(my_logits_n[-1])
                            mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                            if len(my_logits_s)>0:
                                curr_preds_s.append(my_logits_s[-1])
                            mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                            if len(my_logits_n)>0:
                                my_logits_n[-1] = mode_pred_n
                            else:
                                my_logits_n.append(mode_pred_n)
                            if len(my_logits_s)>0:
                                my_logits_s[-1] = mode_pred_s
                            else:
                                my_logits_s.append(mode_pred_s)
                            curr_preds_n = []
                            curr_preds_s = []
                            in_split = 0
                        my_logits_n.append(np.argmax(i_n))
                        my_logits_s.append(np.argmax(i_s))

                    if j==0:
                        curr_preds_n.append(i_n)
                        curr_preds_s.append(i_s)
                        in_split = 1
                if in_split == 1:
                    if len(my_logits_n)>0:
                        curr_preds_n.append(my_logits_n[-1])
                    mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                    if len(my_logits_s)>0:
                        curr_preds_s.append(my_logits_s[-1])
                    mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                    if len(my_logits_n)>0:
                        my_logits_n[-1] = mode_pred_n
                    else:
                        my_logits_n.append(mode_pred_n)
                    if len(my_logits_s)>0:
                        my_logits_s[-1] = mode_pred_s
                    else:
                        my_logits_s.append(mode_pred_s)
                actual_logits_neg.append(my_logits_n)
                actual_logits_spec.append(my_logits_s)

            logits_neg = actual_logits_neg
            label_ids_neg = actual_label_ids_neg
            logits_spec = actual_logits_spec
            label_ids_spec = actual_label_ids_spec

            predictions_neg.append(logits_neg)
            true_labels_neg.append(label_ids_neg)
            predictions_spec.append(logits_spec)
            true_labels_spec.append(label_ids_spec)

            tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
            #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
            eval_loss += tmp_eval_loss
            #valid_loss.append(tmp_eval_loss)
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
        pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
        pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
        labels_flat_neg = [l for l in labels_flat_neg if l!=4]
        report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
        labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
        pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
        pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
        labels_flat_spec = [l for l in labels_flat_spec if l!=4]
        report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
        print(classification_report(labels_flat_neg, pred_flat_neg))
        print(classification_report(labels_flat_spec, pred_flat_spec))
        print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
        print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
        p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
        p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)
        return_dict['Negation - F1'] = f1_n
        return_dict['Negation - Precision'] = p_n
        return_dict['Negation - Recall'] = r_n
        return_dict['Speculation - F1'] = f1_s
        return_dict['Speculation - Precision'] = p_s
        return_dict['Speculation - Recall'] = r_s

        return return_dict

class ScopeModel_Combined:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = 'cuda', learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = SUBTASK
        self.num_labels = 2
        self.scope_method = SCOPE_METHOD
        if train == True:
            if 'xlnet' in SCOPE_MODEL:
                self.model = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')
                #self.model_2 = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')

            elif 'roberta' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')
                #self.model_2 = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')

            elif 'bert' in SCOPE_MODEL:
                self.model = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
                #self.model_2 = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
            elif 'cuenb' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)

            elif 'augnb' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)


            else:
                raise ValueError("Supported model types are: xlnet, roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
            #self.model_2.cuda()

        else:
            self.model.cpu()
            #self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def train(self, train_dataloader, valid_dataloader_negation, valid_dataloader_speculation, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multitask Scope Resolution - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0,
                       }
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint.pt')
        #early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')

        loss_fn = CrossEntropyLoss()
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            self.model.eval()

            eval_loss_neg, eval_accuracy_neg, eval_scope_accuracy_neg = 0, 0, 0
            nb_eval_steps_neg, nb_eval_examples_neg = 0, 0
            predictions_negation , true_labels_negation, ip_mask_neg = [], [], []
            for batch in valid_dataloader_negation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()

                logits = [list(p) for p in logits]

                actual_logits = []
                actual_label_ids = []

                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m, b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                predictions_negation.append(actual_logits)
                true_labels_negation.append(actual_label_ids)

                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_neg += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_neg += tmp_eval_loss.mean().item()
                eval_accuracy_neg += tmp_eval_accuracy

                nb_eval_examples_neg += len(b_input_ids)
                nb_eval_steps_neg += 1

            eval_loss_neg = eval_loss_neg/nb_eval_steps_neg
            print("Negation Validation loss: {}".format(eval_loss_neg))
            print("Negation Validation Accuracy: {}".format(eval_accuracy_neg/nb_eval_steps_neg))
            print("Negation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_neg/nb_eval_steps_neg))
            f1_scope([j for i in true_labels_negation for j in i], [j for i in predictions_negation for j in i], level='scope')
            labels_flat_neg = [l_ii for l in true_labels_negation for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_negation for p_i in p for p_ii in p_i]

            #Speculation
            eval_loss_spec, eval_accuracy_spec, eval_scope_accuracy_spec = 0, 0, 0
            nb_eval_steps_spec, nb_eval_examples_spec = 0, 0
            predictions_speculation , true_labels_speculation, ip_mask = [], [], []
            for batch in valid_dataloader_speculation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()

                logits = [list(p) for p in logits]

                actual_logits = []
                actual_label_ids = []

                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m, b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                predictions_speculation.append(actual_logits)
                true_labels_speculation.append(actual_label_ids)

                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_spec += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_spec += tmp_eval_loss.mean().item()
                eval_accuracy_spec += tmp_eval_accuracy

                nb_eval_examples_spec += len(b_input_ids)
                nb_eval_steps_spec += 1

            eval_loss_spec = eval_loss_spec/nb_eval_steps_spec
            print("Speculation Validation loss: {}".format(eval_loss_spec))
            print("Speculation Validation Accuracy: {}".format(eval_accuracy_spec/nb_eval_steps_spec))
            print("Speculation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_spec/nb_eval_steps_spec))
            f1_scope([j for i in true_labels_speculation for j in i], [j for i in predictions_speculation for j in i], level='scope')
            labels_flat_spec = [l_ii for l in true_labels_speculation for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_speculation for p_i in p for p_ii in p_i]
            labels_flat = labels_flat_neg + labels_flat_spec
            pred_flat = pred_flat_neg + pred_flat_spec
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Best F1'] and early_stopping.early_stop == False:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping.early_stop == False:
                early_stopping(f1, self.model)
            else:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load('checkpoint.pt'))
        #self.model_2.load_state_dict(torch.load('checkpoint2.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name = "SFU", task = "Negation"):
        return_dict = {"Task": f"Multitask Scope Resolution - {task} - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()

            logits = [list(p) for p in logits]

            actual_logits = []
            actual_label_ids = []

            for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                my_logits = []
                curr_preds = []
                in_split = 0
                for i,j,k in zip(l,m,b_ii):
                    '''if k == 0:
                        break'''
                    if j==1:
                        if in_split == 1:
                            if len(my_logits)>0:
                                curr_preds.append(my_logits[-1])
                            mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                            if len(my_logits)>0:
                                my_logits[-1] = mode_pred
                            else:
                                my_logits.append(mode_pred)
                            curr_preds = []
                            in_split = 0
                        my_logits.append(np.argmax(i))
                    if j==0:
                        curr_preds.append(i)
                        in_split = 1
                if in_split == 1:
                    if len(my_logits)>0:
                        curr_preds.append(my_logits[-1])
                    mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                    if len(my_logits)>0:
                        my_logits[-1] = mode_pred
                    else:
                        my_logits.append(mode_pred)
                actual_logits.append(my_logits)

            predictions.append(actual_logits)
            true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        return return_dict

# %% [code] {"papermill":{"duration":0.021352,"end_time":"2022-11-26T23:58:07.096378","exception":false,"start_time":"2022-11-26T23:58:07.075026","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.809076Z","iopub.execute_input":"2022-11-27T16:04:59.809574Z","iopub.status.idle":"2022-11-27T16:04:59.813561Z","shell.execute_reply.started":"2022-11-27T16:04:59.809546Z","shell.execute_reply":"2022-11-27T16:04:59.812594Z"},"jupyter":{"outputs_hidden":false}}
global test

# %% [code] {"id":"gNkZC5AILMQY","papermill":{"duration":0.42108,"end_time":"2022-11-26T23:58:07.531305","exception":false,"start_time":"2022-11-26T23:58:07.110225","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.815681Z","iopub.execute_input":"2022-11-27T16:04:59.816426Z","iopub.status.idle":"2022-11-27T16:04:59.958873Z","shell.execute_reply.started":"2022-11-27T16:04:59.816369Z","shell.execute_reply":"2022-11-27T16:04:59.957916Z"},"jupyter":{"outputs_hidden":false}}
class CueModel_Separate:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = 'cuda', learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        if train == True:
            if 'xlnet' in CUE_MODEL:
                self.model = MultiHeadXLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')
                self.model_2 = MultiHeadXLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')

            elif 'roberta' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')
                self.model_2 = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')

            elif 'bert' in CUE_MODEL:
                self.model = MultiHeadBertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
                self.model_2 = MultiHeadBertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
                test = self.model
            elif 'cuenb' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)
                self.model_2 = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)
            elif 'augnb' in CUE_MODEL:
                self.model = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)
                self.model_2 = MultiHeadRobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels)
            else:
                raise ValueError("Supported model types are: xlnet, roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        if device == 'cuda':
            self.model.cuda()
            self.model_2.cuda()
        else:
            self.model.cpu()
            self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):

        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Negation - Best Precision": 0,
                       "Negation - Best Recall": 0,
                       "Negation - Best F1": 0,
                       "Speculation - Best Precision": 0,
                       "Speculation - Best Recall": 0,
                       "Speculation - Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping_neg = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint.pt')
        early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                loss = loss_neg + loss_spec
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions_neg , true_labels_neg, predictions_spec , true_labels_spec, ip_mask = [], [], [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

                    with torch.no_grad():
                        logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels_neg = b_labels_neg.view(-1)[active_loss]
                        active_labels_spec = b_labels_spec.view(-1)[active_loss]
                        tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                        tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                        tmp_eval_loss = (tmp_eval_loss_neg.mean().item()+tmp_eval_loss_spec.mean().item())/2

                    logits_neg = logits_neg.detach().cpu().numpy()
                    logits_spec = logits_spec.detach().cpu().numpy()
                    label_ids_neg = b_labels_neg.to('cpu').numpy()
                    label_ids_spec = b_labels_spec.to('cpu').numpy()

                    mymasks = b_mymasks.to('cpu').numpy()

                    logits_neg = [list(p) for p in logits_neg]
                    logits_spec = [list(p) for p in logits_spec]

                    actual_logits_neg = []
                    actual_label_ids_neg = []
                    actual_logits_spec = []
                    actual_label_ids_spec = []

                    for l_n,lid_n,l_s,lid_s,m in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks):

                        actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                        actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])

                        curr_preds_n = []
                        my_logits_n = []
                        curr_preds_s = []
                        my_logits_s = []
                        in_split = 0

                        for i_n, i_s, j in zip(l_n, l_s, m):
                            if j==1:
                                if in_split == 1:
                                    if len(my_logits_n)>0:
                                        curr_preds_n.append(my_logits_n[-1])
                                    mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                                    if len(my_logits_s)>0:
                                        curr_preds_s.append(my_logits_s[-1])
                                    mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                                    if len(my_logits_n)>0:
                                        my_logits_n[-1] = mode_pred_n
                                    else:
                                        my_logits_n.append(mode_pred_n)
                                    if len(my_logits_s)>0:
                                        my_logits_s[-1] = mode_pred_s
                                    else:
                                        my_logits_s.append(mode_pred_s)
                                    curr_preds_n = []
                                    curr_preds_s = []
                                    in_split = 0
                                my_logits_n.append(np.argmax(i_n))
                                my_logits_s.append(np.argmax(i_s))

                            if j==0:
                                curr_preds_n.append(i_n)
                                curr_preds_s.append(i_s)
                                in_split = 1
                        if in_split == 1:
                            if len(my_logits_n)>0:
                                curr_preds_n.append(my_logits_n[-1])
                            mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                            if len(my_logits_s)>0:
                                curr_preds_s.append(my_logits_s[-1])
                            mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                            if len(my_logits_n)>0:
                                my_logits_n[-1] = mode_pred_n
                            else:
                                my_logits_n.append(mode_pred_n)
                            if len(my_logits_s)>0:
                                my_logits_s[-1] = mode_pred_s
                            else:
                                my_logits_s.append(mode_pred_s)
                        actual_logits_neg.append(my_logits_n)
                        actual_logits_spec.append(my_logits_s)

                    logits_neg = actual_logits_neg
                    label_ids_neg = actual_label_ids_neg
                    logits_spec = actual_logits_spec
                    label_ids_spec = actual_label_ids_spec

                    predictions_neg.append(logits_neg)
                    true_labels_neg.append(label_ids_neg)
                    predictions_spec.append(logits_spec)
                    true_labels_spec.append(label_ids_spec)

                    tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
                    #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss
                    valid_loss.append(tmp_eval_loss)
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss/nb_eval_steps

            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
            pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
            labels_flat_neg = [l for l in labels_flat_neg if l!=4]
            labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
            pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
            labels_flat_spec = [l for l in labels_flat_spec if l!=4]
            report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
            report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
            print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
            p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
            p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)

            if f1_n>return_dict['Negation - Best F1'] and early_stopping_neg.early_stop == False:
                return_dict['Negation - Best F1'] = f1_n
                return_dict['Negation - Best Precision'] = p_n
                return_dict['Negation - Best Recall'] = r_n
            if early_stopping_neg.early_stop == False:
                early_stopping_neg(f1_n, self.model)
            if f1_s>return_dict['Speculation - Best F1'] and early_stopping_spec.early_stop == False:
                return_dict['Speculation - Best F1'] = f1_s
                return_dict['Speculation - Best Precision'] = p_s
                return_dict['Speculation - Best Recall'] = r_s
            if early_stopping_spec.early_stop == False:
                early_stopping_spec(f1_s, self.model)

            if early_stopping_neg.early_stop and early_stopping_spec.early_stop:
                print("Early stopping")
                break

            '''labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))'''

        self.model.load_state_dict(torch.load('checkpoint.pt'))
        self.model_2.load_state_dict(torch.load('checkpoint2.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Negation - Precision": 0,
                       "Negation - Recall": 0,
                       "Negation - F1": 0,
                       "Speculation - Precision": 0,
                       "Speculation - Recall": 0,
                       "Speculation - F1": 0}
        self.model.eval()
        self.model_2.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions_neg, true_labels_neg, predictions_spec, true_labels_spec, ip_mask = [], [], [], [], []
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

            with torch.no_grad():
                logits_neg, _ = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                _, logits_spec = self.model_2(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                tmp_eval_loss = tmp_eval_loss_neg+tmp_eval_loss_spec
                logits_neg = logits_neg.detach().cpu().numpy()
                logits_spec = logits_spec.detach().cpu().numpy()

            label_ids_neg = b_labels_neg.to('cpu').numpy()
            label_ids_spec = b_labels_spec.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()
            logits_neg = [list(p) for p in logits_neg]
            logits_spec = [list(p) for p in logits_spec]

            actual_logits_neg = []
            actual_label_ids_neg = []
            actual_logits_spec = []
            actual_label_ids_spec = []

            for l_n,lid_n,l_s,lid_s,m in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks):

                actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])

                curr_preds_n = []
                my_logits_n = []
                curr_preds_s = []
                my_logits_s = []
                in_split = 0

                for i_n, i_s, j in zip(l_n, l_s, m):
                    if j==1:
                        if in_split == 1:
                            if len(my_logits_n)>0:
                                curr_preds_n.append(my_logits_n[-1])
                            mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                            if len(my_logits_s)>0:
                                curr_preds_s.append(my_logits_s[-1])
                            mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                            if len(my_logits_n)>0:
                                my_logits_n[-1] = mode_pred_n
                            else:
                                my_logits_n.append(mode_pred_n)
                            if len(my_logits_s)>0:
                                my_logits_s[-1] = mode_pred_s
                            else:
                                my_logits_s.append(mode_pred_s)
                            curr_preds_n = []
                            curr_preds_s = []
                            in_split = 0
                        my_logits_n.append(np.argmax(i_n))
                        my_logits_s.append(np.argmax(i_s))

                    if j==0:
                        curr_preds_n.append(i_n)
                        curr_preds_s.append(i_s)
                        in_split = 1
                if in_split == 1:
                    if len(my_logits_n)>0:
                        curr_preds_n.append(my_logits_n[-1])
                    mode_pred_n = np.argmax(np.average(np.array(curr_preds_n), axis=0), axis=0)
                    if len(my_logits_s)>0:
                        curr_preds_s.append(my_logits_s[-1])
                    mode_pred_s = np.argmax(np.average(np.array(curr_preds_s), axis=0), axis=0)
                    if len(my_logits_n)>0:
                        my_logits_n[-1] = mode_pred_n
                    else:
                        my_logits_n.append(mode_pred_n)
                    if len(my_logits_s)>0:
                        my_logits_s[-1] = mode_pred_s
                    else:
                        my_logits_s.append(mode_pred_s)
                actual_logits_neg.append(my_logits_n)
                actual_logits_spec.append(my_logits_s)

            logits_neg = actual_logits_neg
            label_ids_neg = actual_label_ids_neg
            logits_spec = actual_logits_spec
            label_ids_spec = actual_label_ids_spec

            predictions_neg.append(logits_neg)
            true_labels_neg.append(label_ids_neg)
            predictions_spec.append(logits_spec)
            true_labels_spec.append(label_ids_spec)

            tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
            #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
            eval_loss += tmp_eval_loss
            #valid_loss.append(tmp_eval_loss)
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
        pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
        pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
        labels_flat_neg = [l for l in labels_flat_neg if l!=4]
        report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
        labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
        pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
        pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
        labels_flat_spec = [l for l in labels_flat_spec if l!=4]
        report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
        print(classification_report(labels_flat_neg, pred_flat_neg))
        print(classification_report(labels_flat_spec, pred_flat_spec))
        print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
        print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
        p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
        p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)
        return_dict['Negation - F1'] = f1_n
        return_dict['Negation - Precision'] = p_n
        return_dict['Negation - Recall'] = r_n
        return_dict['Speculation - F1'] = f1_s
        return_dict['Speculation - Precision'] = p_s
        return_dict['Speculation - Recall'] = r_s
        return return_dict

class ScopeModel_Separate:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = 'cuda', learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = SUBTASK
        self.num_labels = 2
        self.scope_method = SCOPE_METHOD
        if train == True:
            if 'xlnet' in SCOPE_MODEL:
                self.model = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')
                self.model_2 = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')

            elif 'roberta' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')
                self.model_2 = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')

            elif 'bert' in SCOPE_MODEL:
                self.model = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
                self.model_2 = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
            elif 'cuenb' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)
                self.model_2 = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)
            elif 'augnb' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)
                self.model_2 = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels)

            else:
                raise ValueError("Supported model types are: xlnet, roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
            self.model_2.cuda()

        else:
            self.model.cpu()
            self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def train(self, train_dataloader, valid_dataloader_negation, valid_dataloader_speculation, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multitask Scope Resolution - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Negation - Best Precision": 0,
                       "Negation - Best Recall": 0,
                       "Negation - Best F1": 0,
                       "Speculation - Best Precision": 0,
                       "Speculation - Best Recall": 0,
                       "Speculation - Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping_neg = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint.pt')
        early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')

        loss_fn = CrossEntropyLoss()
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            self.model.eval()

            eval_loss_neg, eval_accuracy_neg, eval_scope_accuracy_neg = 0, 0, 0
            nb_eval_steps_neg, nb_eval_examples_neg = 0, 0
            predictions_negation , true_labels_negation, ip_mask_neg = [], [], []
            for batch in valid_dataloader_negation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()

                logits = [list(p) for p in logits]

                actual_logits = []
                actual_label_ids = []

                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m, b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                predictions_negation.append(actual_logits)
                true_labels_negation.append(actual_label_ids)

                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_neg += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_neg += tmp_eval_loss.mean().item()
                eval_accuracy_neg += tmp_eval_accuracy

                nb_eval_examples_neg += len(b_input_ids)
                nb_eval_steps_neg += 1

            eval_loss_neg = eval_loss_neg/nb_eval_steps_neg
            print("Negation Validation loss: {}".format(eval_loss_neg))
            print("Negation Validation Accuracy: {}".format(eval_accuracy_neg/nb_eval_steps_neg))
            print("Negation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_neg/nb_eval_steps_neg))
            f1_scope([j for i in true_labels_negation for j in i], [j for i in predictions_negation for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels_negation for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions_negation for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Negation - Best F1'] and early_stopping_neg.early_stop == False:
                return_dict['Negation - Best F1'] = f1
                return_dict['Negation - Best Precision'] = p
                return_dict['Negation - Best Recall'] = r
            print("Negation: F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping_neg.early_stop == False:
                early_stopping_neg(f1, self.model)

            #Speculation
            eval_loss_spec, eval_accuracy_spec, eval_scope_accuracy_spec = 0, 0, 0
            nb_eval_steps_spec, nb_eval_examples_spec = 0, 0
            predictions_speculation , true_labels_speculation, ip_mask = [], [], []
            for batch in valid_dataloader_speculation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()

                logits = [list(p) for p in logits]

                actual_logits = []
                actual_label_ids = []

                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m, b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                predictions_speculation.append(actual_logits)
                true_labels_speculation.append(actual_label_ids)

                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_spec += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_spec += tmp_eval_loss.mean().item()
                eval_accuracy_spec += tmp_eval_accuracy

                nb_eval_examples_spec += len(b_input_ids)
                nb_eval_steps_spec += 1

            eval_loss_spec = eval_loss_spec/nb_eval_steps_spec
            print("Speculation Validation loss: {}".format(eval_loss_spec))
            print("Speculation Validation Accuracy: {}".format(eval_accuracy_spec/nb_eval_steps_spec))
            print("Speculation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_spec/nb_eval_steps_spec))
            f1_scope([j for i in true_labels_speculation for j in i], [j for i in predictions_speculation for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels_speculation for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions_speculation for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Speculation - Best F1'] and early_stopping_spec.early_stop == False:
                return_dict['Speculation - Best F1'] = f1
                return_dict['Speculation - Best Precision'] = p
                return_dict['Speculation - Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping_spec.early_stop == False:
                early_stopping_spec(f1, self.model)
            if early_stopping_neg.early_stop and early_stopping_spec.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load('checkpoint.pt'))
        self.model_2.load_state_dict(torch.load('checkpoint2.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict


    #@telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name = "SFU", task = "negation"):
        return_dict = {"Task": f"Multitask Separate Scope Resolution - {task} - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        self.model_2.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            #batch = torch.stack(batch, dim=0)
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                if task == 'negation':
                    logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                else:
                    logits = self.model_2(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()

            logits = [list(p) for p in logits]

            actual_logits = []
            actual_label_ids = []

            for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):

                actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                my_logits = []
                curr_preds = []
                in_split = 0
                for i,j,k in zip(l,m,b_ii):
                    '''if k == 0:
                        break'''
                    if j==1:
                        if in_split == 1:
                            if len(my_logits)>0:
                                curr_preds.append(my_logits[-1])
                            mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                            if len(my_logits)>0:
                                my_logits[-1] = mode_pred
                            else:
                                my_logits.append(mode_pred)
                            curr_preds = []
                            in_split = 0
                        my_logits.append(np.argmax(i))
                    if j==0:
                        curr_preds.append(i)
                        in_split = 1
                if in_split == 1:
                    if len(my_logits)>0:
                        curr_preds.append(my_logits[-1])
                    mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                    if len(my_logits)>0:
                        my_logits[-1] = mode_pred
                    else:
                        my_logits.append(mode_pred)
                actual_logits.append(my_logits)

            predictions.append(actual_logits)
            true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        return return_dict


        #for step,batch in enumerate(test_dataloader):

            #labels = np.asarray(labels)
            #labels = torch.from_numpy(labels.astype('long'))
            #batch = np.asarray(batch)
            #batch = torch.from_numpy(batch)

            #batch = torch.tensor(batch).to(self.device).long()

            #batch = batch.to(self.device).long() #'list' object has no attribute 'to'
            #batch = torch.tensor(batch).to(self.device) #only one element tensors can be converted to Python scalars
            #batch = tuple(t.to('cpu') for t in batch) #Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper__index_select)
            #batch = tuple(t.to(self.device) for t in batch) #'list' object has no attribute 'to'
            #batch = [t.to(self.device) for t in batch] #'list' object has no attribute 'to'
            #batch = torch.as_tensor([t for t in batch]) #only one element tensors can be converted to Python scalars

            #batch = tuple(torch.tensor(t).to(self.device) for t in batch)#only one element tensors can be converted to Python scalars
            #print("batch: \n")

            #batch = tuple(torch.tensor(t).to('cpu').numpy().detach() for t in batch)

            #batch = tuple(torch.tensor(t).to('cpu').numpy() for t in batch)
            #.numpy()

            #loss.cpu().detach().numpy()
            #batch = tuple(t.to(self.device) for t in batch)
           # b_input_ids, b_input_mask, b_labels, b_mymasks = batch
            #b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

            #b_input_ids = np.asarray(b_input_ids)
            #b_input_ids = torch.from_numpy(b_input_ids.astype('long'))
            ##b_input_ids = torch.tensor(b_input_ids).to(self.device).long()

            #b_input_mask = np.asarray(b_input_mask)
            #b_input_mask = torch.from_numpy(b_input_mask.astype('long'))
            ##b_input_mask = torch.tensor(b_input_mask).to(self.device).long()

            #b_labels = np.asarray(b_labels)
            #b_labels = torch.from_numpy(b_labels.astype('long'))
            ##b_labels = torch.tensor(b_labels).to(self.device).long()

            #b_labels_neg = torch.tensor(b_labels_neg).to(self.device).long()
            #b_labels_spec = torch.tensor(b_labels_spec).to(self.device).long()

            #b_mymasks = np.asarray(b_mymasks)
            #b_mymasks = torch.from_numpy(b_mymasks.astype('long'))
            ##b_mymasks = torch.tensor(b_mymasks).to(self.device).long()

# %% [code] {"id":"26maAzM7aENL","papermill":{"duration":0.628018,"end_time":"2022-11-26T23:58:08.173784","exception":false,"start_time":"2022-11-26T23:58:07.545766","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:04:59.960277Z","iopub.execute_input":"2022-11-27T16:04:59.960756Z","iopub.status.idle":"2022-11-27T16:05:00.287650Z","shell.execute_reply.started":"2022-11-27T16:04:59.960719Z","shell.execute_reply":"2022-11-27T16:05:00.286664Z"},"jupyter":{"outputs_hidden":false}}
#bioscope_full_papers_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
#sfu_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='sfu', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
#bioscope_abstracts_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)


bioscope_full_papers_data= Data('../input/biofullpapers/full_papers.xml', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
#bioscope_full_papers_data= Data('../input/fullpaper/full_papers.xml', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
bioscope_abstracts_data = Data('../input/bio-abstract/abstracts.xml', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
#bioscope_abstracts_data = Data('../input/abstracts/abstracts.xml', dataset_name='bioscope', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)
sfu_data = Data('/kaggle/input/sfu-review-corpus/SFU_Review_Corpus_Negation_Speculation', dataset_name='sfu', error_analysis=ERROR_ANALYSIS_FOR_SCOPE)

# %% [code] {"id":"wqp-LdMXeDyg","papermill":{"duration":1086.562107,"end_time":"2022-11-27T00:16:14.750362","exception":false,"start_time":"2022-11-26T23:58:08.188255","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-11-27T16:05:00.289275Z","iopub.execute_input":"2022-11-27T16:05:00.289702Z","iopub.status.idle":"2022-11-27T16:37:18.920929Z","shell.execute_reply.started":"2022-11-27T16:05:00.289664Z","shell.execute_reply":"2022-11-27T16:37:18.919784Z"},"jupyter":{"outputs_hidden":false}}
for run_num in range(NUM_RUNS):
    first_dataset = None
    other_datasets = []
    if 'sfu' in TRAIN_DATASETS:
        first_dataset = sfu_data
    if 'bioscope_full_papers' in TRAIN_DATASETS:
        if first_dataset == None:
            first_dataset = bioscope_full_papers_data
        else:
            other_datasets.append(bioscope_full_papers_data)
        print("first_dataset")
        print(first_dataset)
        print("....")

    if 'bioscope_abstracts' in TRAIN_DATASETS:
        if first_dataset == None:
            first_dataset = bioscope_abstracts_data
        else:
            other_datasets.append(bioscope_abstracts_data)

    if SUBTASK == 'cue_detection':
        train_dl, val_dls, test_dls = first_dataset.get_cue_dataloader(other_datasets = other_datasets)

        test_dataloaders = {}
        idx = 0
        if 'sfu' in TRAIN_DATASETS:
            if 'sfu' in TEST_DATASETS:
                test_dataloaders['sfu'] = test_dls[idx]
            idx+=1
        elif 'sfu' in TEST_DATASETS:
            sfu_dl, _, _ = sfu_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)

            test_dataloaders['sfu'] = sfu_dl
        if 'bioscope_full_papers' in TRAIN_DATASETS:
            if 'bioscope_full_papers' in TEST_DATASETS:
                test_dataloaders['bioscope_full_papers'] = test_dls[idx]
            idx+=1
        elif 'bioscope_full_papers' in TEST_DATASETS:
            bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
            test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
            #print(bioscope_full_papers_dl)
            #print(_)
        if 'bioscope_abstracts' in TRAIN_DATASETS:
            if 'bioscope_abstracts' in TEST_DATASETS:
                test_dataloaders['bioscope_abstracts'] = test_dls[idx]
            idx+=1
        elif 'bioscope_abstracts' in TEST_DATASETS:
            bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
            test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
        if EARLY_STOPPING_METHOD == 'separate':
            model = CueModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        elif EARLY_STOPPING_METHOD == 'combined':
            model = CueModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        else:
            raise ValueError("EARLY_STOPPING_METHOD must be one of 'separate' and 'combined'")
        print('model chosen====',model.model_name," SUBTASK=",SUBTASK) #####irf
        print('Train dataset====',TRAIN_DATASETS," Test dataset=", TEST_DATASETS) #####irf

        model.train(train_dl, val_dls, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS))
        for k in test_dataloaders.keys():
            print(f"Evaluate on {k}:")
            result_dict = model.evaluate(test_dataloaders[k], test_dl_name = k)
            store_result(result_dict, run_num)


    elif SUBTASK == 'scope_resolution':

        train_dl, [neg_val_dl, spec_val_dl], [neg_test_dls, spec_test_dls] = first_dataset.get_scope_dataloader(other_datasets = other_datasets)
        #first_dataset.get_scope_dataloader(other_datasets = other_datasets)
        #print(train_dl[0])
        #print(neg_val_dl)
        #print(spec_val_dl)

        neg_test_dataloaders = {}
        spec_test_dataloaders = {}
        neg_punct_test_dataloaders = {}
        spec_punct_test_dataloaders = {}
        neg_no_punct_test_dataloaders = {}
        spec_no_punct_test_dataloaders = {}
        idx = 0
        if 'sfu' in TRAIN_DATASETS:
            if 'sfu' in TEST_DATASETS:
                neg_test_dataloaders['sfu'] = neg_test_dls[idx]
                spec_test_dataloaders['sfu'] = spec_test_dls[idx]
            idx+=1
        elif 'sfu' in TEST_DATASETS:
            _, _, [neg_sfu_dl, spec_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
            neg_test_dataloaders['sfu'] = neg_sfu_dl[0]
            spec_test_dataloaders['sfu'] = spec_sfu_dl[0]
        if 'bioscope_full_papers' in TRAIN_DATASETS:
            if 'bioscope_full_papers' in TEST_DATASETS:
                neg_test_dataloaders['bioscope_full_papers'] = neg_test_dls[idx]
                spec_test_dataloaders['bioscope_full_papers'] = spec_test_dls[idx]
            idx+=1
        elif 'bioscope_full_papers' in TEST_DATASETS:
            _, _, [neg_bioscope_full_papers_dl, spec_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
            neg_test_dataloaders['bioscope_full_papers'] = neg_bioscope_full_papers_dl[0]
            spec_test_dataloaders['bioscope_full_papers'] = spec_bioscope_full_papers_dl[0]
        if 'bioscope_abstracts' in TRAIN_DATASETS:
            if 'bioscope_abstracts' in TEST_DATASETS:
                neg_test_dataloaders['bioscope_abstracts'] = neg_test_dls[idx]
                spec_test_dataloaders['bioscope_abstracts'] = spec_test_dls[idx]
            idx+=1
        elif 'bioscope_abstracts' in TEST_DATASETS:
            _, _, [neg_bioscope_abstracts_dl, spec_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.99999999, val_size = 0.00000001)
            neg_test_dataloaders['bioscope_abstracts'] = neg_bioscope_abstracts_dl[0]
            spec_test_dataloaders['bioscope_abstracts'] = spec_bioscope_abstracts_dl[0]

        # Error Analysis
        if 'sfu' in TEST_DATASETS:
            _, _, [neg_punct_sfu_dl, spec_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
            _, _, [neg_no_punct_sfu_dl, neg_no_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
            neg_punct_test_dataloaders['sfu_punct'] = neg_punct_sfu_dl
            spec_punct_test_dataloaders['sfu_punct'] = spec_punct_sfu_dl
            neg_no_punct_test_dataloaders['sfu_no_punct'] = neg_no_punct_sfu_dl
            spec_no_punct_test_dataloaders['sfu_no_punct'] = neg_no_punct_sfu_dl
        if 'bioscope_full_papers' in TEST_DATASETS:

            _, _, [neg_punct_bioscope_full_papers_dl, spec_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
            _, _, [neg_no_punct_bioscope_full_papers_dl, neg_no_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
            neg_punct_test_dataloaders['bioscope_full_papers_punct'] = neg_punct_bioscope_full_papers_dl
            spec_punct_test_dataloaders['bioscope_full_papers_punct'] = spec_punct_bioscope_full_papers_dl
            neg_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = neg_no_punct_bioscope_full_papers_dl
            spec_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = neg_no_punct_bioscope_full_papers_dl
        if 'bioscope_abstracts' in TEST_DATASETS:
            _, _, [neg_punct_bioscope_abstracts_dl, spec_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
            _, _, [neg_no_punct_bioscope_abstracts_dl, neg_no_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
            neg_punct_test_dataloaders['bioscope_abstracts_punct'] = neg_punct_bioscope_abstracts_dl
            spec_punct_test_dataloaders['bioscope_abstracts_punct'] = spec_punct_bioscope_abstracts_dl
            neg_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = neg_no_punct_bioscope_abstracts_dl
            spec_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = neg_no_punct_bioscope_abstracts_dl


        if EARLY_STOPPING_METHOD == 'separate':
            model = ScopeModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        elif EARLY_STOPPING_METHOD == 'combined':
            model = ScopeModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        else:
            raise ValueError("EARLY_STOPPING_METHOD must be one of 'separate' and 'combined'")


        print('model chosen====',model.model_name," SUBTASK=",SUBTASK) #####irf
        print('Train dataset====',TRAIN_DATASETS," Test dataset=", TEST_DATASETS) #####irf

        model.train(train_dl, neg_val_dl, spec_val_dl, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS))
        for k in neg_test_dataloaders.keys():
            print(f"Evaluate on {k}:")
            print("Start here::")
            print(neg_test_dataloaders[k])
            print("End here::")
            result_dict = model.evaluate(neg_test_dataloaders[k], test_dl_name = k, task = 'negation')
            store_result(result_dict, run_num)
        for k in spec_test_dataloaders.keys():
            print(f"Evaluate on {k}:")
            result_dict = model.evaluate(spec_test_dataloaders[k], test_dl_name = k, task = 'speculation')
            store_result(result_dict, run_num)

        # Error Analysis
        if ERROR_ANALYSIS_FOR_SCOPE:
            for k in neg_punct_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                print("Start here::")
                print(*neg_punct_test_dataloaders[k])
                print("End here::")
                model.evaluate(*neg_punct_test_dataloaders[k], test_dl_name = k, task = 'negation')
            for k in spec_punct_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(*spec_punct_test_dataloaders[k], test_dl_name = k, task = 'speculation')
            for k in neg_no_punct_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(*neg_no_punct_test_dataloaders[k], test_dl_name = k, task = 'negation')
            for k in spec_no_punct_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(*spec_no_punct_test_dataloaders[k], test_dl_name = k, task = 'speculation')

    else:
        raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")

    save_result()

    print(f"\n\n************ RUN {run_num+1} DONE! **************\n\n")

# %% [code] {"id":"CwD2OL-SPo21","papermill":{"duration":0.059938,"end_time":"2022-11-27T00:16:14.878256","exception":false,"start_time":"2022-11-27T00:16:14.818318","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"papermill":{"duration":0.060358,"end_time":"2022-11-27T00:16:14.997251","exception":false,"start_time":"2022-11-27T00:16:14.936893","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


# %% [code] {"papermill":{"duration":0.06169,"end_time":"2022-11-27T00:16:15.119590","exception":false,"start_time":"2022-11-27T00:16:15.057900","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
