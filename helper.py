from intend_inference import *
import pandas as pd
import numpy as np


train_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/train.csv')
test_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/test.csv')

intent_info = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/intent_info.csv')

intent_info = {intent_info['intent_class'][i]:intent_info['intent_name'][i] for i in range(intent_info.shape[0])}

train_label = train_data['intent_class']
train_text = train_data['template']

test_label = test_data['intent_class']
test_text = test_data['template']

