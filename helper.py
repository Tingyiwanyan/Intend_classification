from intend_inference import *
import numpy as np


train_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/train.csv')
test_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/test.csv')

intent_info = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/intent_info.csv')

id2label = {int(intent_info['intent_class'][i]):intent_info['intent_name'][i] for i in range(intent_info.shape[0])}
label2id = {intent_info['intent_name'][i]:int(intent_info['intent_class'][i]) for i in range(intent_info.shape[0])}

train_label = train_data['intent_class']
train_text = train_data['template']

test_label = test_data['intent_class']
test_text = test_data['template']

tokenized_train_df = generate_dateframe(list(train_text), list(train_label), "/home/ubuntu/intent_classification/pre_train_models/XLMRoberta-Alexa-Intents-Classification")
tokenized_test_df = generate_dateframe(list(test_text), list(test_label), "/home/ubuntu/intent_classification/pre_train_models/XLMRoberta-Alexa-Intents-Classification")

model_finetune(14, "/home/ubuntu/intent_classification/pre_train_models/XLMRoberta-Alexa-Intents-Classification", '/home/ubuntu/intent_classification/fine_tune_models/my_model', id2label, label2id,\
					tokenized_train_df, tokenized_test_df)




