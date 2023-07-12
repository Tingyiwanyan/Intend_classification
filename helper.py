from Intent_inference import Intent_prediction as Ip
from Intent_inference import *
import numpy as np


train_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/train.csv')
test_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/test.csv')

intent_info = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/intent_info.csv')

id2label = {int(intent_info['intent_class'][i]):intent_info['intent_name'][i] for i in range(intent_info.shape[0])}
label2id = {intent_info['intent_name'][i]:int(intent_info['intent_class'][i]) for i in range(intent_info.shape[0])}

train_label = train_data['intent_class']
train_text = train_data['template']

test_label = test_data['intent_class'].to_list()
test_text = test_data['template'].to_list()

tokenized_train_df = generate_dateframe(list(train_text), list(train_label), "distilbert-base-uncased")
tokenized_test_df = generate_dateframe(list(test_text), list(test_label), "distilbert-base-uncased")

#model_finetune(14, "distilbert-base-uncased", '/home/ubuntu/intent_classification/fine_tune_models/my_model_v1', id2label, label2id,\
#					tokenized_train_df, tokenized_test_df)

#result = intent_inference(list(test_text),'/home/ubuntu/intent_classification/fine_tune_models/my_model')

#intent_info_projection = np.array(intent_info[['intent_class','intent_name']])

Inpred = Ip()
#Inpred.model_evaluation(test_text, test_label, intent_info_projection, '/home/ubuntu/intent_classification/fine_tune_models/my_model')




