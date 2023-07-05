from intend_inference import *
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('XLMRoberta-Alexa-Intents-Classification')

train_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/train.csv')
test_data = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/test.csv')

intent_info = pd.read_csv('/home/ubuntu/intent_classification/dataset/speech-to-intent/intent_info.csv')

id2label = {intent_info['intent_class'][i]:intent_info['intent_name'][i] for i in range(intent_info.shape[0])}
label2id = {intent_info['intent_name'][i]:intent_info['intent_class'][i] for i in range(intent_info.shape[0])}

train_label = train_data['intent_class']
train_text = train_data['template']

test_label = test_data['intent_class']
test_text = test_data['template']

train_df = pd.DataFrame({'text':list(train_text),'label':list(train_label)})

test_df = pd.DataFrame({'text':list(test_text), 'label':list(test_label)})

train_df_torch = Dataset.from_pandas(train_df)

test_df_torch = Dataset.from_pandas(test_df)

tokenized_train_df = train_df_torch.map(preprocess_function(tokenizer), batched=True)

tokenized_test_df = test_df_torch.map(preprocess_function(tokenizer), batched=True)



