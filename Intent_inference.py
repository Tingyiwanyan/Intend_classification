from transformers import AutoTokenizer, \
AutoModelForSequenceClassification, TextClassificationPipeline,\
TrainingArguments, Trainer, DataCollatorWithPadding, \
create_optimizer, TFAutoModelForSequenceClassification, pipeline
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import datasets
import evaluate
import tensorflow as tf
import numpy as np
from sklearn import metrics

TRAIN_BATCH = 32
EVAL_BATCH = 16
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
#WEIGHT_DECAY = 0.01


def intent_inference(text_input: list, model_path: str) -> str:
	"""
	inferencing intent based on innput text

	Parmaters:
	-----------
	text_input: input text string for classification.
	model_path: path to LLM model

	Returns:
	intent: string to show the text intent
	score: classification probability score
	"""

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
	classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

	res = classifier(text_input)

	#intent = res['label']
	#score = res['score']

	return res


def model_finetune(num_labels: int, model_path: str, save_model_path: str, id2label: dict, label2id: dict,\
					train_df: datasets.Dataset, test_df: datasets.Dataset):
	"""
	fine tune existing LLM model

	Parameters:
	-----------
	num_labels: int, the finetune class number.
	model_path: the existing model to be finetuned on.
	save_model_path: the path for saving the finetuned model.
	id2label: map from index to label(https://huggingface.co/docs/transformers/main_classes/configuration)
	label2id: map from label to index
	train_df: tokenized training data
	test_df: tokenized testing data
	"""
	batches_per_epoch = len(train_df) // TRAIN_BATCH
	total_train_steps = int(batches_per_epoch * TRAIN_EPOCHS)
	optimizer, schedule = create_optimizer(init_lr=LEARNING_RATE, num_warmup_steps=0, \
		num_train_steps=total_train_steps)
	tokenizer = AutoTokenizer.from_pretrained(model_path)

	model = TFAutoModelForSequenceClassification.from_pretrained(
		model_path, num_labels=int(num_labels), id2label=id2label, label2id=label2id, \
		ignore_mismatched_sizes=True)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

	tf_train_set = model.prepare_tf_dataset(
		train_df,
		shuffle=True,
		batch_size=TRAIN_BATCH,
		collate_fn=data_collator,)

	tf_validation_set = model.prepare_tf_dataset(
		test_df,
		shuffle=False,
		batch_size=EVAL_BATCH,
		collate_fn=data_collator,)

	model.compile(optimizer=optimizer)

	metric_callback = KerasMetricCallback(metric_fn=compute_metrics, \
		eval_dataset=tf_validation_set)

	push_to_hub_callback = PushToHubCallback(
		output_dir=save_model_path,
		tokenizer=tokenizer,)

	callbacks = [metric_callback, push_to_hub_callback]

	model.fit(x=tf_train_set, validation_data=tf_validation_set, \
		epochs=TRAIN_BATCH, callbacks=callbacks)

	
	"""
	training_args = TrainingArguments(
		output_dir=save_model_path,
	    learning_rate=LEARNING_RATE,
	    per_device_train_batch_size=TRAIN_BATCH,
	    per_device_eval_batch_size=EVAL_BATCH,
	    num_train_epochs=TRAIN_EPOCHS,
	    weight_decay=WEIGHT_DECAY,
	    evaluation_strategy="epoch",
	    save_strategy="epoch",
	    load_best_model_at_end=True,
	    push_to_hub=True,)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_df,
		eval_dataset=test_df,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,)

	trainer.train()
	"""

def compute_metrics(eval_pred):
	"""
	generate accuracy evaluation
	"""

	accuracy = evaluate.load("accuracy")
	predictions, labels = eval_pred

	predictions = np.argmax(predictions, axis=1)

	return accuracy.compute(predictions=predictions, references=labels)


def evaluation_metric(y_pred: list, y_true: list):
	"""
	print evaluation results, precision, recall, f1 score
	"""

	print(metrics.classification_report(y_true, y_pred, digits=3))


def project_labels(label: int, labels_index: np.array):
	"""
	project class index to text label

	Parameters:
	label: class label
	labels_index: intent class label index,

	Returns:
	labeled text
	"""
	index = np.where(labels_index[:,0] == label)[0][0]

	return labels_index[index][1]


def validate_evaluation(test_text: list, test_label:list, labels_index: np.array, if_convert = True):
	"""
	evaluate testing result

	Parameters:
	test_text: testing text to be evaluated
	test_label: ground truth test label
	labels_index: intent class label index

	Returns:
	print the evaluation metrics
	"""

	if if_convert == True:
		testing_label = map(lambda x: project_labels(x, labels_index), test_label)

	predict_label = 2



def generate_dateframe(text: list, label: list, model_path: str) -> datasets.Dataset:
	"""
	generate customer tokenized dataset 

	Parameters:
	-----------
	text: list of string inputs
	label: list of label inputs
	model_path: path to LLM model

	Returns:
	--------
	generated tokenized dataset
	"""
	df = pd.DataFrame({'text': text,'label': label})

	df_torch = Dataset.from_pandas(df)

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	def preprocess_function(text):

		return tokenizer(text["text"], truncation=True)

	return df_torch.map(preprocess_function, batched=True)






