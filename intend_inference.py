from transformers import AutoTokenizer, \
AutoModelForSequenceClassification, TextClassificationPipeline,\
TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import datasets
import evaluate

TRAIN_BATCH = 32
EVAL_BATCH = 16
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def intent_inference(text_input: str, model_path: str) -> str:
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
	model = AutoModelForSequenceClassification.from_pretrained(model_path)
	classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

	res = classifier(text_input)

	intent = res['label']
	score = res['score']

	return intent, score



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

	tokenizer = AutoTokenizer.from_pretrained(model_path)

	model = AutoModelForSequenceClassification.from_pretrained(
		model_path, num_labels=num_labels, id2label=id2label, label2id=label2id)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

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


def compute_metrics(eval_pred):
	"""
	generate accuracy evaluation
	"""

	accuracy = evaluate.load("accuracy")
	predictions, labels = eval_pred

	predictions = np.argmax(predictions, axis=1)

	return accuracy.compute(predictions=predictions, references=labels)
    


def generate_dateframe(text: list[str], label: list[int], model_path: str) -> datasets.Dataset:
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
	def preprocess_function(text: str) -> str:
		
		return tokenizer(text, trucation=True)

	return df_torch.map(preprocess_function, batched=True)






