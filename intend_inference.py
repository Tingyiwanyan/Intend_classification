from transformers import AutoTokenizer, \
AutoModelForSequenceClassification, TextClassificationPipeline,\
TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader



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

	res = classifier(text__input)

	intent = res['label']
	score = res['score']

	return intent, score



def model_finetune(num_labels: int, model_path: str, save_model_path: str, id2label: dict, label2id: dict):
	"""
	fine tune existing LLM model

	Parameters:
	-----------
	num_labels: int, the finetune class number.
	model_path: the existing model to be finetuned on.
	save_model_path: the path for saving the finetuned model.
	id2label: map from index to label(https://huggingface.co/docs/transformers/main_classes/configuration)
	label2id: map from label to index
	"""

	model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=num_labels, id2label=id2label, label2id=label2id)

    #tokenizer = AutoTokenizer.from_pretrained(model_path)



def data_preprocess(text: str, tokenizer) -> str:
	"""
	truncate input text to maximum model input length

	Parameters:
	-----------
	text: input text string
	tokenizer: pre-trained model tokenizer

	Return:
	truncated tokenize text string
	"""

	return tokenizer(text, trucation=True)



