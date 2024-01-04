import gradio as gr
import random
import time
from transformers import pipeline
import requests
import re

def split_text(text):
    match = re.search(r'([^\.?\!]*)([\.?\!])(.*)', text)
    
    if match:
        before = match.group(1).strip()
        after = match.group(3).strip()
        words = [word for word in before.split() if word]
        return before, after, len(words)
    else:
        words = [word for word in text.split() if word]
        return text, "", len(words)

API_URL = "https://api-inference.huggingface.co/models/youlun77/finetuning-sentiment-model-25000-samples"
headers = {"Authorization": "Bearer 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"}

def query_distilbert(payload):
    start = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json(), time.time()-start



def demo_func_distil_label(input_text):
    sentences = input_text.split('.')
    result =[]
    current_sentence = ''
    total_score = 0
    total_time = 0

    for sentence in sentences:
        if sentence.strip() and len(current_sentence + sentence) <= 512:
            current_sentence += sentence
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence
    if current_sentence:
        result.append(current_sentence.strip())

    result = [sentence for sentence in result if sentence]
    for part in result:
        output, duration = query_distilbert({
	    "inputs": str(part),
        })
        print(f'D:{output}')
        total_time += duration
        for array in output:
            for item in array:
                if "label" in item and item["label"] == "LABEL_0":
                    total_score += item["score"]

    avg=total_score/len(result)
    if avg<=0.5:
        return "POSTIVE!", total_time
    else:
        return "NEGATIVE:(", total_time
    

    

    


API_URL_BERT = "https://api-inference.huggingface.co/models/youlun77/finetuning-sentiment-model-25000-samples-BERT"
def query_bert(payload):
    start = time.time()
    response = requests.post(API_URL_BERT, headers=headers, json=payload)
    return response.json(), time.time()-start
	
def demo_func_bert_label(input_text):
    sentences = input_text.split('.')
    result =[]
    current_sentence = ''
    total_score = 0
    total_time = 0

    for sentence in sentences:
        if sentence.strip() and len(current_sentence + sentence) <= 512:
            current_sentence += sentence
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence
    if current_sentence:
        result.append(current_sentence.strip())

    result = [sentence for sentence in result if sentence]
    for part in result:
        output, duration = query_bert({
	    "inputs": str(part),
        })
        print(f'{output}')
        total_time += duration
        for array in output:
            for item in array:
                if "label" in item and item["label"] == "LABEL_0":
                    total_score += item["score"]

    avg=total_score/len(result)
    if avg<=0.5:
        return "POSTIVE!", total_time
    else:
        return "NEGATIVE:(", total_time
    

model_name = "distilbert-base-uncased"
classifier = pipeline('sentiment-analysis', model=model_name)


with gr.Blocks() as distilbot:
    input_text = gr.Textbox()
    with gr.Row():
        with gr.Column(scale=1):
            distil_output_label = gr.Label()
            distil_time_text = gr.Textbox()
        with gr.Column(scale=1):
            bert_output_label = gr.Label()
            bert_time_text = gr.Textbox()

    input_text.submit(demo_func_distil_label, 
                      inputs = input_text, 
                      outputs = [distil_output_label, distil_time_text])
    
    input_text.submit(demo_func_bert_label, 
                      inputs = input_text, 
                      outputs = [bert_output_label, bert_time_text])
    

if __name__ == "__main__":
    distilbot.launch()
