import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

CLASSES = ["NEGATIVE", "POSITIVE"]
MAX_LEN = 128

tokenizer_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def request_inference(text, model_name, triton_url):
    """
    This function makes an sentiment analysis inference request for list of texts
    from Triton Inference Server
    """
    tokenized_text = tokenizer(text,
                               padding='max_length',
                               max_length=MAX_LEN,
                               add_special_tokens=True,
                               return_tensors='np')
    input_ids = tokenized_text['input_ids'].astype(np.int32)
    attention_mask = tokenized_text['attention_mask'].astype(np.int32)
    input0_tensor = httpclient.InferInput("INPUT__0", input_ids.shape, datatype="INT32")
    input0_tensor.set_data_from_numpy(input_ids, binary_data=True)
    input1_tensor = httpclient.InferInput("INPUT__1", attention_mask.shape, datatype="INT32")
    input1_tensor.set_data_from_numpy(attention_mask, binary_data=True)
    outputs = [httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True)]

    triton_client = httpclient.InferenceServerClient(url=triton_url)

    results = triton_client.infer(model_name=model_name,
                                  inputs=[input0_tensor, input1_tensor],
                                  outputs=outputs)
    
    logits = results.as_numpy('OUTPUT__0')
    predictions = []
    
    for i in range(len(logits)):
        pred_class_idx = np.argmax(logits[i])
        predictions.append(CLASSES[pred_class_idx])
    
    return predictions
