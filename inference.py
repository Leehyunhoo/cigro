
import glob
from logging import raiseExceptions
import pickle
import torch
import torch.nn as nn
import numpy as np
import time
from urllib.request import *
from transformers import ElectraTokenizer

import transformers
transformers.logging.set_verbosity_error()
#from PIL import Image

from .dataset import valTransform
from .config import Config
from .model.model import Model

'''
def file_process(class_dict_url):
    #download files from google cloud storage
    weight_url = 'https://storage.googleapis.com/nnc-gifticon/text_folder/model_best.pth'

    urlretrieve(weight_url, '/home/ubuntu/project/deepLearning/weight/model_best.pth')
    print('# download trained weight completed...')
    
    urlretrieve(class_dict_url, '/home/ubuntu/project/deepLearning/data/class_dict.pickle')
    print('# download class dictionary completed...')

    with open('/home/ubuntu/project/deepLearning/data/class_dict.pickle', 'rb') as fr:
        class_dict = pickle.load(fr)

    return class_dict
'''

def load_weight(model_name,model_weight_dir):
    args.backbone_pretrained = False
    #args.device = 'cpu'
    args.model = model_name
    model = Model(args)
    model.load_state_dict(torch.load(model_weight_dir, map_location = args.device)['state_dict'])
    model.to(args.device)
    model.eval()
    return model



def tokenize(sequences):
        input_ids = []
        attention_masks = []

        for seq in sequences:
            encoded_dict = tokenizer.encode_plus(
                seq,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=args.max_length,  # Pad & truncate all sentences.
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids.squeeze(), attention_masks.squeeze()


def Initialize(class_dict_dir, model_weight_dir):

    global args
    global class_dict  
    global model
    global tokenizer

    args = Config()

    model_name = "monologg/koelectra-base-v3-discriminator"#request.data.get('model')
    '''
    class_dict_url = request.data.get('class_dict') 
    class_dict = file_process(class_dict_url)
    '''

    model = load_weight(model_name,model_weight_dir)
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    with open(class_dict_dir, 'rb') as fr:
        class_dict = pickle.load(fr)
    print("Initialize complete")


def Inference(text,k):
    k = int(k)
    print(text)
    #text = text[0]

    print(' ')
    print("device: ", args.device)
    print(' ')

    input_ids, attention_masks = tokenize([text])
    inputs = [torch.as_tensor(input_ids, dtype=torch.long), torch.as_tensor(attention_masks, dtype=torch.long)]
    inputs = [x.unsqueeze(0).to(args.device) for x in inputs]
    


    with torch.no_grad():
        t = time.time()
        output = model(inputs)
        print(' ')
        print('time: ', time.time()-t)
        print(' ')
        print(output)
    
    clss = output.argmax(1).detach().tolist()[0] 
    confidence = nn.Softmax(dim=-1)(output)[:, clss].detach().tolist()[0]
    
    topkk = torch.topk(output, k).indices.detach().cpu().tolist()[0]
    topkk_probs = torch.topk(nn.Softmax(dim=-1)(output), k).values.detach().cpu().tolist()[0]


    gifts = []
    for i in range(k):
        print(i,topkk[i])
        gifts.append(class_dict[f'{topkk[i]}'])


    print(gifts)
    print(' ')
    print('confidence: ', confidence)
    print(' ')       
    return class_dict[f'{clss}'], gifts, topkk_probs