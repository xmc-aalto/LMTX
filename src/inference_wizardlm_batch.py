import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
# from peft import PeftModel
import transformers
import argparse
import numpy as np
from tqdm import tqdm
# import gradio as gr
import json
import random
from torch.utils.data import DataLoader, Dataset
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass




def cut_input(tokenizer, doc, length):
    if len(doc.split(" ")) > length:
        doc = " ".join(doc.split(" ") [:length]).strip()
    doc_tokenized = tokenizer.encode(doc)
    length_doc = len(doc_tokenized)
    while length_doc > length:
        doc = doc.split(" ")
        doc = doc[:-1]
        doc = " ".join(doc).strip()
        doc_tokenized = tokenizer.encode(doc)
        length_doc = len(doc_tokenized)
    return doc

class LLMDataset(Dataset):
    """"""

    def __init__(self, args, tokenizer, encode_title, doc_path, labels_path, shortlist_path, prompt_id):
        """
        Args:
        
        """
        self.tokenizer = tokenizer
        self.prompt_id = prompt_id
        self.encode_title = encode_title
        self.docs_id, self.docs = self.load_data(doc_path)
        self.id2label = self.load_labels(labels_path)
        self.shortlist_docs_id, self.shortlist_labels = self.load_shortlist_labels(shortlist_path)

        assert len(self.shortlist_labels) == len(self.docs)
        assert self.shortlist_docs_id == self.docs_id

        self.max_seq_length = args.max_seq_length
        self.topk = args.topk

    def load_data(self, doc_path):
        docs = []
        docs_id = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                title = instance["title"]
                doc = instance["content"]
                if self.encode_title:
                    doc = title + "\t" + doc
                docs.append(doc)
                docs_id.append(instance["uid"])
        return docs_id, docs

    def load_labels(self, labels_path):
        id2label = []
        with open(labels_path, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                id2label.append(instance["title"])
        return id2label

    def load_shortlist_labels(self, shortlist_path):
        shortlist_pred = []
        doc_ids = []
        with open(shortlist_path, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                shortlist_pred.append(instance["pred_labels"])
                doc_ids.append(instance["doc_id"])
        return doc_ids, shortlist_pred
    def cut_input(self, tokenizer, doc, length):
        if len(doc.split(" ")) > length:
            doc = " ".join(doc.split(" ")[:length]).strip()
        doc_tokenized = tokenizer.encode(doc)
        length_doc = len(doc_tokenized)
        while length_doc > length:
            doc = doc.split(" ")
            doc = doc[:-1]
            doc = " ".join(doc).strip()
            doc_tokenized = tokenizer.encode(doc)
            length_doc = len(doc_tokenized)
        return doc
    def __len__(self):
        return len(self.docs)*self.topk

    def __getitem__(self, idx):
        doc_id = idx // self.topk
        doc = self.docs[doc_id]
        label_id = self.shortlist_labels[doc_id][idx % self.topk]
        label_text = self.id2label[label_id]
        doc = cut_input(self.tokenizer, doc, length=self.max_seq_length)
        prompts_list = [f"document = \"{doc}\". Is the tag \"{label_text}\" relevant to the document? answer yes or no",
                f"document = \"{doc}\". The document is amazon product description, Is the tag \"{label_text}\" relevant to the document? answer yes or no",
                f"document = \"{doc}\". The document is the wikipedia page. Is the tag \"{label_text}\" relevant to the document? answer yes or no",
                f"document = \"{doc}\". The document is the wikipedia page. Is \"{label_text}\" the relevant topic to the document? answer yes or no",
                f"document = \"{doc}\". The document is the amazon product description. Is \"{label_text}\" the relevant product to the document? answer yes or no",
                f"document = \"{doc}\". The document is the wikipedia page. Is \"{label_text}\" the related other wikipedia page title to the document? answer yes or no",
                f"document = \"{doc}\". The document is the wikipedia page. Is \"{label_text}\" the related other wikipedia page name to the document? answer yes or no",
                f"document = \"{doc}\". The document is the wikipedia page. Does another wikipedia page name \"{label_text}\"  have the relation to the document? answer yes or no",
        ]
        input_for_llm = prompts_list[self.prompt_id]      
        prompts = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input_for_llm} ASSISTANT:"""
        #inputs = self.tokenizer.encode(prompts)
        doc_uid = self.docs_id[doc_id]
       
        return doc_uid, label_id, prompts
    @classmethod
    def get_collate_fn(cls, args):
        def create_biencoder_input2(features):
            docs_list = []
            docs_indices = []
            labels_indices = []

            for item in features:
                docs_list.append(item[2])
                docs_indices.append(item[0])
                labels_indices.append(item[1])

            batch_data = {}
            batch_data["docs_id"] = docs_indices
            batch_data["docs"] = docs_list
            batch_data["labels_id"] = labels_indices
            return batch_data
        return create_biencoder_input2

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument('-i', '--trn_docs_path', type=str, default=None, help='')
    parser.add_argument('--shortlist_path', type=str, default=None, help='')
    parser.add_argument('--ouput_trn_llm_path', type=str, default=None, help='')
    parser.add_argument('--labels_path', type=str, default=None, help='')
    parser.add_argument('--topk', type=int, default=10, help='')
    parser.add_argument('--prompt_id', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--max_seq_length', type=int, default=430, help='')
    parser.add_argument("--encode_title", action="store_true", help="")
    

    args = parser.parse_args()

    load_8bit= False,

    # assert base_model, (
    #     "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    # )

    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path = args.model_path)
    tokenizer.padding_side = "left"
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path = args.model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    #model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #model.config.bos_token_id = 1
    #model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def inference(
            batch_data,
            input=None,
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            num_beams=1,
            max_new_tokens=2048,
            **kwargs,
    ):
        
        
        # prompts = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {batch_data} ASSISTANT:"""
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
        #input_ids = inputs["input_ids"].to(device)
        inputs = inputs.to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            # if torch.cuda.device_count() > 1:
            #     generation_output = model.module.generate(
            #         **inputs,
            #         generation_config=generation_config,
            #         return_dict_in_generate=True,
            #         output_scores=True,
            #         max_new_tokens=max_new_tokens,
            #     )
            # else:
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences       
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        # print(len(output))
        # print(output[0])
        # print(output[1])
        #output = output[0].split("ASSISTANT:")[1].strip()
       
        return output



    llm_dataset = LLMDataset(args, tokenizer, args.encode_title, args.trn_docs_path, args.labels_path, args.shortlist_path, args.prompt_id)
    llm_loader = DataLoader(llm_dataset, batch_size=args.batch_size, collate_fn=llm_dataset.get_collate_fn(args),
                                  num_workers=10)
    
    doc2pseudo = {}
    doc2hardneg = {}
    for batch_data in tqdm(llm_loader):
        batch_input = batch_data["docs"]
        batch_docs_id = batch_data["docs_id"] 
        batch_labels_id = batch_data["labels_id"] 
        output = inference(batch_data=batch_input)
        for j, response in enumerate(output):
            docid = batch_docs_id[j]
            labelid = batch_labels_id[j]
            response = response.split("ASSISTANT:")[1].strip()
            
            if docid not in doc2pseudo:
                doc2pseudo[docid] = []
            if docid not in doc2hardneg:
                doc2hardneg[docid] = []
            if response[:10].lower().find("yes") != -1:
                doc2pseudo[docid].append(labelid)
            if response[:10].lower().find("no") != -1:
                doc2hardneg[docid].append(labelid)
        if len(doc2pseudo) % 100 == 0:
            print(f"num of docs: {len(doc2pseudo)}")

    fw = open(args.ouput_trn_llm_path, "w")
    with open(args.trn_docs_path, "r") as f:
        for line in f:
            instance = json.loads(line.strip())
            uid = instance["uid"]
            instance["target_ind"] = doc2pseudo[uid]
            instance["hard_negs"] = doc2hardneg[uid]
            fw.write(json.dumps(instance) + "\n")
    fw.close()

if __name__ == "__main__":
    main()
