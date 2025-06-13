from os.path import join
import sys
import argparse
import glob
import json
import logging
import os
import random
from xclib.utils.sparse import _map_cols, csr_from_arrays
import xclib.evaluation.xc_metrics as xc_metrics


import numpy as np
import torch
from models import BiEncoder 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import random
from transformers import (
    AutoTokenizer
)
from transformers import glue_processors as processors
from torch import nn
import torch.nn.functional as F
import pandas as pd
from scipy.sparse import coo_matrix
from xclib.utils.clustering import cluster_balance, b_kmeans_dense

logger = logging.getLogger(__name__)
from util import (
    EmbeddingDataset,
    EXMCDataset,
    save_checkpoint,
    ShortlistMIPS,
    get_embedding,
)
import collections




def read_true_labels(true_labels_path, num_labels):
    data = []
    row = []
    col = []
    num_instances = 0
    with open(true_labels_path, "r") as f:
        for j, line in enumerate(f):
            instance = json.loads(line)
            doc = instance["content"]
            title = instance["title"]
            pos_label = instance["target_ind"]
            row.extend(len(pos_label)*[j]) 
            col.extend(pos_label)
            data.extend(len(pos_label)*[1])
            num_instances += 1
    return coo_matrix((data, (row, col)), shape=(num_instances, num_labels))

def read_docs_id(docs_path):
    docs_id = []
    with open(docs_path, "r") as f:
        for j, line in enumerate(f):
            instance = json.loads(line)
            doc_id = instance["uid"]
            docs_id.append(doc_id)
    return docs_id



def predict_anns(X, W, k=300, method='hnswlib', space='cosine',
                M=100, efC=300, n_threads=6, add_padding=False):
    """
    Train a nearest neighbor structure on W
    - for a given test point: query the graph for closest label
    """
    num_instances, num_labels = len(X), len(W)
    
    # add a padding index in the end
    if add_padding:
        num_labels += 1
    
    # can handle zero vectors
    graph = ShortlistMIPS(method=method, M=M, efC=efC, efS=k, num_neighbours=k, space=space, num_threads=n_threads)    
    graph.fit(W)
    ind, sim = graph.query(X)
    #pred = csr_from_arrays(ind, sim, (num_instances, num_labels))
    return ind, sim

def generate_shortlist(args, model, tokenizer, id2label):
    model.eval()
    torch.set_grad_enabled(False)
    logger.info("Getting document embeddings")
    val_doc_embeddings = get_embedding(args, args.val_path, model, tokenizer, id2label, encode_label=False, encode_title=args.encode_title, batch_size=args.eval_batch_size)
    logger.info("Getting label embeddings")
    lbl_embeddings = get_embedding(args, args.labels_path, model, tokenizer, id2label, encode_label=True, encode_title=False, batch_size=args.eval_batch_size)
    
    predicted_labels, sim = predict_anns(val_doc_embeddings, lbl_embeddings)

    val_true_labels = read_true_labels(args.val_path, len(id2label))

    val_docs_id = read_docs_id(args.val_path)
    assert len(val_docs_id) == len(predicted_labels)

    with open(args.output_shortlist_path, "w") as fw:
        for i in range(len(predicted_labels)):
            instance = {}
            instance["doc_id"] = val_docs_id[i]
            instance["pred_labels"] = [int(x) for x in predicted_labels[i][:30]]
            fw.write(json.dumps(instance) + "\n")

    pred = csr_from_arrays(predicted_labels, sim, (len(val_doc_embeddings), len(lbl_embeddings)))
    acc = xc_metrics.Metrics(val_true_labels.tocsr())
    acc = acc.eval(pred.tocsr(), 5)
    print(xc_metrics.format(*acc))
    # logger.info(xc_metrics.format(*acc))


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--share_weight', 
        action='store_true', 
        help = "share the same query encoder",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
    )
    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_label_length",
        default=50,
        type=int,
        help="The maximum total label length after tokenization. Sequences longer "
             "than this will be truncated",
    )
    parser.add_argument(
        "--encode_title",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--num_threads",
        default=15,
        type=int,
        help="Number of threads",
    )

    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
            "--val_path",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--output_shortlist_path",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--model_path",
            default=None,
            type=str,
        )
    parser.add_argument("--labels_path", type=str, default=None, help="All labels path")
    args = parser.parse_args()

    return args

def read_labels(args):
    if args.labels_path is not None:
        id2label = {}
        with open(args.labels_path, "r", ) as f:
            for doc_index, line in enumerate(f):
                line = line.strip()
                instance = json.loads(line)
                doc_content = instance['content']
                title = instance["title"]
                id2label[doc_index] = title
    else:
        logger.error('no labels')
    return id2label


def main():
    args = get_arguments()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.log_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    print(logger)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using PyTorch version:', torch.__version__, 'Device:', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = read_labels(args)

    model = BiEncoder(args)
    
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['net'])
        print("load model successfully")
    
    args.device = device
    model.to(device)
    generate_shortlist(args, model, tokenizer, id2label)
    

if __name__ == "__main__":
    main()
