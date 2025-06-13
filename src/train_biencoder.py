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
import xclib.data.data_utils as data_utils
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
    EXMCDatasetHardNeg,
    get_linear_schedule_with_warmup,
    TripletMarginLossOHNM,
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
    pred = csr_from_arrays(ind, sim, (num_instances, num_labels))
    return pred

def validate(args, model, tokenizer, id2label):
    model.eval()
    torch.set_grad_enabled(False)
    logger.info("Getting document embeddings")
    val_doc_embeddings = get_embedding(args, args.val_path, model, tokenizer, id2label, encode_label=False, encode_title=args.encode_title, batch_size=args.eval_batch_size)
    logger.info("Getting label embeddings")
    lbl_embeddings = get_embedding(args, args.labels_path, model, tokenizer, id2label, encode_label=True, encode_title=False, batch_size=args.eval_batch_size)
    
    predicted_labels = predict_anns(val_doc_embeddings, lbl_embeddings)
    val_true_labels = read_true_labels(args.val_path, len(id2label))
    
    acc = xc_metrics.Metrics(val_true_labels.tocsr())
    acc = acc.eval(predicted_labels.tocsr(), 5)
    print(xc_metrics.format(*acc))
    logger.info(xc_metrics.format(*acc))
    return acc[0][0]

def train(args, model, tokenizer, id2label, device):
    
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-06, weight_decay=args.weight_decay)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  number of epoches = %d", args.num_epoches)
    logger.info(" batch size%d", args.train_batch_size)
    set_seed(args)  
    if args.using_hard:
        train_dataset = EXMCDatasetHardNeg(args, tokenizer, id2label, args.encode_title, args.max_seq_length, args.max_label_length)
    else:
        train_dataset = EXMCDataset(args, tokenizer, id2label, args.encode_title, args.max_seq_length, args.max_label_length)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                                                num_training_steps=args.num_epoches*(len(train_dataset)/args.train_batch_size))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=train_dataset.get_collate_fn(args, tokenizer),
                                  num_workers=10, drop_last=True)
    loss_fuction = TripletMarginLossOHNM(
                reduction="mean",
                apply_softmax=args.loss_agressive,
                tau=0.1,
                k=args.loss_num_negatives,
                margin=args.margin)

    for epoch in range(args.num_epoches):   
        model.train()
        torch.set_grad_enabled(True)
        j = 0
        for batch_data in train_loader:
            optimizer.zero_grad()
            batch_size = len(batch_data["doc_tensor"])
            docs_vector, labels_vector = model(batch_data["doc_tensor"].to(device), 
                                                batch_data["doc_attention"].to(device), 
                                                batch_data["labels_tensor"].to(device), 
                                                batch_data["labels_attention"].to(device))
            docs_vector = F.normalize(docs_vector, dim=1)
            labels_vector = F.normalize(labels_vector, dim=1)
            labels_vector = torch.transpose(labels_vector, 0, 1)
            pred = torch.matmul(docs_vector, labels_vector)
            loss = loss_fuction(pred, batch_data["true_Y"].to(device))
            if j % 10 == 0:
                print(f"epoch: {epoch} step: {j} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            j += 1
            
        p1 = validate(args, model, tokenizer, id2label)
        save_checkpoint(args, model, optimizer, scheduler, args.iteration_epoch, args.output_dir)

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
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_epoches",
        default=1,
        type=int,
        required=True,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--encode_title",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--using_hard",
        action="store_true",
        help="Using the hard negatives from LLM",
    )
    parser.add_argument(
        "--num_hard",
        default=3,
        type=int,
        help="Number of hard negatives for each document",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
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
        "--validate_interval",
        default=6,
        type=int,
        help="The hidden  size",
    )
    parser.add_argument(
        "--margin",
        default=0.3,
        type=float,
        help="loss margin",
    )
    parser.add_argument(
        "--loss_num_positives",
        default=1,
        type=int,
        help="positives num",
    )
    parser.add_argument(
        "--loss_num_negatives",
        default=10,
        type=int,
        help="negatives num",
    )
    parser.add_argument('--loss_agressive', 
        action='store_true', 
        help = "loss agressive",
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
        "--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for eval.",
    )
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
   
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
            "--trn_docs_path",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--val_path",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--iteration_epoch",
            default=None,
            type=int,
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    args = get_arguments()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using PyTorch version:', torch.__version__, 'Device:', device)

    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BiEncoder(args)
    model.to(device)
    
    id2label = read_labels(args)
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    print(logger)
    args.device = device
    train(args, model, tokenizer, id2label, device)




if __name__ == "__main__":
    main()
