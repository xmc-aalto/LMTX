import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import numba as nb
import torch.distributed as dist
from tqdm import tqdm, trange
import os
import json
import logging
import random
import pickle
import numpy as np
from xclib.utils.sparse import _map_cols, csr_from_arrays
import torch.nn.functional as F
import torch
from torch.serialization import default_restore_location
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from xclib.utils.shortlist import Shortlist
from evaluation import EvaluateRetrieval

logger = logging.getLogger(__name__)

def map_dense(ind, mapping):
    out = np.full_like(ind, fill_value=0)
    nr, nc = ind.shape
    for i in nb.prange(nr):
        for j in range(nc):
            out[i, j] = mapping[ind[i, j]]
    return out


class ShortlistMIPS(Shortlist):
    """Get nearest labels using their embeddings
    * brute or HNSW algorithm for search
    * option to process label representations with label correlation matrix

    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    """
    def __init__(self, method='hnswlib', num_neighbours=300, M=100, efC=300,
                 efS=300, space='cosine', verbose=True, num_threads=16):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads, space)
        self.valid_indices = None

    def fit(self, X, *args, **kwargs):
        ind = np.where(np.square(X).sum(axis=1) > 0)[0]
        self.valid_indices = ind
        X = X[self.valid_indices]
        super().fit(X)

    def query(self, X, *args, **kwargs):
        ind, sim = super().query(X)
        if self.valid_indices is not None:
            ind = map_dense(ind, self.valid_indices)
        return ind, sim

    def save(self, fname):
        try:
            metadata = {
                'valid_indices': self.valid_indices,
            }
            super().save(fname+".index")
        except ValueError or AttributeError:
            pass
        finally:            
            pickle.dump(metadata, open(fname+".metadata", 'wb'))

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.valid_indices = obj['valid_indices']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")

def save_checkpoint(args, model, optimizer, scheduler, epoch, model_dir) -> str:
    checkpoint = {
                "net": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': scheduler.state_dict()
            }
    model_name = "biencoder_iter_" + str(epoch) + ".pt"
    model_path = os.path.join(model_dir, model_name)
    torch.save(checkpoint, model_path)
    logger.info('Saved checkpoint at %s', model_path)
    return model_path

class EmbeddingDataset(Dataset):
    def __init__(self, args, doc_path, tokenizer, id2label, encode_label, encode_title, max_seq_length = 512):
        self.id2label = id2label
        self.encode_label = encode_label
        self.encode_title = encode_title
        self.all_docs = self.load_doc_data(doc_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        

    def load_doc_data(self, doc_path):
        docs = []
        labels = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                doc = instance["content"]
                title = instance["title"]
                pos_label = instance["target_ind"]
                if self.encode_label:
                    docs.append(title)
                else:
                    if self.encode_title:
                        docs.append(title + "\t" + doc)
                    else:
                        docs.append(doc)
        return docs

    def __getitem__(self, index):
        """Get a label at index"""
        doc = self.all_docs[index]
        doc_token_ids = self.tokenizer.encode(doc, add_special_tokens=True)
        if len(doc_token_ids) > self.max_seq_length:
            doc_token_ids = doc_token_ids[:self.max_seq_length-1] + [self.tokenizer.sep_token_id]
      
        return doc_token_ids, index

    def __len__(self):
        return len(self.all_docs)

    @classmethod
    def get_collate_fn(cls, args, tokenizer):
        def create_biencoder_input2(features):
            doc_list = []
            batch_indices = []
            for index, feature in enumerate(features):
                doc_list.append(feature[0]) 
                batch_indices.append(feature[1])

            max_doc_len = max([len(doc) for doc in doc_list])
            doc_list = [doc + [tokenizer.pad_token_id]*(max_doc_len-len(doc)) for doc in doc_list]
            doc_tensor = torch.LongTensor(np.array(doc_list))
            return {'doc_tensor': doc_tensor,
                    'doc_attention': (doc_tensor!= tokenizer.pad_token_id).long(), 
                    'batch_size' : len(doc_list),
                    "batch_indices": batch_indices
                    }
        return create_biencoder_input2


class EXMCDataset(Dataset):
    def __init__(self, args, tokenizer, id2label, encode_title, max_seq_length, max_label_length):
        self.id2label = id2label
        self.encode_title = encode_title
        self.all_docs, self.true_labels = self.load_doc_data(args.trn_docs_path)
        assert len(self.all_docs) == len(self.true_labels)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_label_length = max_label_length

    def load_labels_data(self, doc_path):
        all_labels = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                title = instance["title"]
                all_labels.append(title)
        return all_labels

    def load_doc_data(self, doc_path):
        docs = []
        labels = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                doc = instance["content"]
                pos_labels = instance["target_ind"]
                if len(pos_labels) > 0:
                    if self.encode_title:
                        title = instance["title"]
                        docs.append(title + "\t" + doc)
                    else:
                        docs.append(doc)
                    labels.append(pos_labels)
        print("the number of training data: " + str(len(docs)))
        return docs, labels


    def __len__(self):
        return len(self.all_docs)

    def __getitem__(self, index):
        """Get a label at index"""
        doc = self.all_docs[index]
        doc_token_ids = self.tokenizer.encode(doc, add_special_tokens=True)
        if len(doc_token_ids) > self.max_seq_length:
            doc_token_ids = doc_token_ids[:self.max_seq_length-1] + [self.tokenizer.sep_token_id]
          
        pos_indices = self.true_labels[index]
        sampled_pos_ind = np.random.choice(pos_indices)
        sample_label_text = self.id2label[sampled_pos_ind]

        label_token_ids = self.tokenizer.encode(sample_label_text, add_special_tokens=True)
        if len(label_token_ids) > self.max_label_length:
            label_token_ids = label_token_ids[:self.max_label_length-1] + [self.tokenizer.sep_token_id]
        return (doc_token_ids, (sampled_pos_ind, pos_indices), label_token_ids, index) 
    @classmethod
    def get_collate_fn(cls, args, tokenizer):
        def create_biencoder_input2(features):
            batch_labels = []
            random_pos_indices = []
            docs_list = []
            labels_list = []
            batch_indices = []
            
            for item in features:
                docs_list.append(item[0])
                random_pos_indices.append(item[1][0])
                batch_labels.append(item[1][1])
                labels_list.append(item[2])
                batch_indices.append(item[3])

            max_doc_len = max([len(doc) for doc in docs_list])
            docs_list = [doc + [tokenizer.pad_token_id]*(max_doc_len-len(doc)) for doc in docs_list]
            doc_tensor = torch.LongTensor(np.array(docs_list))

            max_label_len = max([len(l) for l in labels_list])
            labels_list = [l + [tokenizer.pad_token_id]*(max_label_len-len(l)) for l in labels_list]
            labels_tensor = torch.LongTensor(np.array(labels_list))

            batch_size = len(batch_labels)
            batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     

            random_pos_indices_set = set(random_pos_indices)
            random_pos_indices = np.array(random_pos_indices, dtype=np.int32)
    

            for (i, item) in enumerate(batch_labels):
                intersection = set(item).intersection(random_pos_indices_set)
                result = np.zeros(batch_size, dtype=np.float32)
                for idx in intersection:
                    result += (idx == random_pos_indices)   
                batch_selection[i] = result  

            batch_data = {}
            batch_data["doc_tensor"] = doc_tensor
            batch_data["doc_attention"] = (doc_tensor != tokenizer.pad_token_id).long()
            batch_data["labels_tensor"] = labels_tensor
            batch_data["labels_attention"] = (labels_tensor != tokenizer.pad_token_id).long()
            batch_data["true_Y"] = torch.from_numpy(batch_selection)
            batch_data["Y_shortlist"] = torch.LongTensor(random_pos_indices)
            return batch_data
        return create_biencoder_input2


class EXMCDatasetHardNeg(Dataset):
    def __init__(self, args, tokenizer, id2label, encode_title, max_seq_length, max_label_length):
        self.id2label = id2label
        self.encode_title = encode_title
        self.all_docs, self.true_labels, self.hard_neg_labels = self.load_doc_data(args.trn_docs_path)
        assert len(self.all_docs) == len(self.true_labels)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_label_length = max_label_length
        self.num_hard = args.num_hard

    def load_labels_data(self, doc_path):
        all_labels = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                title = instance["title"]
                all_labels.append(title)
        return all_labels

    def load_doc_data(self, doc_path):
        docs = []
        labels = []
        all_neg_labels = []
        with open(doc_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                doc = instance["content"]
                pos_labels = instance["target_ind"]
                hard_negs = instance["hard_negs"]
                if len(pos_labels) > 0:
                    if self.encode_title:
                        title = instance["title"]
                        docs.append(title + "\t" + doc)
                    else:
                        docs.append(doc)
                    labels.append(pos_labels)
                    all_neg_labels.append(hard_negs)
        print("the number of training data: " + str(len(docs)))
        return docs, labels, all_neg_labels


    def __len__(self):
        return len(self.all_docs)

    def __getitem__(self, index):
        """Get a label at index"""
        doc = self.all_docs[index]
        doc_token_ids = self.tokenizer.encode(doc, add_special_tokens=True)
        if len(doc_token_ids) > self.max_seq_length:
            doc_token_ids = doc_token_ids[:self.max_seq_length-1] + [self.tokenizer.sep_token_id]
          
        pos_indices = self.true_labels[index]
        hard_negs_indices = self.hard_neg_labels[index]
        random.shuffle(hard_negs_indices)
        hard_negs_indices = hard_negs_indices[:self.num_hard]
        sampled_pos_ind = np.random.choice(pos_indices)
        sample_label_text = self.id2label[sampled_pos_ind]

        pos_label_token_ids = self.tokenizer.encode(sample_label_text, add_special_tokens=True)
        if len(pos_label_token_ids) > self.max_label_length:
            pos_label_token_ids = pos_label_token_ids[:self.max_label_length-1] + [self.tokenizer.sep_token_id]
        
        negs_token_ids = []
        for neg_idx in hard_negs_indices:
            neg_label_token_ids = self.tokenizer.encode(self.id2label[neg_idx], add_special_tokens=True)
            if len(neg_label_token_ids) > self.max_label_length:
                neg_label_token_ids = neg_label_token_ids[:self.max_label_length-1] + [self.tokenizer.sep_token_id]
            negs_token_ids.append(neg_label_token_ids)

        return (doc_token_ids, (sampled_pos_ind, pos_indices), pos_label_token_ids, index, hard_negs_indices, negs_token_ids) 
    @classmethod
    def get_collate_fn(cls, args, tokenizer):
        def create_biencoder_input2(features):
            batch_pos_labels = []
            random_pos_indices = []
            docs_list = []
            labels_list = []
            batch_indices = []
            batch_all_labels_indices = []
            batch_labels_set = []
            batch_negs_label_list = []
            for item in features:
                docs_list.append(item[0])
                random_pos_indices.append(item[1][0])
                batch_pos_labels.append(item[1][1])
                batch_all_labels_indices.append(item[1][0])
                labels_list.append(item[2])
                batch_indices.append(item[3])
            
            for item in features:
                labels_list.extend(item[5])
                batch_all_labels_indices.extend(item[4])


            assert len(labels_list) == len(batch_all_labels_indices)

            max_doc_len = max([len(doc) for doc in docs_list])
            docs_list = [doc + [tokenizer.pad_token_id]*(max_doc_len-len(doc)) for doc in docs_list]
            doc_tensor = torch.LongTensor(np.array(docs_list))

            max_label_len = max([len(l) for l in labels_list])
            labels_list = [l + [tokenizer.pad_token_id]*(max_label_len-len(l)) for l in labels_list]
            labels_tensor = torch.LongTensor(np.array(labels_list))

            batch_size = len(batch_pos_labels)

            batch_labels_set = set(batch_all_labels_indices)
            batch_all_labels_indices = np.array(batch_all_labels_indices)
            batch_Y = np.zeros((batch_size, len(batch_all_labels_indices)))
            for (i, one_doc_pos_labels) in enumerate(batch_pos_labels):
                intersection = set(one_doc_pos_labels).intersection(batch_labels_set)
                result = np.zeros(len(batch_all_labels_indices),  dtype=np.float32)
                for pos_idx in intersection:
                    result += (pos_idx == batch_all_labels_indices)   
                batch_Y[i] = result  

            batch_data = {}
            batch_data["doc_tensor"] = doc_tensor
            batch_data["doc_attention"] = (doc_tensor != tokenizer.pad_token_id).long()
            batch_data["labels_tensor"] = labels_tensor
            batch_data["labels_attention"] = (labels_tensor != tokenizer.pad_token_id).long()
            batch_data["true_Y"] = torch.from_numpy(batch_Y)
            batch_data["Y_shortlist"] = torch.LongTensor(random_pos_indices)
            return batch_data
        return create_biencoder_input2

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from
    the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num = num_training_steps - current_step
        den = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(num) / float(den))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class TripletMarginLossOHNM(nn.Module):
    r""" Triplet Margin Loss with Online Hard Ne
    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False, tau=0.1, num_violators=False):
        super(TripletMarginLossOHNM, self).__init__()
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.where(target == 0, input, torch.full_like(input, -10))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=self.k)
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = loss != 0 #torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        if self.apply_softmax:
            prob = torch.softmax(sim_n/self.tau * mask, dim=1)
            loss = loss * prob
        reduced_loss = loss.mean() #self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss


def get_embedding(args, doc_path, model, tokenizer, id2label, encode_label, encode_title, batch_size):
    if encode_label:
        seq_length = args.max_label_length
    else:
        seq_length = args.max_seq_length
    embedding_dataset = EmbeddingDataset(args, doc_path, tokenizer, id2label, 
                                            encode_label, encode_title, max_seq_length=seq_length)
    embedding_dataloader = DataLoader(embedding_dataset,
                                  collate_fn=embedding_dataset.get_collate_fn(args, tokenizer),
                                  batch_size=batch_size,
                                  num_workers=10)
    model.eval()
    embeddings = np.zeros((len(embedding_dataset), 768), dtype=float)
    data_index = []
    with torch.no_grad():
        count = 0
        for batch_data in embedding_dataloader:
            batch_size = batch_data['batch_size']
            batch_input_docs = batch_data["doc_tensor"]
            batch_input_docs_attention = batch_data["doc_attention"]
            batch_index = batch_data["batch_indices"]
            data_index.extend(batch_index)
            doc_embedding = model.encode_doc(batch_input_docs.to(args.device), batch_input_docs_attention.to(args.device))
            embeddings[count:count+batch_size, :] = doc_embedding.detach().cpu().numpy()
            count += batch_size
    assert data_index == list(range(len(embeddings)))
    torch.cuda.empty_cache()
    return embeddings


def top_k_metrics(D, I, label_indices, inst_uids, label_uids, topk_list=[1, 3, 5]):
    topk = max(topk_list)
    qrels = {}
    results = {}
    for idx, targets in enumerate(label_indices):
        qrels[inst_uids[idx]] = {label_uids[t]: 1 for t in targets}
        results[inst_uids[idx]] = {label_uids[I[idx][k]]: float(D[idx][k]) for k in range(topk)}

    return EvaluateRetrieval.evaluate(qrels, results, topk_list)