import sys
import os
import argparse
import json
import random

def read_pseudo_trn(trn_path):
    pseudo_labels = []
    docs_id = []
    with open(trn_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            doc_id = instance["uid"]
            labels = instance["target_ind"]
            docs_id.append(doc_id)
            pseudo_labels.append(labels)
    return docs_id, pseudo_labels


def read_shortlist(shortlist_path):
    docs_id = []
    all_topk_labels = []
    with open(shortlist_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            doc_id = instance["doc_id"]
            topk_labels = instance["pred_labels"]
            docs_id.append(doc_id)
            all_topk_labels.append(topk_labels)
    return docs_id, all_topk_labels


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_trn_path", type=str)
    parser.add_argument('--shortlist_path', type=str, default=None, help='')
    parser.add_argument('--p1_output_path', type=str, default=None, help='')


    args = parser.parse_args()
    pseudo_trn_docs_id, llm_pseudo_labels = read_pseudo_trn(args.pseudo_trn_path)
    shortlist_docs_id, pred_topk_labels = read_shortlist(args.shortlist_path)
    assert pseudo_trn_docs_id == shortlist_docs_id
    
    p1 = 0.0
    for i in range(len(pred_topk_labels)):
        pred_top1_label = pred_topk_labels[i][0]
        if pred_top1_label in llm_pseudo_labels[i][:1]:
            p1 += 1 
    p1 = p1 / len(pred_topk_labels)

    print(f"p1 evaluation with llm: {p1}")
    with open(args.p1_output_path, "w") as fw:
        fw.write(str(p1))
if __name__ == "__main__":
    main()
