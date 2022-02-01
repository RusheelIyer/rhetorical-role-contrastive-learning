"""Tokenizes the sentences with BertTokenizer as tokenisation costs some time.
"""
import sys
from transformers import RobertaTokenizer
from infer_new import write_in_hsln_format

BERT_VOCAB = "roberta-base"
MAX_SEQ_LENGTH = 128


def tokenize():
    [_, input_json] = sys.argv
    tokenizer = RobertaTokenizer.from_pretrained(BERT_VOCAB,
                                                 do_lower_case=True)
    write_in_hsln_format(input_json, 'datasets/pubmed-20k', tokenizer)

    print("pubmed-20k")


tokenize()
