from datetime import datetime
import json, pickle
import timeit, random, lzma
import gzip, os, csv
from collections import defaultdict, Counter

def read_json_list(input_file):
    with gzip.open(input_file) if input_file.endswith(".gz") \
            else open(input_file) as fin:
        for line in fin:
            obj = json.loads(line)
            yield obj



