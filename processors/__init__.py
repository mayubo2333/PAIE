import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    'ace_eeqa':{
        "train_file": './data/ace_eeqa/train_convert.json',
        "dev_file": './data/ace_eeqa/dev_convert.json', 
        "test_file": './data/ace_eeqa/test_convert.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json",
    },
    'rams':{
        "train_file": './data/RAMS_1.0/data/train.jsonlines',
        "dev_file": './data/RAMS_1.0/data/dev.jsonlines',
        "test_file": './data/RAMS_1.0/data/test.jsonlines',
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",
    },
    "wikievent":{
        "train_file": './data/WikiEvent/train.jsonl',
        "dev_file": './data/WikiEvent/dev.jsonl',
        "test_file": './data/WikiEvent/test.jsonl',
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",
    },
}


def build_processor(args, tokenizer):
    if args.dataset_type not in _DATASET_DIR: 
        raise NotImplementedError("Please use valid dataset name")
    args.train_file=_DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']

    if args.model_type=="base":
        with open(_DATASET_DIR[args.dataset_type]['max_span_num_file']) as f:
            args.max_span_num_dict = json.load(f)

    processor = MultiargProcessor(args, tokenizer)
    return processor

