import sys
sys.path.append("../")
import copy
import logging
logger = logging.getLogger(__name__)

from transformers import BartConfig, BartTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from .paie import PAIE
from .single_prompt import BartSingleArg
from utils import EXTERNAL_TOKENS
from processors.processor_multiarg import MultiargProcessor


MODEL_CLASSES = {
    'paie': (BartConfig, PAIE, BartTokenizerFast),
    'base': (BartConfig, BartSingleArg, BartTokenizerFast)
}


def build_model(args, model_type):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    if args.inference_only:
        config = config_class.from_pretrained(args.inference_model_path)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.context_representation = args.context_representation

    # length
    config.max_enc_seq_length = args.max_enc_seq_length
    config.max_dec_seq_length= args.max_dec_seq_length
    config.max_prompt_seq_length=args.max_prompt_seq_length
    config.max_span_length = args.max_span_length

    config.bipartite = args.bipartite
    config.matching_method_train = args.matching_method_train

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, add_special_tokens=True)
    if args.inference_only:
        model = model_class.from_pretrained(args.inference_model_path, from_tf=bool('.ckpt' in args.inference_model_path), config=config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # Add trigger special tokens and continuous token (maybe in prompt)
    new_token_list = copy.deepcopy(EXTERNAL_TOKENS)
    prompts = MultiargProcessor._read_prompt_group(args.prompt_path)
    for event_type, prompt in prompts.items():
        token_list = prompt.split()
        for token in token_list:
            if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                new_token_list.append(token)
    tokenizer.add_tokens(new_token_list)   
    logger.info("Add tokens: {}".format(new_token_list))      
    model.resize_token_embeddings(len(tokenizer))

    if args.inference_only:
        optimizer, scheduler = None, None
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    return model, tokenizer, optimizer, scheduler