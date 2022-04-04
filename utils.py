import re
import time
import string
import random
import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


EXTERNAL_TOKENS = ['<t>', '</t>']
_PREDEFINED_QUERY_TEMPLATE = "Argument: {arg:}. Trigger: {trigger:} "


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_time(f):
    def run(**kw):
        time1 = time.time()
        result = f(**kw)
        time2 = time.time()
        logger.info("The time of executing {}: {}".format(f.__name__, time2-time1))
        return result
    return run


def hungarian_matcher(predicted_spans, target_spans):
    """
    Args:
        predictions: prediction of one arg role type, list of [s,e]
        targets: target of one arg role type, list of [s,e]
    Return:
        (index_i, index_j) where index_i in prediction, index_j in target 
    """
    # L1 cost between spans
    cost_spans = torch.cdist(torch.FloatTensor(predicted_spans).unsqueeze(0), torch.FloatTensor(target_spans).unsqueeze(0), p=1)
    indices = linear_sum_assignment(cost_spans.squeeze(0)) 
    return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace. (Squad Style) """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    s_normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    return s_normalized


def get_best_span(start_logit, end_logit, old_tok_to_new_tok_index, max_span_length):
    # time consuming
    best_score = start_logit[0] + end_logit[0]
    best_answer_span = (0, 0)
    context_length = len(old_tok_to_new_tok_index)

    for start in range(context_length):
        for end in range(start+1, min(context_length, start+max_span_length+1)):
            start_index = old_tok_to_new_tok_index[start][0] # use start token idx
            end_index = old_tok_to_new_tok_index[end-1][1] 

            score = start_logit[start_index] + end_logit[end_index]
            answer_span = (start_index, end_index)
            if score > best_score:
                best_score = score
                best_answer_span = answer_span

    return best_answer_span


def get_best_span_simple(start_logit, end_logit):
    # simple constraint version
    _, s_idx = torch.max(start_logit, dim=0)
    _, e_idx = torch.max(end_logit[s_idx:], dim=0)
    return [s_idx, s_idx+e_idx]


def get_sentence_idx(first_word_locs, word_loc):
    sent_idx = -1
    for i, first_word_loc in enumerate(first_word_locs):
        if word_loc>=first_word_loc:
            sent_idx = i
        else:
            break
    return sent_idx


def get_maxtrix_value(X):
    """
    input: batch of matrices. [B, M, N]
    output: indexes of argmax for each matrix in batch. [B, 2]
    """
    t1 = time.time()
    col_max, col_max_loc = X.max(dim=-1)
    _, row_max_loc = col_max.max(dim=-1)
    t2 = time.time()
    cal_time = (t2-t1)

    row_index = row_max_loc
    col_index = col_max_loc[torch.arange(row_max_loc.size(0)), row_index]

    return torch.stack((row_index, col_index)).T, cal_time


def get_best_indexes(features, feature_id_list, start_logit_list, end_logit_list, args):
    t1 = time.time()
    start_logits = torch.stack(tuple(start_logit_list)).unsqueeze(-1)         # [B, M, 1]
    end_logits = torch.stack(tuple(end_logit_list)).unsqueeze(1)              # [B, 1, M]
    scores = (start_logits + end_logits).float()
    t2 = time.time()
    score_time = t2 - t1

    def generate_mask(feature):
        mask = torch.zeros((args.max_enc_seq_length, args.max_enc_seq_length), dtype=float, device=args.device)
        context_length = len(feature.old_tok_to_new_tok_index)
        for start in range(context_length):
            start_index = feature.old_tok_to_new_tok_index[start][0]
            end_index_list = [feature.old_tok_to_new_tok_index[end-1][1] for end in range(start+1, min(context_length, start+args.max_span_length+1))]
            mask[start_index, end_index_list] = 1.0
        mask[0][0] = 1.0 
        return torch.log(mask).float().unsqueeze(0)
    
    t1 = time.time()
    candidate_masks = {feature_id:generate_mask(features[feature_id]) for feature_id in set(feature_id_list)}
    masks = torch.cat([candidate_masks[feature_id] for feature_id in feature_id_list], dim=0)

    t2 = time.time()
    mask_time = t2-t1
    masked_scores = scores + masks
    max_locs, cal_time = get_maxtrix_value(masked_scores)
    max_locs = [tuple(a) for a in max_locs]

    return max_locs, cal_time, mask_time, score_time


def get_best_index(feature, start_logit, end_logit, max_span_length, max_span_num, delta):
    th = start_logit[0] + end_logit[0]
    answer_span_list = []
    context_length = len(feature.old_tok_to_new_tok_index)

    for start in range(context_length):
        for end in range(start+1, min(context_length, start+max_span_length+1)):
            start_index = feature.old_tok_to_new_tok_index[start][0] # use start token idx
            end_index = feature.old_tok_to_new_tok_index[end-1][1] 

            score = start_logit[start_index] + end_logit[end_index]
            answer_span = (start_index, end_index, score)

            if score > (th+delta):
                answer_span_list.append(answer_span)
    
    if not answer_span_list:
        answer_span_list.append((0, 0, th))
    return filter_spans(answer_span_list, max_span_num)


def filter_spans(candidate_span_list, max_span_num):
    candidate_span_list = sorted(candidate_span_list, key=lambda x:x[2], reverse=True)
    candidate_span_list = [(candidate_span[0], candidate_span[1]) for candidate_span in candidate_span_list]

    def is_intersect(span_1, span_2):
        return False if min(span_1[1], span_2[1]) < max(span_1[0], span_2[0]) else True

    if len(candidate_span_list) == 1:
        answer_span_list = candidate_span_list
    else:
        answer_span_list = []
        while candidate_span_list and len(answer_span_list)<max_span_num:
            selected_span = candidate_span_list[0]
            answer_span_list.append(selected_span)
            candidate_span_list = candidate_span_list[1:]  

            candidate_span_list = [candidate_span for candidate_span in candidate_span_list if not is_intersect(candidate_span, selected_span)]
    return answer_span_list


def check_tensor(tensor, var_name):
    print("******Check*****")
    print("tensor_name: {}".format(var_name))
    print("shape: {}".format(tensor.size()))
    if len(tensor.size())==1 or tensor.size(0)<=3:
        print("value: {}".format(tensor))
    else:
        print("part value: {}".format(tensor[0,:]))
    print("require_grads: {}".format(tensor.requires_grad))
    print("tensor_type: {}".format(tensor.dtype))


from spacy.tokens import Doc
class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def find_head(arg_start, arg_end, doc):
    arg_end -= 1
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    head_text = doc[arg_head]
    return head_text