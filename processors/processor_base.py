import csv
import json
import ipdb
import jsonlines
import torch

from random import sample
from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import copy                             
import logging
logger = logging.getLogger(__name__)


class Event:
    def __init__(self, doc_id, sent_id, sent, event_type, event_trigger, event_args, full_text, first_word_locs=None):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = event_type
        self.trigger = event_trigger
        self.args = event_args
        
        self.full_text = full_text
        self.first_word_locs = first_word_locs


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {} ({}, {})\n".format(arg['role'], arg['text'], arg['start'], arg['end'])
        s += "----------------------------------------------\n"
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                 enc_text, dec_text,
                 enc_tokens, dec_tokens, 
                 old_tok_to_new_tok_index,  
                 event_type, event_trigger, argument_type,
                 enc_input_ids, enc_mask_ids, 
                 dec_input_ids, dec_mask_ids,
                 answer_text, start_position=None, end_position=None):

        self.example_id = example_id
        self.feature_id = feature_id
        self.enc_text = enc_text
        self.dec_text = dec_text
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.argument_type = argument_type
        
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.dec_input_ids = dec_input_ids
        self.dec_mask_ids = dec_mask_ids

        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "argument_type: {}\n".format(self.argument_type)
        s += "enc_tokens: {}\n".format(self.enc_tokens)
        s += "dec_tokens: {}\n".format(self.dec_tokens)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_input_ids: {}\n".format(self.dec_input_ids)
        s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
        
        s += "answer_text: {}\n".format(self.answer_text)
        s += "start_position: {}\n".format(self.start_position)
        s += "end_position: {}\n".format(self.end_position) 
        return s


class DSET_processor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
        self.collate_fn = None


    def _read_jsonlines(self, input_file):
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines


    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)


    def _read_roles(self, role_path):
        template_dict = {}
        role_dict = {}

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                
                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)

        return template_dict, role_dict


    def _create_example_ace(self, lines):
        examples = []
        for doc_idx, line in enumerate(lines):
            if not line['event']:
                continue
            events = line['event']
            offset = line['s_start']
            full_text = copy.deepcopy(line['sentence'])
            text = line['sentence']
            for event_idx, event in enumerate(events):
                event_type = event[0][1]
                event_trigger = dict()
                start = event[0][0] - offset; end = start+1
                event_trigger['start'] = start; event_trigger['end'] = end
                event_trigger['text'] = " ".join(text[start:end])
                event_trigger['offset'] = offset

                event_args = list()
                for arg_info in event[1:]:
                    arg = dict()
                    start = arg_info[0]-offset; end = arg_info[1]-offset+1
                    role = arg_info[2]
                    arg['start'] = start; arg['end'] = end
                    arg['role'] = role; arg['text'] = " ".join(text[start:end])
                    event_args.append(arg)

                examples.append(Event(doc_idx, event_idx, text, event_type, event_trigger, event_args, full_text))
            
        print("{} examples collected.".format(len(examples)))
        return examples


    def _create_example_rams(self, lines):
        # maximum doc length is 543 in train (max input ids 803), 394 in dev, 478 in test
        # too long, so we use a window to cut the sentences.
        W = self.args.window_size
        assert(W%2==0)
        all_args_num = 0

        examples = []
        for line in lines:
            if len(line["evt_triggers"]) == 0:
                continue
            doc_key = line["doc_key"]
            events = line["evt_triggers"]

            full_text = copy.deepcopy(list(chain(*line['sentences'])))
            cut_text = list(chain(*line['sentences']))
            sent_length = sum([len(sent) for sent in line['sentences']])

            text_tmp = []
            first_word_locs = []
            for sent in line["sentences"]:
                first_word_locs.append(len(text_tmp))
                text_tmp += sent

            for event_idx, event in enumerate(events):                
                event_trigger = dict()
                event_trigger['start'] = event[0]
                event_trigger['end'] = event[1]+1
                event_trigger['text'] = " ".join(full_text[event_trigger['start']:event_trigger['end']])
                event_type = event[2][0][0]

                offset, min_s, max_e = 0, 0, W+1
                event_trigger['offset'] = offset
                if sent_length > W+1:
                    if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                        cut_text = full_text[:(W+1)]
                    else:   # trigger word is located at the latter of the sents
                        offset = sent_length - (W+1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        event_trigger['offset'] = offset
                        cut_text = full_text[-(W+1):]

                event_args = list()
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # match trigger span    
                        all_args_num += 1

                        evt_arg = dict()
                        evt_arg['start'] = arg_info[1][0]
                        evt_arg['end'] = arg_info[1][1]+1
                        evt_arg['text'] = " ".join(full_text[evt_arg['start']:evt_arg['end']])
                        evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                        if evt_arg['start']<min_s or evt_arg['end']>max_e:
                            self.invalid_arg_num += 1
                        else:
                            evt_arg['start'] -= offset
                            evt_arg['end'] -= offset 
                            event_args.append(evt_arg)

                if event_idx > 0:
                    examples.append(Event(doc_key+str(event_idx), None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
                else:
                    examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
            
        print("{} examples collected. {} arguments dropped.".format(len(examples), self.invalid_arg_num))
        return examples


    def _create_example_wikievent(self, lines):
        W = self.args.window_size
        assert(W%2==0)
        all_args_num = 0

        examples = []
        for line in lines:
            entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
            events = line["event_mentions"]
            if not events:
                continue
            doc_key = line["doc_id"]
            full_text = line['tokens']
            sent_length = len(full_text)

            curr_loc = 0
            first_word_locs = []
            for sent in line["sentences"]:
                first_word_locs.append(curr_loc)
                curr_loc += len(sent[0])

            for event in events:
                event_type = event['event_type']
                cut_text = full_text
                event_trigger = event['trigger']

                offset, min_s, max_e = 0, 0, W+1
                if sent_length > W+1:
                    if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                        cut_text = full_text[:(W+1)]
                    elif event_trigger['start'] >= sent_length-W/2:   # trigger word is located at the latter of the sents
                        offset = sent_length - (W+1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[-(W+1):]
                    else:
                        offset = event_trigger['start'] - W//2
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[offset:(offset+W+1)]
                event_trigger['offset'] = offset
                        
                event_args = list()
                for arg_info in event['arguments']:
                    all_args_num += 1

                    evt_arg = dict()
                    arg_entity = entity_dict[arg_info['entity_id']]
                    evt_arg['start'] = arg_entity['start']
                    evt_arg['end'] = arg_entity['end']
                    evt_arg['text'] = arg_info['text']
                    evt_arg['role'] = arg_info['role']
                    if evt_arg['start']<min_s or evt_arg['end']>max_e:
                        self.invalid_arg_num += 1
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset 
                        event_args.append(evt_arg)
                examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))

        logger.info("{} examples collected. {} dropped.".format(len(examples), self.invalid_arg_num))
        return examples


    def create_example(self, file_path):
        self.invalid_arg_num = 0
        if self.args.dataset_type=='ace_eeqa':
            lines = self._read_jsonlines(file_path)
            return self._create_example_ace(lines)
        elif self.args.dataset_type=='rams':
            lines = self._read_jsonlines(file_path)
            return self._create_example_rams(lines)
        elif self.args.dataset_type=='wikievent':
            lines = self._read_jsonlines(file_path)
            return self._create_example_wikievent(lines)
        else:
            raise NotImplementedError()
    

    def convert_examples_to_features(self, examples):
        features = []
        for (example_idx, example) in enumerate(examples):
            sent = example.sent  
            event_type = example.type
            event_args = example.args
            event_trigger = example.trigger['text']
            event_args_name = [arg['role'] for arg in event_args]
            enc_text = " ".join(sent)

            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                old_tok_to_char_index.append(curr)
                curr += len(tok)+1
            assert(len(old_tok_to_char_index)==len(sent))

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            enc_tokens = self.tokenizer.convert_ids_to_tokens(enc_input_ids)  
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for char_idx in old_tok_to_char_index:
                new_tok = enc.char_to_token(char_idx)
                old_tok_to_new_tok_index.append(new_tok)    
    
            for arg in self.argument_dict[event_type.replace(':', '.')]:
                dec_text = 'Argument ' + arg + ' in ' + event_trigger + ' event ?' + " "
                     
                dec = self.tokenizer(dec_text)
                dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
                dec_tokens = self.tokenizer.convert_ids_to_tokens(dec_input_ids) 
                while len(dec_input_ids) < self.args.max_dec_seq_length:
                    dec_input_ids.append(self.tokenizer.pad_token_id)
                    dec_mask_ids.append(self.args.pad_mask_token)
        
                start_position, end_position, answer_text = None, None, None
                if arg in event_args_name:
                    arg_idx = event_args_name.index(arg)
                    event_arg_info = event_args[arg_idx]
                    answer_text = event_arg_info['text']
                    # index before BPE, plus 1 because having inserted start token
                    start_old, end_old = event_arg_info['start'], event_arg_info['end']
                    start_position = old_tok_to_new_tok_index[start_old]
                    end_position = old_tok_to_new_tok_index[end_old] if end_old<len(old_tok_to_new_tok_index) else old_tok_to_new_tok_index[-1]+1 
                else:
                    start_position, end_position = 0, 0
                    answer_text = "__ No answer __"

                feature_idx = len(features)
                features.append(
                      InputFeatures(example_idx, feature_idx, 
                                    enc_text, dec_text,
                                    enc_tokens, dec_tokens,
                                    old_tok_to_new_tok_index,
                                    event_type, event_trigger, arg,
                                    enc_input_ids, enc_mask_ids, 
                                    dec_input_ids, dec_mask_ids,
                                    answer_text, start_position, end_position
                                )
                )
        return features

    
    def convert_features_to_dataset(self, features):

        all_enc_input_ids = torch.tensor([f.enc_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_enc_mask_ids = torch.tensor([f.enc_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_input_ids = torch.tensor([f.dec_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_mask_ids = torch.tensor([f.dec_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        
        all_start_positions = torch.tensor([f.start_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_end_positions = torch.tensor([f.end_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_example_idx = torch.tensor([f.example_id for f in features], \
            dtype=torch.long).to(self.args.device)
        all_feature_idx = torch.tensor([f.feature_id for f in features], \
            dtype=torch.long).to(self.args.device)

        dataset = TensorDataset(all_enc_input_ids, all_enc_mask_ids,
                                all_dec_input_ids, all_dec_mask_ids,
                                all_start_positions, all_end_positions,
                                all_example_idx, all_feature_idx,
                            )
        return dataset


    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        if set_type=='train':
            file_path = self.args.train_file
        elif set_type=='dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file
        
        examples = self.create_example(file_path)
        if set_type=='train' and self.args.keep_ratio<1.0:
            sample_num = int(len(examples)*self.args.keep_ratio)
            examples = sample(examples, sample_num)
            logger.info("Few shot setting: keep ratio {}. Only {} training samples remained.".format(\
                self.args.keep_ratio, len(examples))
            )

        features = self.convert_examples_to_features(examples)
        dataset = self.convert_features_to_dataset(features)

        if set_type != 'train':
            dataset_sampler = SequentialSampler(dataset)
        else:
            dataset_sampler = RandomSampler(dataset)
        if self.collate_fn:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size)
        return examples, features, dataloader, self.invalid_arg_num