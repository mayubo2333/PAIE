import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
    ):

        self.cfg = cfg
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self._init_metric()


    def _init_metric(self):
        self.metric = {
            "global_steps": 0,
            "smooth_loss": 0.0,
        }


    def write_log(self):
        logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
        logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
        self.metric['smooth_loss'] = 0.0


    def train_one_step(self):
        self.model.train()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        inputs = self.convert_batch_to_inputs(batch)
        loss, _= self.model(**inputs)

        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()

        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        
        self.metric['smooth_loss'] += loss.item()/self.cfg.logging_steps
        if (self.metric['global_steps']+1)%self.cfg.gradient_accumulation_steps==0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.metric['global_steps'] += 1


    def convert_batch_to_inputs(self, batch):
        raise NotImplementedError()


class Trainer(BaseTrainer):
    def __init__(self, cfg=None, data_loader=None, model=None, optimizer=None, scheduler=None):
        super().__init__(cfg, data_loader, model, optimizer, scheduler)

    
    def convert_batch_to_inputs(self, batch):
        if self.cfg.model_type=="paie":
            inputs = {
                'enc_input_ids':  batch[0].to(self.cfg.device), 
                'enc_mask_ids':   batch[1].to(self.cfg.device), 
                'dec_prompt_ids':           batch[4].to(self.cfg.device),
                'dec_prompt_mask_ids':      batch[5].to(self.cfg.device),
                'target_info':              batch[6], 
                'old_tok_to_new_tok_indexs':batch[7],
                'arg_joint_prompts':        batch[8],
                'arg_list':       batch[9],
            }
        elif self.cfg.model_type=="base":
            inputs = {
                'enc_input_ids':  batch[0].to(self.cfg.device), 
                'enc_mask_ids':   batch[1].to(self.cfg.device), 
                'decoder_prompt_ids_list':      [item.to(self.cfg.device) for item in batch[2]], 
                'decoder_prompt_mask_list': [item.to(self.cfg.device) for item in batch[3]],
                'arg_list':       batch[9],
                'decoder_prompt_start_positions_list': [item.to(self.cfg.device) for item in batch[12]],
                'decoder_prompt_end_positions_list': [item.to(self.cfg.device) for item in batch[13]],
                'start_position_ids': [item.to(self.cfg.device) for item in batch[14]],
                'end_position_ids': [item.to(self.cfg.device) for item in batch[15]],
            }

        return inputs
