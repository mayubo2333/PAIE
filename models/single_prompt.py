import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel


class BartSingleArg(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.w_prompt_start = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end = nn.Parameter(torch.rand(config.d_model, ))

        self.model._init_weights(self.w_prompt_start)
        self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='sum')
        self.logsoft_fct = nn.LogSoftmax(dim=-1)


    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        decoder_prompt_ids_list=None,
        decoder_prompt_mask_list=None,
        decoder_prompt_start_positions_list=None,
        decoder_prompt_end_positions_list=None,
        start_position_ids=None,
        end_position_ids=None,
        arg_list=None,
    ):
        context_outputs = self.model(
            enc_input_ids,
            attention_mask=enc_mask_ids,
            return_dict=True,
        )
        context_encoder_outputs = context_outputs.encoder_last_hidden_state
        context_decoder_outputs = context_outputs.last_hidden_state

        logit_lists = list()
        total_loss = list()
        for i, (decoder_prompt_ids, decoder_prompt_mask, decoder_prompt_start_positions, decoder_prompt_end_positions) in \
            enumerate(zip(decoder_prompt_ids_list, decoder_prompt_mask_list, decoder_prompt_start_positions_list, decoder_prompt_end_positions_list)):

            prompt_decoder_outputs = self.model.decoder(
                input_ids=decoder_prompt_ids,
                attention_mask=decoder_prompt_mask,
                encoder_hidden_states=context_encoder_outputs[i:i+1].repeat(decoder_prompt_ids.size(0), 1, 1),
                encoder_attention_mask=enc_mask_ids[i:i+1].repeat(decoder_prompt_ids.size(0), 1),
            )
            prompt_decoder_outputs = prompt_decoder_outputs.last_hidden_state   #[Arg_num, Query_L, H]

            for j, (p_start, p_end, prompt_decoder_output) in enumerate(zip(decoder_prompt_start_positions, decoder_prompt_end_positions, prompt_decoder_outputs)):
                prompt_query_sub = prompt_decoder_output[p_start:p_end]
                prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                prompt_query = torch.cat((prompt_query, prompt_query_sub), dim=0) if j>0 else prompt_query_sub
            
            start_prompt_query = (prompt_query*self.w_prompt_start[None, :]).unsqueeze(-1)        #[Arg_num, H, 1]
            end_prompt_query = (prompt_query*self.w_prompt_end[None, :]).unsqueeze(-1)            # [Arg_num, H, 1]

            start_logits = torch.bmm(context_decoder_outputs[i:i+1].repeat(len(start_prompt_query),1,1), start_prompt_query).squeeze(-1)             # [Arg_num, L]
            end_logits = torch.bmm(context_decoder_outputs[i:i+1].repeat(len(end_prompt_query),1,1), end_prompt_query).squeeze(-1)
            start_logits = start_logits.masked_fill_(~enc_mask_ids[i:i+1].repeat(len(start_prompt_query),1).bool(), -20)
            end_logits = end_logits.masked_fill_(~enc_mask_ids[i:i+1].repeat(len(start_prompt_query),1).bool(), -20)

            if start_position_ids is not None and end_position_ids is not None:
                start_logsoftmax = self.logsoft_fct(start_logits)
                end_logsoftmax = self.logsoft_fct(end_logits)
                start_loss = -torch.mean(torch.sum(start_position_ids[i]*start_logsoftmax, dim=1), dim=0)
                end_loss = -torch.mean(torch.sum(end_position_ids[i]*end_logsoftmax, dim=1), dim=0)
                total_loss.append((start_loss + end_loss) / 2)

            output = dict()
            for j, arg_role in enumerate(arg_list[i]):
                start_logits_list, end_logits_list = [start_logits[j]], [end_logits[j]]
                output[arg_role] = [start_logits_list, end_logits_list]
            logit_lists.append(output)

        if total_loss:
            return torch.mean(torch.stack(total_loss)), logit_lists
        else:
            return [], logit_lists
