import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.opt.modeling_opt import *
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import *

class InstructGLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids= None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask):   # For Inference text Generation
        # Notably, our input here is numberical inputs_embeds, i.e. we already map inputs_ids to embeddings in pretrain.py via 'first_model'
        self.eval()
        in_embeds=in_embeds
        attention_mask=attention_mask

        output = self.generate(    
            inputs_embeds=in_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=80,
        )

        return output

class OptGLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        head_mask= None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask):   # For Inference text Generation
        # Notably, our input here is numberical inputs_embeds, i.e. we already map inputs_ids to embeddings in pretrain.py via 'first_model'
        self.eval()
        in_embeds=in_embeds
        attention_mask=attention_mask

        output = self.generate(
            inputs_embeds=in_embeds,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_k=25,
            top_p=0.9,
            no_repeat_ngram_size=10,
            early_stopping=True
        )

        return output