"""
Model with LLaVA like projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSeq2SeqLM, AutoModel)
from .utils import SimpleResBlock

class Model2(nn.Module):
    def __init__(self,                 
                vision_model_path='facebook/dinov2-small',
                lm_path='t5-small',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder = AutoModel.from_pretrained(vision_model_path)
        self.ln_vision = nn.LayerNorm(self.visual_encoder.config.hidden_size)
        print('FREEZING VISUAL ENCODER')
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_path)
        lm_config = AutoConfig.from_pretrained(lm_path)
        lm_config.dense_act_fn = "gelu"
        self.lm_decoder = AutoModelForSeq2SeqLM.from_pretrained(
            lm_path, config=lm_config
        )
        print('FREEZING LANGUAGE DECODER')
        for name, param in self.lm_decoder.named_parameters():
            param.requires_grad = False
            param.data = param.data
        self.lm_decoder.eval()
        
        mlp_depth = 3
        modules = [nn.Linear(self.visual_encoder.config.hidden_size, self.lm_decoder.config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(SimpleResBlock(self.lm_decoder.config.hidden_size))
        self.proj = nn.Sequential(*modules)
        
    def forward(self, samples):
        image_embeds = self.ln_vision(self.visual_encoder(samples['pixel_values']).last_hidden_state)
        image_embeds = F.dropout(image_embeds, p=0.1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        lm_inputs = self.proj(image_embeds)
        lm_atts = torch.ones(lm_inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        targets = samples['input_ids'].masked_fill(samples['input_ids'] == self.lm_tokenizer.pad_token_id, -100)       
        outputs = self.lm_decoder(inputs_embeds=lm_inputs, 
                                    attention_mask=lm_atts, 
                                    decoder_attention_mask=samples['attention_mask'],        
                                    labels=targets,
                                    return_dict=True,   
                                    )  
        return outputs.loss
        
    def generate(self, samples, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, 
                 temperature=0.7, num_captions=1, repetition_penalty=1.0, length_penalty=1.0):
        
        image_embeds = self.ln_vision(self.visual_encoder(samples['pixel_values']).last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        lm_inputs = self.proj(image_embeds)
        lm_atts = torch.ones(lm_inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)     
        
        if sample:
            outputs = self.lm_decoder.generate(
                inputs_embeds=lm_inputs,
                attention_mask=lm_atts,
                do_sample=sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        else:
            outputs = self.lm_decoder.generate(
                inputs_embeds=lm_inputs,
                attention_mask=lm_atts,
                do_sample=sample,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        output_text = self.lm_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
    

def model2_decoder(pretrained='',**kwargs):
    model = Model2(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    