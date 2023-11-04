"""
Model with LLaVA like projection, CausalLM-LLama/Mistral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoConfig, CLIPConfig, CLIPVisionModel,
                          AutoModelForCausalLM, AutoModel)
from .utils import SimpleResBlock
from .modeling_llama import LlamaForCausalLM

class Model3(nn.Module):
    def __init__(self,             
                config, 
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
        self.factor = int(256/config['max_length'])
        if 'clip' in config['vision_model_path']:
            self.visual_encoder = CLIPVisionModel.from_pretrained(vision_model_path, cache_dir="/p/scratch/ccstdl/raj3/imgcaption-data")
        elif 'dino' in config['vision_model_path']:
            self.visual_encoder = AutoModel.from_pretrained(vision_model_path, cache_dir="/p/scratch/ccstdl/raj3/imgcaption-data")
        else:
            NotImplementedError
        img_hidden_size = self.visual_encoder.config.hidden_size
        self.ln_vision = nn.LayerNorm(img_hidden_size * self.factor)
        print('FREEZING VISUAL ENCODER')
        self.visual_encoder.requires_grad_(False)

        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_path, cache_dir="/p/scratch/ccstdl/raj3/imgcaption-data")
        lm_config = AutoConfig.from_pretrained(lm_path, cache_dir="/p/scratch/ccstdl/raj3/imgcaption-data")
        lm_config.dense_act_fn = "gelu"
        self.lm_decoder = LlamaForCausalLM.from_pretrained(
            lm_path, config=lm_config, cache_dir="/p/scratch/ccstdl/raj3/imgcaption-data"
        )
        print('FREEZING LANGUAGE DECODER')
        self.lm_decoder.requires_grad_(False)
        
        mlp_depth = 4
        modules = [nn.Linear(img_hidden_size * self.factor, self.lm_decoder.config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(SimpleResBlock(self.lm_decoder.config.hidden_size))
        self.proj = nn.Sequential(*modules)
        
    def forward(self, samples):
        image_embeds = self.visual_encoder(samples['pixel_values']).last_hidden_state
        image_embeds = image_embeds[:, 1:, :]
        bs, pn, hs = image_embeds.shape
        image_embeds = image_embeds.view(bs, int(pn / self.factor), int(hs * self.factor))
        image_embeds = F.dropout(image_embeds, p=0.1)
        image_embeds = self.ln_vision(image_embeds)
        
        lm_inputs = self.proj(image_embeds)
        lm_atts = torch.ones(lm_inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        targets = samples['input_ids'].masked_fill(samples['input_ids'] == self.lm_tokenizer.pad_token_id, -100)       
        outputs = self.lm_decoder(inputs_embeds=lm_inputs,
                                    attention_mask=lm_atts, 
                                    # decoder_attention_mask=samples['attention_mask'],        
                                    labels=targets,
                                    return_dict=True,   
                                    )  
        return outputs.loss
        
    def generate(self, samples, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, 
                 temperature=0.7, num_captions=1, repetition_penalty=1.0, length_penalty=1.0):
        
        image_embeds = self.visual_encoder(samples['pixel_values']).last_hidden_state
        image_embeds = image_embeds[:, 1:, :]
        bs, pn, hs = image_embeds.shape
        image_embeds = image_embeds.view(bs, int(pn / self.factor), int(hs * self.factor))
        image_embeds = self.ln_vision(image_embeds)
        
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
    

def model3_decoder(pretrained='',**kwargs):
    model = Model3(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    