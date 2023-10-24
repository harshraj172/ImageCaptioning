import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForConditionalGeneration)

class BaseModel(nn.Module):
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
        self.ln_vision = LayerNorm()
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False

        self.lm_tokenizer = T5TokenizerFast.from_pretrained(lm_path)
        lm_config = T5Config.from_pretrained(lm_hfpath)
        lm_config.dense_act_fn = "gelu"
        self.lm_decoder = AutoModelForConditionalGeneration.from_pretrained(
            lm_hfpath, config=lm_config
        )
        for name, param in self.lm_decoder.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()
            
        self.proj = nn.Linear(
            self.visual_encoder.config.hidden_size, self.lm_decoder.config.hidden_size
        )

        
    def forward(self, image, caption):
        
        image_embeds = self.ln_vision(self.visual_encoder(image).pooler_output)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        lm_inputs = self.proj(image_embeds)
        lm_atts = torch.ones(lm_inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        text_tokens = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        targets = text_tokens.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)       
     
        outputs = self.lm_decoder(inputs_embeds=lm_inputs, 
                                    attention_mask=lm_atts, 
                                    decoder_attention_mask=samples['attention_mask'],        
                                    labels = targets,
                                    return_dict = True,   
                                    )   
        return outputs.loss
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        
        image_embeds = self.ln_vision(self.visual_encoder(image).pooler_output)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        lm_inputs = self.proj(image_embeds)
        lm_atts = torch.ones(lm_inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)     

        outputs = self.lm_decoder.generate(
            inputs_embeds=lm_inputs,
            attention_mask=lm_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
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
    

def base_decoder(pretrained='',**kwargs):
    model = BaseModel(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    