import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.image_processor import VaeImageProcessor
import os


class LocalEncoderWrapper(nn.Module):
    def __init__(self, name, device, dtype):
        super().__init__()
        self.name = name
        self.projector = nn.Linear(768, 4).to(device, dtype=dtype) 

    def forward(self, x):

        return torch.randn(x.shape[0], 256, 4, device=x.device, dtype=x.dtype)


class HOCompPipeline:
    def __init__(self, base_model_id, local_lora_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        

        
        self.transformer = FluxTransformer2DModel.from_pretrained(
            base_model_id, subfolder="transformer", torch_dtype=dtype
        ).to(device)
        self.transformer.requires_grad_(False)
        
        if os.path.exists(local_lora_path):
            self.transformer.load_lora_weights(local_lora_path, adapter_name="hocomp")
            self.transformer.fuse_lora(lora_scale=1.0)

        self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype).to(device)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer_2")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        self.img_proc = VaeImageProcessor(vae_scale_factor=8)
        
        self.id_encoder = LocalEncoderWrapper("ID_Net", device, dtype)
        self.detail_encoder = LocalEncoderWrapper("Detail_Net", device, dtype)
        
        self.mask_proj = nn.Linear(1, 4).to(device, dtype=dtype)

    def encode_text(self, prompt):
        """Standard T5 Encoding."""
        txt_input = self.tokenizer_2(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder_2(txt_input.input_ids)[0].to(self.dtype)

    def image_to_tokens(self, image):
        """Encodes Image to 4-channel Tokens via VAE."""
        img_t = self.img_proc.preprocess(image, height=1024, width=1024).to(self.device, dtype=self.dtype)
        latents = self.vae.encode(img_t).latent_dist.sample() * self.vae.config.scaling_factor
        # [B, 4, H, W] -> [B, H*W, 4]
        b, c, h, w = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(b, -1, c)

    @torch.no_grad()
    def __call__(self, prompt, bg_img, fg_img, box, num_inference_steps=25):


        

        text_emb = self.encode_text(prompt) # [1, 512, 4096]
        

        bg_tokens = self.image_to_tokens(bg_img)     
        fg_id_tokens = self.id_encoder(fg_img)       
        fg_det_tokens = self.detail_encoder(fg_img)  
        
        H, W = 1024, 1024
        mask = torch.zeros((1, 1024, 1024, 1), device=self.device, dtype=self.dtype)
        mask[:, box[1]:box[3], box[0]:box[2], :] = 1.0

        mask_tokens = F.interpolate(mask.permute(0,3,1,2), size=(128,128), mode="nearest").permute(0,2,3,1).reshape(1, -1, 1)
        mask_tokens = self.mask_proj(mask_tokens) # [1, 16384, 4]
        

        noise = torch.randn(1, 4, 128, 128, device=self.device, dtype=self.dtype)
        noise_tokens = noise.permute(0, 2, 3, 1).reshape(1, -1, 4)
        
        combined_tokens = torch.cat([
            noise_tokens, 
            bg_tokens, 
            fg_id_tokens, 
            fg_det_tokens,
            mask_tokens
        ], dim=1)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        latents_seq = combined_tokens
        
        for t in self.scheduler.timesteps:
            

            output = self.transformer(
                hidden_states=latents_seq,      # <--- 4 Channel Sequence
                encoder_hidden_states=text_emb, # <--- Text
                timestep=t,
                return_dict=False
            )[0]
            
            noise_pred = output[:, :noise_tokens.shape[1], :]

            noise_pred_map = noise_pred.reshape(1, 128, 128, 4).permute(0, 3, 1, 2)
            noise_map = latents_seq[:, :noise_tokens.shape[1], :].reshape(1, 128, 128, 4).permute(0, 3, 1, 2)
            
            updated_noise = self.scheduler.step(noise_pred_map, t, noise_map).prev_sample
            
            updated_noise_tokens = updated_noise.permute(0, 2, 3, 1).reshape(1, -1, 4)
            latents_seq = torch.cat([updated_noise_tokens, bg_tokens, fg_id_tokens, fg_det_tokens, mask_tokens], dim=1)

        image = self.vae.decode(updated_noise / self.vae.config.scaling_factor).sample
        return image[0]