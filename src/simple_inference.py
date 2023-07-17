import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

from .patch_lora import safe_open, parse_safeloras_embeds, apply_learned_embed_in_clip
def patch_ti(pipe, ti_paths):
    for weight_path in ti_paths.split('|'):
        token = None
        idempotent_token = True

        safeloras = safe_open(weight_path, framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)

        apply_learned_embed_in_clip(
            tok_dict,
            pipe.text_encoder,
            pipe.tokenizer,
            token=token,
            idempotent=idempotent_token,
        )


def main(args):

    prompts = []

    if args.patch_ti is not None:
        print(f"Inference using Ti {args.pretrained_model_name_or_path}")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)            

        patch_ti(pipe, f"{args.pretrained_model_name_or_path}/step_inv_{args.patch_ti.max_train_steps_ti}.safetensors")

        inverted_tokens = args.patch_ti.placeholder_tokens.replace('|', '')
        if args.patch_ti.use_template == "object":
            prompts += [f"a photo of {inverted_tokens}"]
        elif args.patch_ti.use_template == "style":
            prompts += [f"a photo in the style of {inverted_tokens}"]
        else:
            raise ValueError("unknown concept type!")          

    if args.multi_concept is not None:
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        model_id = args.pretrained_model_name_or_path
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)                    
        for c, t in args.multi_concept:
            c = c.replace('-', ' ')
            if t == "object":
                prompts += [f"a photo of {c}"]
            elif t == "style":
                prompts += [f"a photo in the style of {c}"]
            else:
                raise ValueError("unknown concept type!")        

            

    torch.manual_seed(1)
    output_folder = f"{args.pretrained_model_name_or_path}/generated_images"
    os.makedirs(output_folder, exist_ok=True)

    for prompt in prompts:
        print(f'Inferencing: {prompt}')
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7, num_images_per_prompt=8).images
        for i, im in enumerate(images):
            im.save(f"{output_folder}/o_{prompt.replace(' ', '-')}_{i}.jpg")  