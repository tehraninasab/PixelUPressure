import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import multiprocessing
import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import argparse
import logging
from datetime import datetime
import torch.distributed as dist
from torchvision.utils import save_image
# lora_utils.py
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
file_handler = logging.FileHandler(f'out/finetune_chexpert_{timestamp}.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_root_path,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.df = pd.read_csv(csv_file)
        self.image_root_path = image_root_path
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        
        if center_crop:
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        self.create_captions()

    def create_captions(self):
        conditions = [
            'No Finding', 'Cardiomegaly','Lung Opacity', 'Edema', 
            'Pneumonia', 'Pneumothorax', 'Pleural Effusion',
            'Support Devices'
        ]
        
        captions = []
        for _, row in self.df.iterrows():
            findings = []
            for condition in conditions:
                if row[condition] == 1:
                    findings.append(condition)
            
            caption = "Chest X-ray showing " + ", ".join(findings) if findings else "Normal chest X-ray with no significant findings"
            captions.append(caption)
        
        self.df['caption'] = captions

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root_path, row['Path'].replace('CheXpert-v1.0/','CheXpert-v1.0_512x512/'))
        caption = row['caption']

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        encoding = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0]
        }

def train_one_epoch(
    accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    dataloader,
    optimizer,
    epoch,
    args
):  
    # decide which modules to sync/accumulate
    modules_to_accumulate = []
    if args.train_unet or args.use_lora or args.use_dora or args.use_bitfit or args.use_difffit:
        modules_to_accumulate.append(unet)
    if args.train_vae:
        modules_to_accumulate.append(vae)
    if args.train_text_encoder:
        modules_to_accumulate.append(text_encoder)
        
    # sanity check: at least UNet should be in there
    assert modules_to_accumulate, "You must train at least one module!"
        
    if args.train_unet:
        unet.train()
    if args.train_vae:
        vae.train()
    if args.train_text_encoder:
        text_encoder.train()
        
    logger.info(f"Starting training for epoch {epoch}")
    total_loss = 0
    unwrapped_vae = accelerator.unwrap_model(vae)
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
        if args.debug and step >= 2:
            logger.info("Debug mode: stopping after 2 steps")
            break
        with accelerator.accumulate(*modules_to_accumulate):
            # 1) ENCODE → latents
            dist = unwrapped_vae.encode(batch["image"]).latent_dist
            latents = dist.sample()
            if not args.train_vae:
                latents = latents.detach()
            latents = latents * unwrapped_vae.config.scaling_factor

            # 2) DIFFUSION NOISE
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3) UNET PREDICTION
            encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=batch["attention_mask"])[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            diff_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean()
            
            # 4) OPTIONAL VAE RECONSTRUCTION
            if args.train_vae:
                recon = unwrapped_vae.decode(latents / unwrapped_vae.config.scaling_factor).sample
                recon_loss = torch.nn.functional.mse_loss(recon, batch["image"])
                loss = diff_loss + args.vae_recon_weight * recon_loss
            else:
                loss = diff_loss

            # 5) BACKPROPAGATION
            # any_grad = any(p.requires_grad for p in unet.parameters())
            # logger.info(f"Any UNet param requires grad? {any_grad}")
            # logger.info(f"Loss requires grad? {loss.requires_grad}")
            if not loss.requires_grad:
                raise RuntimeError("Loss is not connected to any trainable parameters. Check that delta_weight is used in the forward pass.")
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                # collect all parameters into one list
                params_to_clip = []
                if args.train_unet:
                    params_to_clip += list(unet.parameters())
                if args.train_vae:
                    params_to_clip += list(vae.parameters())
                if args.train_text_encoder:
                    params_to_clip += list(text_encoder.parameters())

                # single clip call unscales once, then clips everything
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().item()

        if step % args.save_steps == 0 and epoch % args.save_epochs == 0:
            save_progress(accelerator, unet, vae, text_encoder, args.output_dir, epoch, step, args)


    return total_loss / len(dataloader)

def save_progress(accelerator, unet, vae, text_encoder, output_dir, epoch, step, args):
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch}-{step}")
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Saving checkpoint to {save_path}")

        # Save model weights
        if args.train_unet:
            torch.save(accelerator.unwrap_model(unet).state_dict(), os.path.join(save_path, "unet.pt"))
        if args.train_vae:
            torch.save(accelerator.unwrap_model(vae).state_dict(), os.path.join(save_path, "vae.pt"))
        if args.train_text_encoder:
            torch.save(accelerator.unwrap_model(text_encoder).state_dict(), os.path.join(save_path, "text_encoder.pt"))

        accelerator.save_state(save_path)

        # ---- Generate and save a few validation images ----
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                tokenizer=CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer"),
                scheduler=DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler"),
                safety_checker=None,
                feature_extractor=None,
                torch_dtype=torch.float16,
            ).to(accelerator.device)

            conditions = ["Cardiomegaly", "No Finding", "Pleural Effusion", "Support Devices"]
            n_images_per_condition = 10
            for i, condition in enumerate(conditions):
                image_dir = os.path.join(save_path, "val_images", condition.replace(" ", "_"))
                os.makedirs(image_dir, exist_ok=True)
                prompt = f"Chest X-ray showing {condition}"

                for j in range(n_images_per_condition):
                    image = pipeline(prompt=prompt, num_inference_steps=50).images[0]
                    image.save(os.path.join(image_dir, f"sample_{j}.png"))

                logger.info(f"Saved {n_images_per_condition} validation images at {image_dir}")

            del pipeline
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed to generate validation images: {e}")

def apply_lora_adapters(unet, args):
    # UNet LoRA or DoRA
    peft_config_unet = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        use_dora=args.use_dora  # 👈 add this line
    )

    unet = get_peft_model(unet, peft_config_unet)

    # Freeze everything except LoRA/DoRA params
    for name, p in unet.named_parameters():
        p.requires_grad = "lora_" in name or "dora_" in name  # handles both

    return unet

def apply_bitfit(model):
    for name, param in model.named_parameters():
        if '.bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def apply_difffit(model):
    count = 0
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if hasattr(module, "weight") and module.weight is not None:
                delta = torch.nn.Parameter(torch.zeros_like(module.weight), requires_grad=True)
                module.register_parameter("delta_weight", delta)
                count += delta.numel()

    logger.info(f"✅ DiffFit applied: {count} total trainable delta parameters.")
    return model

def patch_for_difffit(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            original_forward = module.forward
            def new_forward(x, module=module):
                w = module.weight + getattr(module, "delta_weight", 0)
                return torch.nn.functional.linear(x, w, module.bias)
            module.forward = new_forward

        elif isinstance(module, torch.nn.Conv2d):
            original_forward = module.forward
            def new_forward(x, module=module):
                w = module.weight + getattr(module, "delta_weight", 0)
                return torch.nn.functional.conv2d(
                    x, w, module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )
            module.forward = new_forward


def count_parameters(model):
    """
    Count all parameters in the model, regardless of whether they are trainable or not.
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    if __name__ == "__main__":
        multiprocessing.set_start_method('spawn', force=True)
        dist.init_process_group(backend='nccl')
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if dist.get_rank() == 0 else logging.WARN,
    )
    
    # 1) Load all models (unet, vae, text_encoder, tokenizer, scheduler) ———
    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")
    
    # 1) Optionally wrap in LoRA (all PEFT logic lives in lora_utils.py) ———
    if args.use_lora or args.use_dora:
        unet = apply_lora_adapters(unet, args)
    elif args.use_bitfit:
        logger.info("Applying BitFit to UNet...")
        unet = apply_bitfit(unet)
    elif args.use_difffit:
        logger.info("Applying DiffFit to UNet...")
        unet = apply_difffit(unet)
        patch_for_difffit(unet)
        assert any(p.requires_grad for _, p in unet.named_parameters()), "❌ No trainable parameters found in UNet after applying DiffFit"
    
    # 3) Freeze/unfreeze based on args
    #    In LoRA mode, unet/text_encoder already have only LoRA params trainable.
    if not (args.use_lora or args.use_dora or args.use_bitfit or args.use_difffit):
        unet.requires_grad_(args.train_unet)
    
    text_encoder.requires_grad_(args.train_text_encoder)    
    vae.requires_grad_(args.train_vae)    
    
    logger.info(f"UNet trainable parameters: {count_trainable_parameters(unet)}/{count_parameters(unet)}")
    logger.info(f"Text Encoder trainable parameters: {count_trainable_parameters(text_encoder)}/{count_parameters(text_encoder)}")
    logger.info(f"VAE trainable parameters: {count_trainable_parameters(vae)}/{count_parameters(vae)}")
    n_deltas = sum(1 for _, p in unet.named_parameters() if 'delta_weight' in _ and p.requires_grad)
    logger.info(f"Total DiffFit delta weights: {n_deltas}")


    # Setup training data
    train_dataset = CheXpertDataset(
        csv_file=args.train_data_path,
        image_root_path=args.image_root_path,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        multiprocessing_context='spawn'
    )
  
    # after you set requires_grad_ on each component...
    # 4) Build your optimizer over exactly the trainable params
    param_groups = []
    if args.train_unet or args.use_lora or args.use_dora or args.use_bitfit or args.use_difffit:
        param_groups.append({
            "params": [p for n, p in unet.named_parameters() if p.requires_grad],
            "lr":      args.unet_lr,
            "betas":   (args.adam_beta1, args.adam_beta2),
            "weight_decay": args.adam_weight_decay,
            "eps":     args.adam_epsilon,
        })
    if args.train_text_encoder:
        param_groups.append({
            "params": text_encoder.parameters(),
            "lr":      args.text_lr,
            "betas":   (args.adam_beta1, args.adam_beta2),
            "weight_decay": args.adam_weight_decay,
            "eps":     args.adam_epsilon,
        })
    if args.train_vae:
        param_groups.append({
            "params": vae.parameters(),
            "lr":      args.vae_lr,
            "betas":   (args.adam_beta1, args.adam_beta2),
            "weight_decay": args.adam_weight_decay,
            "eps":     args.adam_epsilon,
        })

    optimizer = torch.optim.AdamW(param_groups)
    
    # Prepare everything in one go
    unet, text_encoder, vae, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader
    )

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # Training loop
    for epoch in range(args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            accelerator,
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            train_dataloader,
            optimizer,
            epoch,
            args
        )
        
        if dist.get_rank() == 0:
            logger.info(f"Epoch {epoch}: Average loss = {train_loss}")
        
        if epoch == args.num_train_epochs - 1 and dist.get_rank() == 0:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
            )
            pipeline.save_pretrained(args.output_dir)
        
        if args.debug:
            logger.info("Debug mode: stopping after 1 epoch")
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on CheXpert")
    parser.add_argument("--model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_data_path", type=str, default='dataset/chexpert/chexpert_train.csv')
    parser.add_argument("--image_root_path", type=str, default='/usr/local/faststorage/zahrat/chexpert/')
    parser.add_argument("--output_dir", type=str, default='saved_models/sd15_finetuning')
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--unet_lr",  type=float, default=1e-5, help="learning rate for UNet")
    parser.add_argument("--text_lr",  type=float, default=5e-6, help="learning rate for text encoder")
    parser.add_argument("--vae_lr",   type=float, default=1e-6, help="learning rate for VAE")

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--train_unet", action='store_true', help="Whether to train the UNet model")
    parser.add_argument("--train_vae", action='store_true', help="Whether to train the VAE model")
    parser.add_argument("--train_text_encoder", action='store_true', help="Whether to train the text encoder model")
    parser.add_argument("--vae_recon_weight", type=float, default=1.0, help="weight for VAE reconstruction loss when training the VAE")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode for faster training with fewer epochs")
    parser.add_argument("--use_lora", action="store_true", help="Wrap UNet (and optionally text encoder) with LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", nargs="+", type=str,
                        default=["to_q", "to_k", "to_v", "to_out.0"],
                        help="Which UNet attention sub‐modules to apply LoRA to")
    parser.add_argument("--use_dora", action="store_true", help="Enable DoRA: Decomposed LoRA-style fine-tuning")
    parser.add_argument("--use_bitfit", action="store_true", help="Enable BitFit: train only bias terms")
    parser.add_argument("--use_difffit", action="store_true", help="Enable DiffFit: train only difference of weight deltas")

    
    args = parser.parse_args()
    
    components = "+".join(part for part, flag in [
        ("unet", args.train_unet),
        ("vae",  args.train_vae),
        ("text", args.train_text_encoder),
        (f"dora{args.lora_rank}", args.use_dora),
        (f"lora{args.lora_rank}", args.use_lora and not args.use_dora),
        ("bitfit", args.use_bitfit),
        ("difffit", args.use_difffit)
    ] if flag) or "frozen"
    
    if args.use_lora and args.use_dora:
        logger.warning("Both LoRA and DoRA flags are set. Defaulting to DoRA (use_dora=True).")
    
    args.output_dir = os.path.join(args.output_dir, f"sd15_{components}")
    
    main(args)
