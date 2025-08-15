import os
import sys
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from safetensors.torch import load_file
from PIL import Image
import numpy as np
from diffusers.models import UNet2DConditionModel
from peft import LoraConfig, get_peft_model, PeftModel
import random


# from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
# from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import argparse

def create_chestxray_caption(row, label_columns):
    findings = [label for label in label_columns if row[label] == 1]
    if findings:
        return "Chest X-ray showing " + ", ".join(findings)
    else:
        return "Normal chest X-ray with no significant findings"

def create_retinopathy_caption(row):
    """
    Create standardized captions for diabetic retinopathy images based on severity level
    
    Severity levels:
    0 - No DR
    1 - Mild DR
    2 - Moderate DR
    3 - Severe DR
    4 - Proliferative DR
    """
    severity_descriptions = {
        0: "no diabetic retinopathy",
        1: "mild non-proliferative diabetic retinopathy",
        2: "moderate non-proliferative diabetic retinopathy",
        3: "severe non-proliferative diabetic retinopathy",
        4: "proliferative diabetic retinopathy"
    }
    
    level = row['level']
    eye_position = row.get('eye_position', None)
    
    if eye_position:
        caption = f"Retinal image of {eye_position} eye showing {severity_descriptions[level]}"
    else:
        caption = f"Retinal image showing {severity_descriptions[level]}"
            
    return caption

def create_isic2019_caption(row):
    disease_mapping = {
            'MEL': 'melanoma',
            'NV': 'melanocytic nevus',
            'BCC': 'basal cell carcinoma',
            'AK': 'actinic keratosis',
            'BKL': 'benign keratosis-like lesion',
            'DF': 'dermatofibroma',
            'VASC': 'vascular lesion',
            'SCC': 'squamous cell carcinoma',
            'UNK': 'unknown',
        }
    
    
    # Add disease information
    for disease in disease_mapping:
        if disease in row and row[disease] == 1:
            full_name = disease_mapping[disease]
            caption = f"a dermoscopic image with {full_name} ({disease})"
            break
    
    # If no disease is marked, use default caption
    if not caption:
        caption = "a dermoscopic image of a normal skin"
        
    return caption
    
def get_valid_output_csv_path(output_csv_path):
    if os.path.isfile(output_csv_path):
        raise FileExistsError(
            f"Output CSV file '{output_csv_path}' already exists. Please choose a different name or delete the existing file."
        )
    return output_csv_path

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

def load_finetuned_pipeline(model_path, model_version, device):
    
    if "lora" in model_version or "dora" in model_version or "bitfit" in model_version or "difffit" in model_version:
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", type=torch.float16)
        
        if "lora" in model_version or "dora" in model_version:
            rank = int(model_version[4:])

            # Load base UNet from pretrained SD1.5

            # Reapply same LoRA configuration used in training
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                use_dora="dora" in model_version # Use Dora if specified
            )

            unet = get_peft_model(unet, lora_config)
        elif "bitfit" in model_version:
            # Apply BitFit configuration
            unet = apply_bitfit(unet)
        elif "difffit" in model_version:
            # Apply DiffFit configuration
            unet = apply_difffit(unet)
            patch_for_difffit(unet)
            assert any(p.requires_grad for _, p in unet.named_parameters()), "❌ No trainable parameters found in UNet after applying DiffFit"

        # Path to your checkpoint
        checkpoint_path = f"{model_path}/model.safetensors"
        
        # Load safetensors file
        unet_weights = load_file(checkpoint_path)

        # Load LoRA weights into your LoRA-wrapped UNet
        unet.load_state_dict(unet_weights, strict=False)
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")

    else:
        # Load the entire pipeline directly from the finetuned model path
        pipe = StableDiffusionPipeline.from_pretrained(
            #'stabilityai/stable-diffusion-2-1',
            'runwayml/stable-diffusion-v1-5',
            safety_checker=None,  # Disable safety checker for medical images
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32
        )
    if "unet" in model_version:
        # Load the UNet model from the specified path
        checkpoint = torch.load(f'{model_path}/unet.pt')
        pipe.unet.load_state_dict(checkpoint)
        print(f"Loaded UNet model from {model_path}/unet.pt")
    if "vae" in model_version:
        # Load the VAE model from the specified path
        checkpoint = torch.load(f'{model_path}/vae.pt')
        pipe.vae.load_state_dict(checkpoint)
        print(f"Loaded VAE model from {model_path}/vae.pt")
    if "text" in model_version:
        # Load the text encoder model from the specified path
        checkpoint = torch.load(f'{model_path}/text_encoder.pt')
        pipe.text_encoder.load_state_dict(checkpoint)
        print(f"Loaded Text Encoder model from {model_path}/text_encoder.pt")

    pipe = pipe.to(device)
    
    # Enable memory optimization if on GPU
    if "cuda" in str(device):
        pipe.enable_attention_slicing()
    
    if "cuda" in str(device):
        pipe.unet = pipe.unet.half()
        pipe.vae = pipe.vae.half()
        pipe.text_encoder = pipe.text_encoder.half()
    
    return pipe

# def load_finetuned_pipeline(model_path, device):
    
#     # Load the entire pipeline directly from the finetuned model path
#     pipe = StableDiffusionPipeline.from_pretrained(
#         #'stabilityai/stable-diffusion-2-1',
#         'runwayml/stable-diffusion-v1-5',
#         safety_checker=None,  # Disable safety checker for medical images
#         torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32
#     )
#     checkpoint = load_file(f'{model_path}/model.safetensors')
#     pipe.unet.load_state_dict(checkpoint)
#     pipe = pipe.to(device)
    
#     # Enable memory optimization if on GPU
#     if "cuda" in str(device):
#         pipe.enable_attention_slicing()
    
#     return pipe

def synthesize_worker(rank, device, samples_split, save_dir, output_csv_dir, records_list, model_args, path, dataset_type):
    print(f"Rank {rank} using {device}, num samples: {len(samples_split)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

    torch.cuda.set_device(device)

    local_seed = int(model_args["base_seed"]) + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    # a generator the diffusers pipeline will use
    generator = torch.Generator(device=device).manual_seed(local_seed)


    pipe = load_finetuned_pipeline(model_args["model_path"], model_args["model_version"], device)
    batch_size = model_args.get("batch_size", 4)
    print(f"Rank {rank} using device {device} with batch size {batch_size}")
    for batch_start in tqdm(range(0, len(samples_split), batch_size), desc=f"GPU {rank}"):
        batch_samples = samples_split[batch_start:batch_start + batch_size]

        prompts = [sample[0] for sample in batch_samples]
        label_dicts = [sample[1] for sample in batch_samples]
        global_idxs = [sample[2] for sample in batch_samples]
        
        images = pipe(
            prompts, 
            num_inference_steps=50, 
            guidance_scale=7.5,
            height=512,  
            width=512
        ).images

        for img, labels, idx in zip(images, label_dicts, global_idxs):
            img_filename = f"synthetic_{idx:07d}.png"
            img_path = os.path.join(save_dir, img_filename)
            img.save(img_path)

            if dataset_type == "chexpert":
                record = {"Path": os.path.join(path, img_filename)}
            elif dataset_type == "diabetic_retinopathy":
                record = {"image": img_filename}
            else:  # isic2019
                record = {"image": img_filename}
                
            record.update(labels)
            records_list.append(record)

def prepare_chexpert_data(combination_csv):
    """Prepare data for CheXpert dataset"""
    combination_counts = pd.read_csv(combination_csv)
    label_columns = [col for col in combination_counts.columns if col.lower() != 'count']
    
    if len(label_columns) == 0:
        print(f"Error: No label columns found in combination CSV.")
        sys.exit(1)
    
    all_samples = []
    idx = 0
    for _, row in combination_counts.iterrows():
        labels = {label: int(row[label]) for label in label_columns}
        caption = create_chestxray_caption(row, label_columns)
        count = int(row['count'])
        for _ in range(count):
            all_samples.append((caption, labels, idx))
            idx += 1
    
    return all_samples



def prepare_isic2019_data(combination_csv):
    "Prepare data for ISIC 2019 dataset"
    combination_counts = pd.read_csv(combination_csv)
    label_columns = [col for col in combination_counts.columns if col.lower() != 'count']
    
    if len(label_columns) == 0:
        print(f"Error: No label columns found in combination CSV.")
        sys.exit(1)
        
    all_samples = []
    idx = 0
    for _, row in combination_counts.iterrows():
        labels = {label: int(row[label]) for label in label_columns}
        
        caption = create_isic2019_caption(row)
        count = int(row['count'])
        for _ in range(count):
            all_samples.append((caption, labels, idx))
            idx += 1

    return all_samples

def prepare_retinopathy_data(combination_csv):
    """Prepare data for Diabetic Retinopathy dataset"""
    retinopathy_data = pd.read_csv(combination_csv)
    
    # Check if the CSV has the required columns
    required_cols = ['level']
    for col in required_cols:
        if col not in retinopathy_data.columns:
            print(f"Error: Required column '{col}' not found in the CSV file.")
            print("Current columns:", retinopathy_data.columns)
            sys.exit(1)
    
    # Add eye_position if it doesn't exist
    if 'eye_position' not in retinopathy_data.columns:
        # Try to infer from image name if it exists
        if 'image' in retinopathy_data.columns:
            retinopathy_data['eye_position'] = retinopathy_data['image'].apply(
                lambda x: 'left' if '_left' in str(x) else 'right' if '_right' in str(x) else None
            )
        else:
            # Alternate between left and right if no image column
            retinopathy_data['eye_position'] = ['left' if i % 2 == 0 else 'right' for i in range(len(retinopathy_data))]
    
    # If count column exists, use it for replication, otherwise assume 1
    if 'count' in retinopathy_data.columns:
        has_count = True
    else:
        has_count = False
        retinopathy_data['count'] = 1
    
    all_samples = []
    idx = 0
    for _, row in retinopathy_data.iterrows():
        # Create label dictionary
        labels = {'level': int(row['level'])}
        if 'eye_position' in row:
            labels['eye_position'] = row['eye_position']
        
        caption = create_retinopathy_caption(row)
        count = int(row['count']) if has_count else 1
        
        for _ in range(count):
            all_samples.append((caption, labels, idx))
            idx += 1
    
    return all_samples

def main():
    #     combination_csv='/home/mila/p/parham.saremi/dbpp/datasets/isic2019/isic2019_combdist_train.csv',
    # finetuned_model_path='/network/scratch/p/parham.saremi/dbpp-models/isic-checkpiont-200-0',

    parser = argparse.ArgumentParser(description="Synthesize images using a fine-tuned Stable Diffusion pipeline.")
    parser.add_argument("--dataset_type", type=str, default="chexpert", choices=["chexpert", "diabetic_retinopathy", "isic2019"],
                        help="Type of dataset to synthesize images for.")
    parser.add_argument("--combination_csv", type=str, default='/home/mila/z/zahra.tehraninasab/workshop/dbpp/datasets/chexpert/chexpert_combdist_train.csv',
                        help="Path to the combination CSV file.")
    parser.add_argument("--finetuned_model_path", type=str, default='/network/scratch/z/zahra.tehraninasab/dbpp-models/chexpert-checkpoint-64-0',
                        help="Path to the directory containing the fine-tuned model.")
    parser.add_argument("--lora_weights_path", type=str, default='/network/scratch/z/zahra.tehraninasab/dbpp-models/checkpoint_47/pytorch_lora_weights.bin',)
    parser.add_argument("--output_csv_path", type=str, default="synthetic_metadata.csv",
                        help="Path to save the output metadata CSV file.")
    parser.add_argument("--save_dir", type=str, default="synthesized_images_chexpert",
                        help="Directory to save the synthesized images.")
    parser.add_argument("--path", type=str, default="synthesized_images_chexpert",
                        help="Path prefix for the synthesized images in the metadata.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for image synthesis.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for the samples to synthesize.")
    parser.add_argument("--stop_index", type=int, default=-1,
                        help="Stop index for the samples to synthesize (-1 for all).")
    parser.add_argument("--seed", type=int, default=50,
                        help="Random seed for reproducibility.")
    parser.add_argument("--model_version", type=str, default="unet",
                        help="Version of the model to use (e.g., 'unet', 'lora').")
    

    args = parser.parse_args()

    # Validate dataset type
    if args.dataset_type not in ["chexpert", "diabetic_retinopathy", "isic2019"]:
        print(f"Error: Invalid dataset type '{args.dataset_type}'. Must be 'chexpert', 'diabetic_retinopathy', or 'isic2019'.")
        sys.exit(1)

    if not os.path.isfile(args.combination_csv):
        print(f"Error: Combination file '{args.combination_csv}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.finetuned_model_path):
        print(f"Error: Finetuned model directory '{args.finetuned_model_path}' does not exist.")
        sys.exit(1)

    if not args.combination_csv.lower().endswith('.csv'):
        print("Error: Combination file must be a CSV.")
        sys.exit(1)

    output_csv_path = get_valid_output_csv_path(args.output_csv_path)

    # Make save directory if needed
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare data based on dataset type
    if args.dataset_type == "chexpert":
        all_samples = prepare_chexpert_data(args.combination_csv)
    elif args.dataset_type == "diabetic_retinopathy":
        all_samples = prepare_retinopathy_data(args.combination_csv)
    else:  # isic2019
        all_samples = prepare_isic2019_data(args.combination_csv)

    total_samples = len(all_samples)
    print(f"Total number of samples in the dataset: {total_samples}")

    start_index = max(0, args.start_index)
    
    stop_index = total_samples if (args.stop_index == -1 or args.stop_index > total_samples) else args.stop_index
    print(f"Start index: {start_index}, Stop index: {stop_index}")
    all_samples = all_samples[start_index:stop_index]

    print(f"Total number of samples to synthesize: {len(all_samples)} from {total_samples}. Starting from {start_index} to {stop_index}.")

    # Multi-GPU setup
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print(f"Multiple GPUs detected: {n_gpus} GPUs")

        splits = [all_samples[i::n_gpus] for i in range(n_gpus)]

        manager = mp.Manager()
        records_list = manager.list()

        model_args = {
            "model_path": args.finetuned_model_path,
            "model_version": args.model_version,
            "batch_size": args.batch_size,
            "base_seed": args.seed,
        }

        processes = []
        for rank in range(n_gpus):
            p = mp.Process(target=synthesize_worker, args=(
                rank, 
                f'cuda:{rank}', 
                splits[rank], 
                args.save_dir, 
                os.path.dirname(output_csv_path), 
                records_list, 
                model_args, 
                args.path,
                args.dataset_type
            ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        df_records = pd.DataFrame(list(records_list))
        df_records.to_csv(output_csv_path, index=False)

    else:
        print("Single GPU or CPU mode.")
        records_list = []

        model_args = {
            "model_path": args.finetuned_model_path,
            "batch_size": args.batch_size,
            "base_seed": args.seed,
        }

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        synthesize_worker(
            rank=0,
            device=device,
            samples_split=all_samples,
            save_dir=args.save_dir,
            output_csv_dir=os.path.dirname(output_csv_path),
            records_list=records_list,
            model_args=model_args,
            path=args.path,
            dataset_type=args.dataset_type
        )

        df_records = pd.DataFrame(records_list)
        df_records.to_csv(output_csv_path, index=False)

    print(f"\n✅ Synthesis complete! Images saved to '{args.save_dir}', metadata CSV saved to '{output_csv_path}'.")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()