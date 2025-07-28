#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import sys
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxMVKontextPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


sys.path.append(os.getcwd())
from mv_kontext.mv_datasets.navi_datasets import loader as navi_loader
from mv_kontext.mv_datasets.navi_test_datasets import loader as navi_test_loader
from mv_kontext.utils.data_handlers import convert_tensor_to_img_grid, img_tensor_to_pil
from mv_kontext.pipeline.render_func import render_pipe

if is_wandb_available():
    import wandb
    os.environ["WANDB_API_KEY"] = "6f0e0b7ac10956a085dfcc856fff68135ed68b87"
    os.environ["WANDB_MODE"] = "offline"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.35.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(test_dataloader, flux_transformer, args, accelerator, weight_dtype, global_step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        flux_transformer = accelerator.unwrap_model(flux_transformer)
        pipeline = FluxMVKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=flux_transformer,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        initial_channels = transformer.config.in_channels
        pipeline = FluxMVKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)
        assert pipeline.transformer.config.in_channels == initial_channels * 2, (
            f"{pipeline.transformer.config.in_channels=}"
        )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)



    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)


    for step, batch in enumerate(test_dataloader):
    
        ## init input
        sample_render_extrinsic = batch["sample_render_extrinsics"]
        sample_render_intrinsic = batch["sample_render_intrinsics"]
        sample_render_gt_images = batch["sample_render_gt_images"]
        sample_render_gt_masks = batch["sample_render_gt_masks"]
        sample_render_idxs = batch["sample_render_idxs"]
        sample_render_ori_gt_images = batch["sample_render_ori_gt_images"]
        sample_render_instructions = batch["sample_render_instructions"]

        sample_recon_intrinsics = batch["sample_recon_intrinsics"]
        sample_recon_extrinsics = batch["sample_recon_extrinsics"]
        sample_recon_images = batch["sample_recon_images"]
        sample_recon_masks = batch["sample_recon_masks"]
        sample_recon_idxs = batch["sample_recon_idxs"]
        sample_recon_depths = batch["sample_recon_depths"]
        sample_recon_depth_confs = batch["sample_recon_depth_confs"]
        sample_recon_world_points = batch["sample_recon_world_points"]
        sample_recon_world_points_confs = batch["sample_recon_world_points_confs"]
        
        sample_multi_view_paths = batch["sample_multi_view_paths"]

        if isinstance(args.conf_threshold, list):
            conf_threshold = random.choice(args.conf_threshold)
        else:
            conf_threshold = args.conf_threshold

        ## render
        novel_images, novel_depths, recon_depths = render_pipe(
            sample_recon_images, 
            sample_recon_extrinsics, 
            sample_recon_intrinsics, 
            sample_recon_masks, 
            sample_recon_depths, 
            sample_recon_depth_confs, 
            sample_recon_world_points, 
            sample_recon_world_points_confs,
            sample_render_extrinsic,
            sample_render_intrinsic,
            sample_render_ori_gt_images,
            args.use_point_map, 
            args.max_points, 
            conf_threshold, 
            args.conf_threshold_value, 
            args.apply_mask, 
            args.radius, 
            args.points_per_pixel, 
            args.bin_size, 
            accelerator.device, 
            viz_depth=False
        )
        novel_cond_images =((novel_images / 255.0).permute(0, 3, 1, 2)) # [B, 3, H, W]

        height, width = novel_cond_images.shape[-2], novel_cond_images.shape[-1]

     
        with autocast_ctx:
            image = pipeline(
                prompt=sample_render_instructions[0],
                image=novel_cond_images,
                num_inference_steps=28,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=512,
                height=height,
                width=width,
            ).images[0]

        render_cond_img_grid = convert_tensor_to_img_grid(novel_cond_images, rescale=False, num_rows=2)

        sample_render_ori_gt_images_grid = convert_tensor_to_img_grid(sample_render_ori_gt_images, rescale=True, num_rows=2)
        image_logs.append(
            {"image": image, "instruction": sample_render_instructions[0], "render_cond_image": render_cond_img_grid, "sample_render_ori_gt_images_grid": sample_render_ori_gt_images_grid}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                image = log["image"]
                instruction = log["instruction"]
                render_cond_image = log["render_cond_image"]
                tracker.writer.add_images(instruction, image, global_step, dataformats="NHWC")
                tracker.writer.add_images(instruction, render_cond_image, global_step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []
            for log_idx, log in enumerate(image_logs):
                image = log["image"]
                instruction = log["instruction"]
                render_cond_image = log["render_cond_image"]
                sample_render_ori_gt_images_grid = log["sample_render_ori_gt_images_grid"]

                save_dir = os.path.join(args.output_dir, f"output_g{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f"output_idx_{log_idx}.png"))
                render_cond_image.save(os.path.join(save_dir, f"render_cond_image_idx_{log_idx}.png"))
                sample_render_ori_gt_images_grid.save(os.path.join(save_dir, f"sample_render_ori_gt_images_grid_idx_{log_idx}.png"))
                with open(os.path.join(save_dir, f"instruction_idx_{log_idx}.txt"), "w") as f:
                    f.write(instruction)

                formatted_images.append(wandb.Image(image, caption=f"g{global_step}_idx_{log_idx}_{instruction}"))
                formatted_images.append(wandb.Image(render_cond_image, caption=f"g{global_step}_idx_{log_idx}_Conditioning"))
                formatted_images.append(wandb.Image(sample_render_ori_gt_images_grid, caption=f"g{global_step}_idx_{log_idx}_GT"))

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()
        return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_instruction = log["instruction"]
            validation_render_cond_image = log["render_cond_image"]
            validation_sample_render_ori_gt_images_grid = log["sample_render_ori_gt_images_grid"]
            img_str += f"prompt: {validation_instruction}\n"
            images = [validation_render_cond_image, validation_sample_render_ori_gt_images_grid] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
    # control-lora-{repo_id}

    These are Control LoRA weights trained on {base_model} with new type of conditioning.
    {img_str}

    ## License

    Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
    """

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "control-lora",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Control LoRA training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="control-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--use_lora_bias", action="store_true", help="If training the bias of lora_B layers.")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument("--train_norm_layers", action="store_true", help="Whether to train the norm scales.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=None,
        help="The directory to save the wandb logs to.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--log_dataset_samples", action="store_true", help="Whether to log somple dataset samples.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_train_control_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the guidance scale used for transformer.",
    )

    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoders to CPU when they are not used.",
    )

    ## MV-Custom
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to the meta data file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--min_recon_num",
        type=int,
        default=1,
        help="Minimum number of reconstructed images to be used for training.",
    )
    parser.add_argument(
        "--max_recon_num",
        type=int,
        default=8,
        help="Maximum number of reconstructed images to be used for training.",
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Whether to use the point map for training.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Maximum number of points to be used for training.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=20,
        nargs="+",
        help="Confidence threshold for the points.",
    )
    parser.add_argument(
        "--conf_threshold_value",
        type=float,
        default=1.0,
        help="Confidence threshold value for the points.",
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        help="Whether to apply the mask for the points.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.003,
        help="Radius of the points.",
    )
    parser.add_argument(
        "--points_per_pixel",
        type=int,
        default=10,
        help="Points per pixel for the renderer.",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=None,
        help="Bin size for the renderer.",
    )

    parser.add_argument(
        "--test_meta_path",
        type=str,
        default=None,
        help="Path to the test meta file.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Path to the test data directory.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Validation steps.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.meta_path is None and args.data_dir is None:
        raise ValueError("Specify either `--meta_path` or `--data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the Flux transformer."
        )

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError("`gaussian` LoRA init scheme isn't supported when `use_lora_bias` is True.")


    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # import ipdb; ipdb.set_trace()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models loaded successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    vae.to(dtype=torch.float32)  # keep the VAE in float32.
    flux_transformer.to(dtype=weight_dtype, device=accelerator.device)

    if args.train_norm_layers:
        for name, param in flux_transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    if args.lora_layers is not None:
        if args.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
            # add the input layer to the mix.
            if "x_embedder" not in target_modules:
                target_modules.append("x_embedder")
        elif args.lora_layers == "all-linear":
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
    else:
        target_modules = [
            "x_embedder",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    )
    flux_transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        model = unwrap_model(model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        if args.train_norm_layers:
                            transformer_norm_layers_to_save = {
                                f"transformer.{name}": param
                                for name, param in model.named_parameters()
                                if any(k in name for k in NORM_LAYER_PREFIXES)
                            }
                            transformer_lora_layers_to_save = {
                                **transformer_lora_layers_to_save,
                                **transformer_norm_layers_to_save,
                            }
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                FluxMVKontextPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(flux_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                ).to(accelerator.device, weight_dtype)

                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = FluxMVKontextPipeline.lora_state_dict(input_dir)
            transformer_lora_state_dict = {
                f"{k.replace('transformer.', '')}": v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and "lora" in k
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_lora_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            if args.train_norm_layers:
                transformer_norm_state_dict = {
                    k: v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
                }
                transformer_._transformer_norm_layers = FluxMVKontextPipeline._load_norm_into_transformer(
                    transformer_norm_state_dict,
                    transformer=transformer_,
                    discard_original_layers=False,
                )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [flux_transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare dataset and dataloader.
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    train_dataloader = navi_loader(train_batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, meta_path=args.meta_path, data_dir=args.data_dir, min_recon_num=args.min_recon_num, max_recon_num=args.max_recon_num, shuffle=False, rank=rank, world_size=world_size)

    test_dataloader = navi_test_loader(train_batch_size=1, num_workers=0, meta_path=args.test_meta_path, data_dir=args.test_data_dir, shuffle=False)
 

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # Initialize trackers with project name and specify wandb directory if needed
        if args.report_to == "wandb" and hasattr(args, "wandb_dir"):
            accelerator.init_trackers(
                args.tracker_project_name, 
                config=tracker_config,
                init_kwargs={"wandb": {"dir": args.wandb_dir}} if args.wandb_dir else None
            )
        else:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader) * total_batch_size}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    text_encoding_pipeline = FluxMVKontextPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if accelerator.is_main_process and args.report_to == "wandb" and args.log_dataset_samples:
        logger.info("Logging some dataset samples.")
        formatted_recon_images = []
        formatted_recon_masks = []
        formatted_recon_depth_color = []

        formatted_render_ori_gt_images = []
        formatted_render_novel_depth_color = []
        formatted_render_novel_images = []

        for i, batch in enumerate(train_dataloader):
            if len(formatted_render_ori_gt_images) >= 1:
                break

            sample_render_extrinsic = batch["sample_render_extrinsics"]
            sample_render_intrinsic = batch["sample_render_intrinsics"]
            sample_render_gt_images = batch["sample_render_gt_images"]
            sample_render_gt_masks = batch["sample_render_gt_masks"]
            sample_render_idxs = batch["sample_render_idxs"]
            sample_render_ori_gt_images = batch["sample_render_ori_gt_images"]
            sample_render_instructions = batch["sample_render_instructions"]

            sample_recon_intrinsics = batch["sample_recon_intrinsics"]
            sample_recon_extrinsics = batch["sample_recon_extrinsics"]
            sample_recon_images = batch["sample_recon_images"]
            sample_recon_masks = batch["sample_recon_masks"]
            sample_recon_idxs = batch["sample_recon_idxs"]
            sample_recon_depths = batch["sample_recon_depths"]
            sample_recon_depth_confs = batch["sample_recon_depth_confs"]
            sample_recon_world_points = batch["sample_recon_world_points"]
            sample_recon_world_points_confs = batch["sample_recon_world_points_confs"]

            multi_view_paths = batch["sample_multi_view_paths"]
            
            if isinstance(args.conf_threshold, list):
                conf_threshold = random.choice(args.conf_threshold)
            else:
                conf_threshold = args.conf_threshold

            novel_images, novel_depths, recon_depths, novel_depths_color, novel_depths_normalized, recon_depths_color, recon_depths_normalized = render_pipe(
                sample_recon_images, 
                sample_recon_extrinsics, 
                sample_recon_intrinsics, 
                sample_recon_masks, 
                sample_recon_depths, 
                sample_recon_depth_confs, 
                sample_recon_world_points, 
                sample_recon_world_points_confs,
                sample_render_extrinsic,
                sample_render_intrinsic,
                sample_render_ori_gt_images,
                args.use_point_map, 
                args.max_points, 
                conf_threshold, 
                args.conf_threshold_value, 
                args.apply_mask, 
                args.radius, 
                args.points_per_pixel, 
                args.bin_size, 
                accelerator.device, 
                viz_depth=True
                )
            recon_depths_colors = torch.from_numpy((recon_depths_color / 255.0)).permute(0, 1, 4, 2, 3) # [B, S, 3, H, W]
            novel_depths_colors = torch.from_numpy((novel_depths_color / 255.0)).permute(0, 3, 1, 2) # [B, 3, H, W]
            novel_images = (novel_images / 255.0).permute(0, 3, 1, 2) # [B, 3, H, W]

            for recon_img, recon_mask, recon_depth_color, render_ori_gt_img, render_novel_img, render_novel_depth_color, render_instruction, multi_view_path in zip(sample_recon_images, sample_recon_masks, recon_depths_colors, sample_render_ori_gt_images, novel_images, novel_depths_colors, sample_render_instructions, multi_view_paths):
                recon_img_grid = convert_tensor_to_img_grid(recon_img, rescale=False, num_rows=2)
                recon_mask_grid = convert_tensor_to_img_grid(recon_mask, rescale=False, num_rows=2)
                recon_depth_color_grid = convert_tensor_to_img_grid(recon_depth_color, rescale=False, num_rows=2)

                render_novel_img_grid = convert_tensor_to_img_grid(render_novel_img, rescale=False, num_rows=2)
                render_novel_depth_color_grid = convert_tensor_to_img_grid(render_novel_depth_color, rescale=False, num_rows=2)
                render_ori_gt_img_grid = convert_tensor_to_img_grid(render_ori_gt_img, rescale=True, num_rows=2)

                obj_id, camera_id = multi_view_path.split("/")[-1], multi_view_path.split("/")[-2]

                formatted_recon_images.append(recon_img_grid)
                formatted_recon_masks.append(recon_mask_grid)
                formatted_recon_depth_color.append(recon_depth_color_grid)
                formatted_render_ori_gt_images.append(render_ori_gt_img_grid)
                formatted_render_novel_images.append(render_novel_img_grid)
                formatted_render_novel_depth_color.append(render_novel_depth_color_grid)
                break
        

        logged_artifacts = []
        for recon_img, recon_mask, render_ori_gt_img, recon_depth_color, render_novel_img, render_novel_depth_color in zip(formatted_recon_images, formatted_recon_masks, formatted_render_ori_gt_images, formatted_recon_depth_color, formatted_render_novel_images, formatted_render_novel_depth_color):
            logged_artifacts.append(wandb.Image(recon_img, caption=f"Recons Images {obj_id} {camera_id}"))
            logged_artifacts.append(wandb.Image(recon_mask, caption=f"Recons Masks {obj_id} {camera_id}"))
            logged_artifacts.append(wandb.Image(recon_depth_color, caption=f"Recons Depth Color {obj_id} {camera_id}"))
            logged_artifacts.append(wandb.Image(render_ori_gt_img, caption=f"GT Renders {obj_id} {camera_id}"))
            logged_artifacts.append(wandb.Image(render_novel_img, caption=f"Novel Renders {obj_id} {camera_id}"))
            logged_artifacts.append(wandb.Image(render_novel_depth_color, caption=f"Novel Depth Color {obj_id} {camera_id}"))
            
        wandb_tracker = [tracker for tracker in accelerator.trackers if tracker.name == "wandb"]
        wandb_tracker[0].log({"dataset_samples": logged_artifacts})

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                ## init input
                sample_render_extrinsic = batch["sample_render_extrinsics"]
                sample_render_intrinsic = batch["sample_render_intrinsics"]
                sample_render_gt_images = batch["sample_render_gt_images"]
                sample_render_gt_masks = batch["sample_render_gt_masks"]
                sample_render_idxs = batch["sample_render_idxs"]
                sample_render_ori_gt_images = batch["sample_render_ori_gt_images"]
                sample_render_instructions = batch["sample_render_instructions"]

                sample_recon_intrinsics = batch["sample_recon_intrinsics"]
                sample_recon_extrinsics = batch["sample_recon_extrinsics"]
                sample_recon_images = batch["sample_recon_images"]
                sample_recon_masks = batch["sample_recon_masks"]
                sample_recon_idxs = batch["sample_recon_idxs"]
                sample_recon_depths = batch["sample_recon_depths"]
                sample_recon_depth_confs = batch["sample_recon_depth_confs"]
                sample_recon_world_points = batch["sample_recon_world_points"]
                sample_recon_world_points_confs = batch["sample_recon_world_points_confs"]
                
                sample_multi_view_paths = batch["sample_multi_view_paths"]

                if isinstance(args.conf_threshold, list):
                    conf_threshold = random.choice(args.conf_threshold)
                else:
                    conf_threshold = args.conf_threshold
                
                ## render
                novel_images, novel_depths, recon_depths = render_pipe(
                    sample_recon_images, 
                    sample_recon_extrinsics, 
                    sample_recon_intrinsics, 
                    sample_recon_masks, 
                    sample_recon_depths, 
                    sample_recon_depth_confs, 
                    sample_recon_world_points, 
                    sample_recon_world_points_confs,
                    sample_render_extrinsic,
                    sample_render_intrinsic,
                    sample_render_ori_gt_images,
                    args.use_point_map, 
                    args.max_points, 
                    conf_threshold, 
                    args.conf_threshold_value, 
                    args.apply_mask, 
                    args.radius, 
                    args.points_per_pixel, 
                    args.bin_size, 
                    accelerator.device, 
                    viz_depth=False
                )
                novel_images =((novel_images / 255.0).permute(0, 3, 1, 2) - 0.5) * 2.0 # [B, 3, H, W]

                # Convert images to latent space
                # vae encode
                pixel_latents = encode_images(sample_render_ori_gt_images, vae.to(accelerator.device), weight_dtype)
                context_latents = encode_images(novel_images, vae.to(accelerator.device), weight_dtype)

                if args.offload:
                    # offload vae to CPU.
                    vae.cpu()

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                

                # pack the latents.
                packed_noisy_model_input = FluxMVKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=noisy_model_input.shape[1],
                    height=noisy_model_input.shape[2],
                    width=noisy_model_input.shape[3],
                )
                packed_context_latents = FluxMVKontextPipeline._pack_latents(
                    context_latents,
                    batch_size=bsz,
                    num_channels_latents=context_latents.shape[1],
                    height=context_latents.shape[2],
                    width=context_latents.shape[3],
                )

                # Concatenate across seq_len.
                packed_concatenated_noisy_model_input = torch.cat([packed_noisy_model_input, packed_context_latents], dim=1)

                # latent image ids for RoPE.
                latent_ids = FluxMVKontextPipeline._prepare_latent_image_ids(
                    bsz,
                    noisy_model_input.shape[2] // 2,
                    noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                image_ids = FluxMVKontextPipeline._prepare_latent_image_ids(
                    bsz,
                    context_latents.shape[2] // 2,
                    context_latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                image_ids[..., 0] = 1
                latent_image_ids = torch.cat([latent_ids, image_ids], dim=0)

                # handle guidance
                if unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # text encoding.
                captions = sample_render_instructions
                text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None
                    )
                # this could be optimized by not having to do any text encoding and just
                # doing zeros on specified shapes for `prompt_embeds` and `pooled_prompt_embeds`
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()
                if args.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                # Predict.
                model_pred = flux_transformer(
                    hidden_states=packed_concatenated_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
                model_pred = FluxMVKontextPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    # Handle symbolic links safely
                                    try:
                                        if os.path.islink(removing_checkpoint):
                                            os.unlink(removing_checkpoint)
                                        else:
                                            shutil.rmtree(removing_checkpoint)
                                    except Exception as e:
                                        logger.warning(f"Failed to remove checkpoint {removing_checkpoint}: {e}")

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save checkpoint at step {global_step}: {e}")
                            # 尝试释放一些空间，删除最旧的检查点
                            try:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if checkpoints:
                                    oldest_checkpoint = os.path.join(args.output_dir, checkpoints[0])
                                    if os.path.islink(oldest_checkpoint):
                                        os.unlink(oldest_checkpoint)
                                    else:
                                        shutil.rmtree(oldest_checkpoint)
                                    logger.info(f"Removed oldest checkpoint {oldest_checkpoint} to free space")
                            except Exception as inner_e:
                                logger.warning(f"Failed to remove old checkpoint: {inner_e}")

                    if global_step % args.validation_steps == 0:
                        if test_dataloader is not None:
                            try:
                                image_logs = log_validation(
                                    test_dataloader=test_dataloader,
                                    flux_transformer=flux_transformer,
                                    args=args,
                                    accelerator=accelerator,
                                        weight_dtype=weight_dtype,
                                        global_step=global_step,
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to run validation: {e}")
                                image_logs = None

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_transformer = unwrap_model(flux_transformer)
        if args.upcast_before_saving:
            flux_transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(flux_transformer)
        if args.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in flux_transformer.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}
        try:
            FluxMVKontextPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
            )
            logger.info(f"Successfully saved final model weights to {args.output_dir}")
        except Exception as e:
            logger.error(f"Failed to save final model weights: {e}")
            # 尝试清理一些空间并重试
            try:
                # 删除所有检查点，保留最新的一个
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                if len(checkpoints) > 1:
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    latest_checkpoint = checkpoints[-1]
                    for checkpoint in checkpoints[:-1]:
                        checkpoint_path = os.path.join(args.output_dir, checkpoint)
                        if os.path.islink(checkpoint_path):
                            os.unlink(checkpoint_path)
                        else:
                            shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed all checkpoints except {latest_checkpoint} to free space")
                    
                    # 重试保存
                    FluxMVKontextPipeline.save_lora_weights(
                        save_directory=args.output_dir,
                        transformer_lora_layers=transformer_lora_layers,
                    )
                    logger.info(f"Successfully saved final model weights after cleanup")
            except Exception as retry_e:
                logger.error(f"Failed to save model even after cleanup: {retry_e}")

        del flux_transformer
        del text_encoding_pipeline
        del vae
        free_memory()

        # Run a final round of validation.
        image_logs = None
        if test_dataloader is not None:
            try:
                image_logs = log_validation(
                    test_dataloader=test_dataloader,
                    flux_transformer=None,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    global_step=global_step,
                    is_final_validation=True,
                )
            except Exception as e:
                logger.warning(f"Failed to run final validation: {e}")
                image_logs = None

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "*.pt", "*.bin"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
