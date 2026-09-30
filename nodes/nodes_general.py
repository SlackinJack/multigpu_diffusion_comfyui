import gc
import numpy as np
import os
import random
import torch


from PIL import Image
# from compel import Compel, ReturnedEmbeddingsType
# from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline


import folder_paths


from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


"""
class EncodePromptWithCompel:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "checkpoint":   MODEL,
            "model_type":   COMPEL_MODEL_LIST,
            "prompt":       PROMPT,
        }
    }

    RETURN_TYPES, FUNCTION, CATEGORY = CONDITIONING, "encode", ROOT_CATEGORY_TOOLS

    def encode(self, checkpoint, model_type, prompt):
        torch_dtype = torch.float32

        if model_type in ["sd1", "sd2"]:
            pipeline_class = StableDiffusionPipeline
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        else:
            pipeline_class = StableDiffusionXLPipeline
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=os.path.join(get_models_dir(), checkpoint["checkpoint"]),
            use_safetensors=True,
            local_files_only=True,
        ).to("cpu")

        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=embeddings_type,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        embeds, pooled_embeds = compel([prompt])
        del compel
        del pipe
        gc.collect()
        return ([[embeds, { "pooled_output": pooled_embeds }]],)
"""


class TorchConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "torch_cache_limit":            ("INT", { "default": 16, "min": 0}),
            "torch_accumlated_cache_limit": ("INT", { "default": 128, "min": 0}),
            "torch_capture_scalar":         BOOLEAN_DEFAULT_FALSE,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = TORCH_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


class GroupOffloadConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "optional": {
            "transformer":  OFFLOAD_CONFIG,
            "encoder":      OFFLOAD_CONFIG,
            "vae":          OFFLOAD_CONFIG,
            "misc":         OFFLOAD_CONFIG,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = GROUP_OFFLOAD_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


class OffloadConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "offload_device":       ("STRING", {"default": "cpu", "multiline": False}),
            "offload_type":         (["leaf_level", "block_level"], {"default": "leaf_level"}),
            "num_blocks_per_group": ("INT", {"default": 2, "min": 1}),
            "use_stream":           BOOLEAN_DEFAULT_FALSE,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = OFFLOAD_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


class CompileConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "compile_transformer":      BOOLEAN_DEFAULT_FALSE,
            "compile_vae":              BOOLEAN_DEFAULT_FALSE,
            "compile_encoder":          BOOLEAN_DEFAULT_FALSE,
            "compile_backend":          (["default", "inductor", "eager"], { "default": "default" }),
            "compile_mode":             (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], { "default": "default" }),
            "compile_options":          ("STRING", { "default": "{}", "multiline": True }),
            "dynamic":                  BOOLEAN_DEFAULT_TRUE,
            "fullgraph":                BOOLEAN_DEFAULT_TRUE,
        }
    }

    RETURN_TYPES, FUNCTION, CATEGORY = COMPILE_CONFIG, "get_config", ROOT_CATEGORY_CONFIG

    def get_config(self, compile_transformer, compile_vae, compile_encoder, compile_backend, compile_mode, compile_options, dynamic, fullgraph):
        out = {}
        if compile_transformer is True:     out["compile_transformer"] = True
        if compile_vae is True:             out["compile_vae"] = True
        if compile_encoder is True:         out["compile_encoder"] = True
        if compile_backend != "default":    out["compile_backend"] = compile_backend
        if compile_mode != "default":       out["compile_mode"] = compile_mode
        if len(compile_options) > 0:        out["compile_options"] = compile_options
        if dynamic is True:                 out["dynamic"] = True
        if fullgraph is True:               out["fullgraph"] = True
        return (out,)


class AttentionBackendConfig:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "backend": ATTN_BACKEND_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = ATTN_BACKEND_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


class EnvironmentVariable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", { "default": "key", "multiline": False }),
                "value": ("STRING", { "default": "value", "multiline": False }),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = ENV_VARS_CONFIG, "get", ROOT_CATEGORY_CONFIG
    def get(self, key, value):
        return ({key:value},)


class EnvironmentVariableJoiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "env1": ENV_VARS_CONFIG, "env2": ENV_VARS_CONFIG, "env3": ENV_VARS_CONFIG, "env4": ENV_VARS_CONFIG,
                "env5": ENV_VARS_CONFIG, "env6": ENV_VARS_CONFIG, "env7": ENV_VARS_CONFIG, "env8": ENV_VARS_CONFIG, "env9": ENV_VARS_CONFIG,
                "env10": ENV_VARS_CONFIG, "env11": ENV_VARS_CONFIG, "env12": ENV_VARS_CONFIG, "env13": ENV_VARS_CONFIG, "env14": ENV_VARS_CONFIG,
                "env15": ENV_VARS_CONFIG, "env16": ENV_VARS_CONFIG, "env17": ENV_VARS_CONFIG, "env18": ENV_VARS_CONFIG, "env19": ENV_VARS_CONFIG,
                "env20": ENV_VARS_CONFIG, "env21": ENV_VARS_CONFIG, "env22": ENV_VARS_CONFIG, "env23": ENV_VARS_CONFIG, "env24": ENV_VARS_CONFIG,
                "env25": ENV_VARS_CONFIG, "env26": ENV_VARS_CONFIG, "env27": ENV_VARS_CONFIG, "env28": ENV_VARS_CONFIG, "env29": ENV_VARS_CONFIG,
                "env30": ENV_VARS_CONFIG, "env31": ENV_VARS_CONFIG, "env32": ENV_VARS_CONFIG,
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = ENV_VARS_CONFIG, "join", ROOT_CATEGORY_CONFIG
    def join(self, **kwargs):
        out = {}
        for k,v in kwargs.items():
            for k2,v2 in v.items():
                out[k2] = v2
        return (out,)


class CustomTimesteps:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "timesteps_csv": ("STRING", { "default": "999, 499, 0", "multiline": True }) } }
    RETURN_TYPES, FUNCTION, CATEGORY = TIMESTEPS, "get_timesteps", ROOT_CATEGORY_CONFIG
    def get_timesteps(self, timesteps_csv): 
        t = timesteps_csv.replace("\n", "").replace(" ", "").split(",")
        t = [float(x) for x in t]
        t = [int(x) for x in t]
        return (t,)


class CustomSigmas:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "sigmas_csv": ("STRING", { "default": "14.20, 7.10, 3.55", "multiline": True }) } }
    RETURN_TYPES, FUNCTION, CATEGORY = SIGMAS, "get_sigmas", ROOT_CATEGORY_CONFIG
    def get_sigmas(self, sigmas_csv):
        s = sigmas_csv.replace("\n", "").replace(" ", "").split(",")
        s = [float(x) for x in s]
        return (s,)


class CustomSaveImage:
    # A direct copy of ComfyUI's SaveImage, but with all metadata always disabled
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 0 # 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class CustomPreviewImage(CustomSaveImage):
    # A direct copy of ComfyUI's PreviewImage, but with all metadata always disabled
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 0 # 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                }