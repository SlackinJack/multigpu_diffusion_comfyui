import gc
import os
import torch


from compel import Compel, ReturnedEmbeddingsType
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline


from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


class CheckpointSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "checkpoint": CHECKPOINT_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class ModelSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": MODEL_LIST, "config": MODEL_CONFIG_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class UnsafeModelSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": UNSAFE_MODEL_LIST, "config": MODEL_CONFIG_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class LoraSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "lora": LORA_LIST, "weight": LORA_WEIGHT } }
    RETURN_TYPES, FUNCTION, CATEGORY = LORA, "get", ROOT_CATEGORY_GENERAL
    def get(self, lora, weight): return ({ f"{get_models_dir()}/loras/{lora}": weight },)


class MultiLoraJoiner:
    @classmethod
    def INPUT_TYPES(s): return { "optional": { "lora_1": LORA, "lora_2": LORA, "lora_3": LORA, "lora_4": LORA,
                                               "lora_5": LORA, "lora_6": LORA, "lora_7": LORA, "lora_8": LORA,
                                               "lora_9": LORA, "lora_10": LORA, "lora_11": LORA, "lora_12": LORA,
                                               "lora_13": LORA, "lora_14": LORA, "lora_15": LORA, "lora_16": LORA,
                                               "lora_17": LORA, "lora_18": LORA, "lora_19": LORA, "lora_20": LORA,
                                               "lora_21": LORA, "lora_22": LORA, "lora_23": LORA, "lora_24": LORA,
                                               "lora_25": LORA, "lora_26": LORA, "lora_27": LORA, "lora_28": LORA,
                                               "lora_29": LORA, "lora_30": LORA, "lora_31": LORA, "lora_32": LORA } }
    RETURN_TYPES, FUNCTION, CATEGORY = LORA, "join", ROOT_CATEGORY_GENERAL
    def join(self, **kwargs):
        out = {}
        for k, v in kwargs.items():
            for adapter, scale in v.items():
                out[adapter] = scale
        return (out,)


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
            "compile_options":          ("STRING", { "default": "", "multiline": False }),
            "compile_fullgraph_off":    BOOLEAN_DEFAULT_FALSE,
        }
    }

    RETURN_TYPES, FUNCTION, CATEGORY = COMPILE_CONFIG, "get_config", ROOT_CATEGORY_CONFIG

    def get_config(self, compile_transformer, compile_vae, compile_encoder, compile_backend, compile_mode, compile_options, compile_fullgraph_off):
        out = {}
        if compile_transformer is True:     out["compile_transformer"] = True
        if compile_vae is True:             out["compile_vae"] = True
        if compile_encoder is True:         out["compile_encoder"] = True
        if compile_backend != "default":    out["compile_backend"] = compile_backend
        if compile_mode != "default":       out["compile_mode"] = compile_mode
        if len(compile_options) > 0:        out["compile_options"] = compile_options
        if compile_fullgraph_off is True:   out["compile_fullgraph_off"] = True
        return (out,)
