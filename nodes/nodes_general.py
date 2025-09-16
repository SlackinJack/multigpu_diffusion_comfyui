import gc
import torch


from compel import Compel, ReturnedEmbeddingsType
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline


from ..multigpu_diffusion.modules.utils import *
from .globals import *


class HostConfig:
    @classmethod
    def INPUT_TYPES(s):
        global GENERIC_CONFIGS, GENERIC_CONFIGS_COMFY, GENERIC_CONFIGS_OPTIONAL
        required = GENERIC_CONFIGS
        required.update(GENERIC_CONFIGS_COMFY)
        optional = GENERIC_CONFIGS_OPTIONAL
        return { "required": required, "optional": optional }

    RETURN_TYPES    = GENERIC_CONFIG
    FUNCTION        = "get_config"
    CATEGORY        = ROOT_CATEGORY_CONFIG

    def get_config(self, **kwargs):
        return (kwargs,)


class SchedulerSelector:
    # TODO: input sigmas
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler":                SCHEDULER_LIST,
            "use_karras_sigmas":        BOOLEAN_DEFAULT_FALSE,
            "timestep_spacing":         TIMESTEP_LIST,
            "rescale_betas_zero_snr":   BOOLEAN_DEFAULT_FALSE,
            "use_exponential_sigmas":   BOOLEAN_DEFAULT_FALSE,
            "use_beta_sigmas":          BOOLEAN_DEFAULT_FALSE,

        }
    }
    RETURN_TYPES    = SCHEDULER
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, **kwargs):
        scheduler_config = {}
        for k, v in kwargs.items():
            if str(v) != "default":
                scheduler_config[k] = v
        return (scheduler_config,)


class CheckpointSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "checkpoint": CHECKPOINT_LIST } }
    RETURN_TYPES    = CHECKPOINT
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, checkpoint): return (f"{checkpoints_dir}/{checkpoint}",)


class VAESelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "vae": VAE_LIST } }
    RETURN_TYPES    = VAE
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, vae): return (f"{vae_dir}/{vae}",)


class MotionModuleSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "motion_module": MOTION_MODULE_LIST } }
    RETURN_TYPES    = MOTION_MODULE
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, motion_module): return (f"{motion_adapter_dir}/{motion_module}",)


class MotionAdapterSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "motion_adapter": MOTION_ADAPTER_LIST } }
    RETURN_TYPES    = MOTION_ADAPTER
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, motion_adapter): return (f"{motion_adapter_dir}/{motion_adapter}",)


class MotionAdapterLoraSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "motion_adapter_lora": MOTION_MODULE_LIST } }
    RETURN_TYPES    = MOTION_ADAPTER_LORA
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, motion_adapter_lora): return (f"{motion_adapter_dir}/{motion_adapter_lora}",)


class ControlNetSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "controlnet": CONTROLNET_LIST, "scale": CONTROLNET_SCALE } }
    RETURN_TYPES    = CONTROLNET
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, controlnet, scale): return ({ f"{controlnet_dir}/{controlnet}": scale },)


class IPAdapterSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "ip_adapter": IPADAPTER_LIST, "scale": IP_ADAPTER_SCALE } }
    RETURN_TYPES    = IPADAPTER
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, ip_adapter, scale): return ({ f"{ipadapter_dir}/{ip_adapter}": scale },)


class GGUFSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "gguf": GGUF_LIST } }
    RETURN_TYPES    = MODEL_GGUF
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, gguf): return (f"{checkpoints_dir}/{gguf}",)


class LoraSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "lora": LORA_LIST, "weight": LORA_WEIGHT } }
    RETURN_TYPES    = LORA
    FUNCTION        = "get"
    CATEGORY        = ROOT_CATEGORY_GENERAL
    def get(self, lora, weight): return ({ f"{loras_dir}/{lora}": weight },)


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
    RETURN_TYPES    = LORA
    FUNCTION        = "join"
    CATEGORY        = ROOT_CATEGORY_GENERAL
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
            "checkpoint":   CHECKPOINT,
            "model_type":   COMPEL_MODEL_LIST,
            "prompt":       PROMPT,
        }
    }

    RETURN_TYPES    = CONDITIONING
    FUNCTION        = "encode"
    CATEGORY        = ROOT_CATEGORY_TOOLS

    def encode(self, checkpoint, model_type, prompt):
        torch_dtype = torch.float32

        if model_type in ["sd1", "sd2"]:
            pipeline_class = StableDiffusionPipeline
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        else:
            pipeline_class = StableDiffusionXLPipeline
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=checkpoint,
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

