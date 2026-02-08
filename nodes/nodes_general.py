import gc
import json
import os
import torch


from compel import Compel, ReturnedEmbeddingsType
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline


from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


class SchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler": (["ddim", "ddpm", "deis", "dpm_2", "dpm_2_a", "dpm_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "euler", "euler_a", "heun", "ipndm", "lms", "pndm", "tcd", "unipc"], { "default": "ddim" }),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class AdvancedSchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler":                (["ddim", "ddpm", "deis", "dpm_2", "dpm_2_a", "dpm_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "euler", "euler_a", "heun", "ipndm", "lms", "pndm", "tcd", "unipc"], { "default": "ddim" }),
            "timestep_spacing":         (["default", "leading", "linspace", "trailing"], { "default": "default" }),
            "beta_schedule":            (["default", "linear", "scaled_linear", "squaredcos_cap_v2"], { "default": "default" }),
            "beta_start":               ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_end":                 ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "rescale_betas_zero_snr":   TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,

        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs):
        scheduler_config = {}
        for k, v in kwargs.items():
            if k in ["scheduler", "beta_start", "beta_end"]:
                scheduler_config[k] = v
            elif k in ["timestep_spacing", "beta_schedule"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)


class AdvancedFMSchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "shift":                    ("FLOAT", { "default": 1.00000, "min": 0.00000, "step": 0.00001 }),
            "use_dynamic_shifting":     TRILEAN_WITH_DEFAULT,
            "base_shift":               ("FLOAT", { "default": 0.50000, "min": 0.00000, "step": 0.00001 }),
            "max_shift":                ("FLOAT", { "default": 1.15000, "min": 0.00000, "step": 0.00001 }),
            "base_image_seq_len":       ("INT", { "default": 256 }),
            "max_image_seq_len":        ("INT", { "default": 4096 }),
            "invert_sigmas":            TRILEAN_WITH_DEFAULT,
            "shift_terminal":           ("FLOAT", { "default": 0.00000, "min": 0.00000, "step": 0.00001 }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,
            "time_shift_type":          (["exponential", "linear"], { "default": "exponential" }),
            "stochastic_sampling":      TRILEAN_WITH_DEFAULT,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = FM_EULER_SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs):
        scheduler_config = {"scheduler": "fm_euler"}
        for k, v in kwargs.items():
            if k in ["shift", "base_shift", "max_shift", "base_image_seq_len", "max_image_seq_len", "shift_terminal", "time_shift_type"]:
                scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)


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


class BNBQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "load_in_8bit":                     BOOLEAN_DEFAULT_FALSE,
            "load_in_4bit":                     BOOLEAN_DEFAULT_FALSE,
            "llm_int8_threshold":               ("FLOAT", { "default": 6.0 }),
            # "llm_int8_skip_modules":           ("STRING",),
            "llm_int8_enable_fp32_cpu_offload": BOOLEAN_DEFAULT_FALSE,
            "llm_int8_has_fp16_weight":         BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_compute_dtype":           VARIANT,
            "bnb_4bit_quant_type":              (["nf4", "fp4"], { "default": "nf4" }),
            "bnb_4bit_use_double_quant":        BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_quant_storage":           (["bf16", "fp8", "fp16", "fp32", "fp64", "cp32", "cp64", "cp128", "int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int16", "int32", "int64","bool"], { "default": "fp16" }),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        kwargs["backend"] = "bitsandbytes"
        return (kwargs,)


class QTOQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "quant_type": (["float8", "int2", "int4", "int8"], { "default": "int8" }),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        kwargs["backend"] = "quanto"
        return (kwargs,)


class SNQQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
                "quant_type": (["int16", "int8", "int7", "int6", "int5", "int4", "int3", "int2", "uint16", "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "float16", "float8_em4m3fn", "float7_em3m3fn", "float6_em3m2fn", "float5_em2m2fn", "float4_em2m1fn", "float3_em1m1fn", "float2_em1m0fn", "float1_em1m0fnu"], { "default": "int8" }),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        kwargs["backend"] = "sdnq"
        return (kwargs,)


class TAOQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "quant_type": (["int4wo", "int4dq", "int8wo", "int8dq", "uint1wo", "uint2wo", "uint3wo", "uint4wo", "uint5wo", "uint6wo", "uint7wo", "float8wo"], { "default": "int8dq" }),
                "quant_type":  ("STRING",),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        # return (f"tao,{quantize_to}",)
        kwargs["backend"] = "torchao"
        return (kwargs,)


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


class QuantizationConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "optional": {
            "transformer": MODEL_QUANT_CONFIG,
            "encoder":     MODEL_QUANT_CONFIG,
            "vae":         MODEL_QUANT_CONFIG,
            "tokenizer":   MODEL_QUANT_CONFIG,
            "misc":        MODEL_QUANT_CONFIG,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


"""
class ADSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "positive_prompt": PROMPT,
                "seed": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
                "num_frames": NUM_FRAMES,
            },
            "optional": {
                "negative_prompt": PROMPT,
                "ip_image": IMAGE,
                "ip_adapter_scale": IP_ADAPTER_SCALE,
                "control_image": IMAGE,
                "controlnet_scale": CONTROLNET_SCALE,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        width,
        height,
        positive_prompt,
        seed,
        steps,
        guidance_scale,
        num_frames,
        negative_prompt=None,
        ip_image=None,
        ip_adapter_scale=None,
        control_image=None,
        controlnet_scale=None,
    ):
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        data = {
            "positive": positive_prompt,
            "width":    width,
            "height":   height,
            "seed":     seed,
            "steps":    steps,
            "cfg":      guidance_scale,
            "frames":   num_frames,
        }

        if negative_prompt is not None: data["negative"] = negative_prompt

        if ip_image is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            if ip_adapter_scale is not None: data["ip_adapter_scale"] = ip_adapter_scale

        if control_image is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            if controlnet_scale is not None: data["controlnet_scale"] = controlnet_scale

        response = get_current_manager().get_result(host, data)
        if response is not None:
            images = decode_b64_and_unpickle(response)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            print("Successfully created media")
            return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
        assert False, "No media generated.\nCheck console for details."
"""


class SDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "positive_embeds": CONDITIONING,
                "negative_embeds": CONDITIONING,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "s33d": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
                "clip_skip": CLIP_SKIP,
                "denoising_start_step": DENOISING_START_STEP,
                "denoising_end_step": DENOISING_END_STEP,
                "ip_adapter_scale": IP_ADAPTER_SCALE,
                "controlnet_scale": CONTROLNET_SCALE,
            },
            "optional": {
                "ip_image": IMAGE,
                "control_image": IMAGE,
                "latent": LATENT,
                "scheduler": SCHEDULER,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        positive_embeds,
        negative_embeds,
        width,
        height,
        s33d,
        steps,
        guidance_scale,
        clip_skip,
        denoising_start_step,
        denoising_end_step,
        ip_adapter_scale,
        controlnet_scale,
        ip_image=None,
        control_image=None,
        latent=None,
        scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            "clip_skip":        clip_skip,
            "denoising_start":  denoising_start_step,
            "denoising_end":    denoising_end_step,
            "positive_embeds":  pickle_and_encode_b64(positive_embeds),
            "negative_embeds":  pickle_and_encode_b64(negative_embeds),
        }

        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if scheduler is not None:       data["scheduler"] = json.dumps(scheduler)
        if ip_image is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            if ip_adapter_scale is not None: data["ip_adapter_scale"] = ip_adapter_scale
        if control_image is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            if controlnet_scale is not None: data["controlnet_scale"] = controlnet_scale

        response = get_current_manager().get_result(host, data)
        if response is not None:
            image_out, latent_out = response
            print("Successfully created media")
            return (host, convert_b64_to_nhwc_tensor(image_out), { "samples": decode_b64_and_unpickle(latent_out) },)
        assert False, "No media generated.\nCheck console for details."


class SDSamplerPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "positive": PROMPT,
                "negative": PROMPT,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "s33d": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
                "clip_skip": CLIP_SKIP,
                "denoising_start_step": DENOISING_START_STEP,
                "denoising_end_step": DENOISING_END_STEP,
                "ip_adapter_scale": IP_ADAPTER_SCALE,
                "controlnet_scale": CONTROLNET_SCALE,
                "use_compel": BOOLEAN_DEFAULT_FALSE,
            },
            "optional": {
                "ip_image": IMAGE,
                "control_image": IMAGE,
                "latent": LATENT,
                "scheduler": SCHEDULER,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        positive,
        negative,
        width,
        height,
        s33d,
        steps,
        guidance_scale,
        clip_skip,
        denoising_start_step,
        denoising_end_step,
        ip_adapter_scale,
        controlnet_scale,
        use_compel,
        ip_image=None,
        control_image=None,
        latent=None,
        scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            "clip_skip":        clip_skip,
            "denoising_start":  denoising_start_step,
            "denoising_end":    denoising_end_step,
            "positive":         positive,
            "negative":         negative,
            "use_compel":       use_compel,
        }

        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if scheduler is not None:       data["scheduler"] = json.dumps(scheduler)
        if ip_image is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            if ip_adapter_scale is not None: data["ip_adapter_scale"] = ip_adapter_scale
        if control_image is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            if controlnet_scale is not None: data["controlnet_scale"] = controlnet_scale

        response = get_current_manager().get_result(host, data)
        if response is not None:
            image_out, latent_out = response
            print("Successfully created media")
            return (host, convert_b64_to_nhwc_tensor(image_out), { "samples": decode_b64_and_unpickle(latent_out) },)
        assert False, "No media generated.\nCheck console for details."


class SVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "image": IMAGE,
                "s33d": SEED,
                "steps": STEPS,
                "decode_chunk_size": DECODE_CHUNK_SIZE,
                "num_frames": NUM_FRAMES,
                "motion_bucket_id": MOTION_BUCKET_ID,
                "noise_aug_strength": NOISE_AUG_STRENGTH,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        width,
        height,
        image,
        s33d,
        steps,
        decode_chunk_size,
        num_frames,
        motion_bucket_id,
        noise_aug_strength
    ):
        assert (image is not None), "You must provide an image."

        image = image.squeeze(0)                    # NHWC -> HWC
        b64_image = convert_tensor_to_b64(image)
        data = {
            "image":                b64_image,
            "width":                width,
            "height":               height,
            "seed":                 s33d,
            "steps":                steps,
            "decode_chunk_size":    decode_chunk_size,
            "frames":               num_frames,
            "motion_bucket_id":     motion_bucket_id,
            "noise_aug_strength":   noise_aug_strength,
        }
        response = get_current_manager().get_result(host, data)
        if response is not None:
            images = decode_b64_and_unpickle(response)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            print("Successfully created media")
            return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
        assert False, "No media generated.\nCheck console for details."


class SDUpscaleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "image": IMAGE,
                "positive_prompt": PROMPT,
                "seed": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
            },
            "optional": {
                "negative_prompt": PROMPT,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        image,
        positive_prompt,
        seed,
        steps,
        guidance_scale,
        negative_prompt=None
    ):
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        if image.size(0) > 1:   images = list(torch.unbind(image, 0))   # NHWC -> [HWC], len == N
        else:                   images = [image.squeeze(0)]             # NHWC -> [HWC], len == 1
        tensors = []
        i = 0
        for im in images:
            i += 1
            print(f"Upscaling image: {i}/{len(images)}")
            b64_image = convert_tensor_to_b64(im)
            data = {
                "image": b64_image,
                "positive": positive_prompt,
                "seed": seed,
                "steps": steps,
                "cfg": guidance_scale,
            }

            if negative_prompt is not None: data["negative"] = negative_prompt

            try:
                response = get_current_manager().get_result(host, data)
                if response is not None:
                    print(f"Finished upscaling image: {i}/{len(images)}")
                    im2 = decode_b64_and_unpickle(response)
                    tensors.append(convert_image_to_hwc_tensor(im2))
                else:
                    if len(images) == 1:
                        print("No media generated")
                    else:
                        print(f"Error processing image: {i}/{len(images)}")
            except Exception as e:
                print("Error getting data from server.")
                print(str(e))
        print("Successfully created media")
        return (host, torch.stack(tuple(tensors)),)       # HWC -> NHWC


class WanSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                # "positive_embeds": CONDITIONING,
                # "negative_embeds": CONDITIONING,
                "positive": PROMPT,
                "negative": PROMPT,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "s33d": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
                # "clip_skip": CLIP_SKIP,
                # "denoising_start_step": DENOISING_START_STEP,
                # "denoising_end_step": DENOISING_END_STEP,
                # "ip_adapter_scale": IP_ADAPTER_SCALE,
                # "controlnet_scale": CONTROLNET_SCALE,
                "num_frames": NUM_FRAMES,
            },
            "optional": {
                "image": IMAGE,
                # "ip_image": IMAGE,
                # "control_image": IMAGE,
                # "latent": LATENT,
                # "scheduler": SCHEDULER,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        # positive_embeds,
        # negative_embeds,
        positive,
        negative,
        width,
        height,
        s33d,
        steps,
        guidance_scale,
        # clip_skip,
        # denoising_start_step,
        # denoising_end_step,
        # ip_adapter_scale,
        # controlnet_scale,
        num_frames,
        image=None,
        # ip_image=None,
        # control_image=None,
        # latent=None,
        # scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            # "clip_skip":        clip_skip,
            # "denoising_start":  denoising_start_step,
            # "denoising_end":    denoising_end_step,
            # "positive_embeds":  pickle_and_encode_b64(positive_embeds),
            # "negative_embeds":  pickle_and_encode_b64(negative_embeds),
            "positive":         positive,
            "negative":         negative,
            "frames":           num_frames,
        }

        """
        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if scheduler is not None:       data["scheduler"] = json.dumps(scheduler)
        if ip_image is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            if ip_adapter_scale is not None: data["ip_adapter_scale"] = ip_adapter_scale
        if control_image is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            if controlnet_scale is not None: data["controlnet_scale"] = controlnet_scale
        """
        if image is not None:
            image = image.squeeze(0)              # NHWC -> HWC
            data["image"] = convert_tensor_to_b64(image)

        response = get_current_manager().get_result(host, data)
        """
        if response is not None:
            image_out, latent_out = response
            print("Successfully created media")
            return (host, convert_b64_to_nhwc_tensor(image_out), { "samples": decode_b64_and_unpickle(latent_out) },)
        assert False, "No media generated.\nCheck console for details."
        """
        if response is not None:
            images = decode_b64_and_unpickle(response)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            print("Successfully created media")
            return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
        assert False, "No media generated.\nCheck console for details."


class ZImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "positive": PROMPT,
                "negative": PROMPT,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "s33d": SEED,
                "steps": STEPS,
                "guidance_scale": CFG,
                "clip_skip": CLIP_SKIP,
                "denoising_start_step": DENOISING_START_STEP,
                "denoising_end_step": DENOISING_END_STEP,
                # "ip_adapter_scale": IP_ADAPTER_SCALE,
                # "controlnet_scale": CONTROLNET_SCALE,
            },
            "optional": {
                # "ip_image": IMAGE,
                # "control_image": IMAGE,
                "latent": LATENT,
                "scheduler": FM_EULER_SCHEDULER,
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "generate", ROOT_CATEGORY_SAMPLERS

    def generate(
        self,
        host,
        positive,
        negative,
        width,
        height,
        s33d,
        steps,
        guidance_scale,
        clip_skip,
        denoising_start_step,
        denoising_end_step,
        # ip_adapter_scale,
        # controlnet_scale,
        # ip_image=None,
        # control_image=None,
        latent=None,
        scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            "clip_skip":        clip_skip,
            "denoising_start":  denoising_start_step,
            "denoising_end":    denoising_end_step,
            "positive":         positive,
            "negative":         negative,
        }

        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if scheduler is not None:       data["scheduler"] = json.dumps(scheduler)
        """
        if ip_image is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            if ip_adapter_scale is not None: data["ip_adapter_scale"] = ip_adapter_scale
        if control_image is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            if controlnet_scale is not None: data["controlnet_scale"] = controlnet_scale
        """

        response = get_current_manager().get_result(host, data)
        if response is not None:
            image_out, latent_out = response
            print("Successfully created media")
            return (host, convert_b64_to_nhwc_tensor(image_out), { "samples": decode_b64_and_unpickle(latent_out) },)
        assert False, "No media generated.\nCheck console for details."
