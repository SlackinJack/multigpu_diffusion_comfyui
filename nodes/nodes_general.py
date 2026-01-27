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
    def INPUT_TYPES(s): return { "required": { "scheduler": SCHEDULER_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, scheduler): return ({ "scheduler": scheduler },)


class AdvancedSchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler":                SCHEDULER_LIST,
            "timestep_spacing":         TIMESTEP_LIST,
            "beta_schedule":            BETA_LIST,
            "beta_start":               BETA_START,
            "beta_end":                 BETA_END,
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
            # "quantize_to": BNB_QUANTS,
            "load_in_8bit": BOOLEAN_DEFAULT_FALSE,
            "load_in_4bit": BOOLEAN_DEFAULT_FALSE,
            "llm_int8_threshold": ("FLOAT", { "default": 6.0 }),
            # "llm_int8_skip_modules": ("STRING",),
            "llm_int8_enable_fp32_cpu_offload": BOOLEAN_DEFAULT_FALSE,
            "llm_int8_has_fp16_weight": BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_compute_dtype": VARIANT,
            "bnb_4bit_quant_type": BNB_QUANT_TYPES,
            "bnb_4bit_use_double_quant": BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_quant_storage": BNB_QUANT_STORAGE_TYPES,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        # match quantize_to:
        #     case "int4":    return (f"bnb,{quantize_to},{int4_compute_type},{int4_quant_storage},{int4_quant_type}",)
        #     case _:         return (f"bnb,{quantize_to}",)
        kwargs["backend"] = "bitsandbytes"
        # if llm_int8_skip_modules is not None: kwargs["llm_int8_skip_modules"] = llm_int8_skip_modules.split(",")
        return (kwargs,)


class TAOQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "quant_type":  ("STRING",) } } # TAO_QUANTS
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        # return (f"tao,{quantize_to}",)
        kwargs["backend"] = "torchao"
        return (kwargs,)


class TorchConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "torch_cache_limit": TORCH_CACHE_LIMIT,
            "torch_accumlated_cache_limit": TORCH_ACCUMULATED_CACHE_LIMIT,
            "torch_capture_scalar": BOOLEAN_DEFAULT_FALSE,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = TORCH_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)


class CompileConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "compile_unet":             BOOLEAN_DEFAULT_FALSE,
            "compile_vae":              BOOLEAN_DEFAULT_FALSE,
            "compile_encoder":          BOOLEAN_DEFAULT_FALSE,
            "compile_backend":          COMPILE_BACKENDS,
            "compile_mode":             COMPILE_MODES,
            "compile_options":          ("STRING", { "default": "", "multiline": False }),
            "compile_fullgraph_off":    BOOLEAN_DEFAULT_FALSE,
        }
    }

    RETURN_TYPES, FUNCTION, CATEGORY = COMPILE_CONFIG, "get_config", ROOT_CATEGORY_CONFIG

    def get_config(self, compile_unet, compile_vae, compile_encoder, compile_backend, compile_mode, compile_options, compile_fullgraph_off):
        out = {}
        if compile_unet is True:            out["compile_unet"] = True
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
            "quantize_unet": MODEL_QUANT_CONFIG,
            "quantize_encoder": MODEL_QUANT_CONFIG,
            "quantize_vae": MODEL_QUANT_CONFIG,
            "quantize_tokenizer": MODEL_QUANT_CONFIG,
            "quantize_misc": MODEL_QUANT_CONFIG,
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
