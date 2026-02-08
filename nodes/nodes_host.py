import os
import time


from comfy.utils import ProgressBar


from .data_types import *
from ..modules.host_manager import HostManager
from ..multigpu_diffusion.modules.utils import *


hm = HostManager()


def get_current_manager():
    global hm
    return hm


class CreateHost:
    @classmethod
    def INPUT_TYPES(s): return { "required": HOST_CONFIGS }
    RETURN_TYPES, FUNCTION, CATEGORY = HOST, "create_host", ROOT_CATEGORY_GENERAL
    def create_host(self, **kwargs):
        global hm
        host = hm.launch_host(kwargs)
        return (host,)


class CloseHost:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": { "host": HOST, "wait_for_close": BOOLEAN_DEFAULT_TRUE },
        "optional": { "image": IMAGE, "latent": LATENT }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE", "LATENT",), "destroy_host", ROOT_CATEGORY_GENERAL
    def destroy_host(self, host, wait_for_close, image=None, latent=None):
        global hm
        assert not (image is None and latent is None), "An output needs to be chained to this node in order for this node to work"
        hm.close_host_process(host, "Closed by node", wait_for_close=wait_for_close)
        return (image, latent,)


class ApplyPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "backend_config": BACKEND_CONFIG,
                "checkpoint": MODEL,
                "pipeline_type": SUPPORTED_MODEL_LIST,
                "variant": VARIANT,
                "vae_fp16": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_tiling": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_slicing": BOOLEAN_DEFAULT_FALSE,
                "enable_attention_slicing": BOOLEAN_DEFAULT_FALSE,
                "xformers_efficient": BOOLEAN_DEFAULT_FALSE,
            },
            "optional": {
                "lora": LORA,
                "transformer": MODEL,
                "vae": MODEL,
                "control_net": MODEL,
                "ip_adapter": MODEL,
                "text_encoder": MODEL,
                "text_encoder_2": MODEL,
                "text_encoder_3": MODEL,
                # "motion_module": MODEL,
                # "motion_adapter": MODEL,
                # "motion_adapter_lora": MOTION_ADAPTER_LORA,
                "compile_config": COMPILE_CONFIG,
                "quantization_config": QUANT_CONFIG,
                "torch_config": TORCH_CONFIG,
                "group_offload_config": GROUP_OFFLOAD_CONFIG,
            },
        }
    RETURN_TYPES, FUNCTION, CATEGORY = HOST, "apply_pipeline", ROOT_CATEGORY_CONFIG
    def apply_pipeline(self, **kwargs):
        global hm
        host = kwargs.pop("host")
        data = {}
        for k,v in kwargs.items():
            if k not in ["checkpoint", "transformer", "vae", "control_net", "ip_adapter", "motion_module", "motion_adapter", "text_encoder", "text_encoder_2", "text_encoder_3"]:
                data[k] = v
                continue
            else:
                if v.get("checkpoint") is not None:
                    data[k] = { "checkpoint": os.path.join(get_models_dir(), v["checkpoint"]) }
                    continue
                else:
                    data[k] = { "model": os.path.join(get_models_dir(), v["model"]), "config": os.path.join(get_models_dir(), v["config"]) }
                    continue

        response = hm.post_to_address(host, "apply", data)
        if response is None or response.status_code != 200:
            hm.close_host_process(host, "Failed to initialize pipeline", with_assert="Failed to initialize pipeline.\n\nCheck console for details.")
        return (host,)


class OffloadPipeline:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": { "host": HOST },
        "optional": { "image": IMAGE, "latent": LATENT },
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "offload_pipeline", ROOT_CATEGORY_CONFIG
    def offload_pipeline(self, host, image=None, latent=None):
        global hm
        assert not (image is None and latent is None), "An output needs to be chained to this node in order for this node to work"
        response = hm.get_from_address(host, "offload")
        # TODO: maybe do something with response
        return (host, image, latent,)
