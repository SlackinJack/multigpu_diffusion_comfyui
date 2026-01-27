import os


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
        address = hm.launch_host(kwargs)
        return (address,)


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
                "address": HOST,
                "backend_config": BACKEND_CONFIG,
                "checkpoint": MODEL,
                "pipeline_type": SUPPORTED_MODEL_LIST,
                "variant": VARIANT,
                "vae_fp16": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_tiling": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_slicing": BOOLEAN_DEFAULT_FALSE,
                "xformers_efficient": BOOLEAN_DEFAULT_FALSE,
                "enable_model_cpu_offload": BOOLEAN_DEFAULT_FALSE,
                "enable_sequential_cpu_offload": BOOLEAN_DEFAULT_FALSE,
            },
            "optional": {
                "lora": LORA,
                "vae": MODEL,
                "control_net": MODEL,
                "ip_adapter": MODEL,
                # "motion_module": MODEL,
                # "motion_adapter": MODEL,
                # "motion_adapter_lora": MOTION_ADAPTER_LORA,
                "compile_config": COMPILE_CONFIG,
                "quantization_config": QUANT_CONFIG,
                "torch_config": TORCH_CONFIG,
            },
        }
    RETURN_TYPES, FUNCTION, CATEGORY = HOST, "apply_pipeline", ROOT_CATEGORY_CONFIG
    def apply_pipeline(self, **kwargs):
        global hm
        address = kwargs.pop("address")
        data = {}
        for k,v in kwargs.items():
            if k not in ["checkpoint", "vae", "control_net", "ip_adapter", "motion_module", "motion_adapter"]:
                data[k] = v
                continue
            else:
                if v.get("checkpoint") is not None:
                    data[k] = os.path.join(get_models_dir(), v["checkpoint"])
                    continue
                else:
                    data[k] = os.path.join(get_models_dir(), v["model"])
                    data[k + "_config"] = os.path.join(get_models_dir(), v["config"])
                    continue
        response = hm.post_to_address(address, "apply", data)
        if response.status_code != 200:
            hm.close_host_process(address, "Failed to initialize pipeline", with_assert="Failed to initialize pipeline.\n\nCheck console for details.")
        return (address,)


class OffloadPipeline:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": { "address": HOST },
        "optional": { "image": IMAGE, "latent": LATENT },
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "IMAGE", "LATENT",), "offload_pipeline", ROOT_CATEGORY_CONFIG
    def offload_pipeline(self, address, image=None, latent=None):
        global hm
        assert not (image is None and latent is None), "An output needs to be chained to this node in order for this node to work"
        response = hm.get_from_address(address, "offload")
        # TODO: maybe do something with response
        return (address, image, latent,)
