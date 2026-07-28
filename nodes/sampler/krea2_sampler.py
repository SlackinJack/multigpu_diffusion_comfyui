import json
import torch


from comfy.utils import ProgressBar


from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


IMAGE_SUCCESS_MESSAGE = "🏁 Successfully created image"
MULTI_SUCCESS_MESSAGE = "🏁 Successfully created frames"
NO_MEDIA_GENERATED = "No media generated.\nCheck console for details."


class Krea2Sampler:
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
                "denoising_start_step": DENOISING_START_STEP,
                "denoising_end_step": DENOISING_END_STEP,
                # "ip_adapter_scale": IP_ADAPTER_SCALE,
                # "controlnet_scale": CONTROLNET_SCALE,
            },
            "optional": {
                # "ip_image": IMAGE,
                # "control_image": IMAGE,
                "latent": LATENT,
                "fm_scheduler": FM_SCHEDULER,
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
        denoising_start_step,
        denoising_end_step,
        # ip_adapter_scale,
        # controlnet_scale,
        # ip_image=None,
        # control_image=None,
        latent=None,
        fm_scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            "denoising_start":  denoising_start_step,
            "denoising_end":    denoising_end_step,
            "positive":         positive,
            "negative":         negative,
        }

        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if fm_scheduler is not None:    data["scheduler"] = json.dumps(fm_scheduler)
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

        pbar = ProgressBar(100)
        response = get_current_manager().get_result(host, data, pbar=pbar)
        assert response is not None, NO_MEDIA_GENERATED
        image_out, latent_out = response
        get_current_manager().log(IMAGE_SUCCESS_MESSAGE)
        return (host, convert_b64_to_nhwc_tensor(image_out), { "samples": decode_b64_and_unpickle(latent_out) },)
