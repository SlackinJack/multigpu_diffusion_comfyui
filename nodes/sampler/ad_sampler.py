import json
import torch


from comfy.utils import ProgressBar


from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


IMAGE_SUCCESS_MESSAGE = "🏁 Successfully created image"
MULTI_SUCCESS_MESSAGE = "🏁 Successfully created frames"
NO_MEDIA_GENERATED = "No media generated.\nCheck console for details."


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
        assert response is not None, NO_MEDIA_GENERATED
        images = decode_b64_and_unpickle(response)
        tensors = []
        for i in images:
            tensors.append(convert_image_to_hwc_tensor(i))
        get_current_manager().log(MULTI_SUCCESS_MESSAGE)
        return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
"""
