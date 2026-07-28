import json
import torch


from comfy.utils import ProgressBar


from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


IMAGE_SUCCESS_MESSAGE = "🏁 Successfully created image"
MULTI_SUCCESS_MESSAGE = "🏁 Successfully created frames"
NO_MEDIA_GENERATED = "No media generated.\nCheck console for details."


class WanSampler:
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
                "num_frames": NUM_FRAMES,
            },
            "optional": {
                "image": IMAGE,
                # "latent": LATENT,
                # "scheduler": SCHEDULER,
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
        num_frames,
        image=None,
        # latent=None,
        # scheduler=None,
    ):
        data = {
            "width":            width,
            "height":           height,
            "seed":             s33d,
            "steps":            steps,
            "cfg":              guidance_scale,
            "positive":         positive,
            "negative":         negative,
            "frames":           num_frames,
        }

        # if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        # if scheduler is not None:       data["scheduler"] = json.dumps(scheduler)

        if image is not None:
            image = image.squeeze(0)              # NHWC -> HWC
            data["image"] = convert_tensor_to_b64(image)

        pbar = ProgressBar(100)
        response = get_current_manager().get_result(host, data, pbar=pbar)
        assert response is not None, NO_MEDIA_GENERATED
        images = decode_b64_and_unpickle(response)
        tensors = []
        for i in images:
            tensors.append(convert_image_to_hwc_tensor(i))
        get_current_manager().log(MULTI_SUCCESS_MESSAGE)
        return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
