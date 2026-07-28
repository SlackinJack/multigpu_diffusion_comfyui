import json
import torch


from comfy.utils import ProgressBar


from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


IMAGE_SUCCESS_MESSAGE = "🏁 Successfully created image"
MULTI_SUCCESS_MESSAGE = "🏁 Successfully created frames"
NO_MEDIA_GENERATED = "No media generated.\nCheck console for details."


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
            get_current_manager().log(f"⏳ Upscaling image: {i}/{len(images)}")
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
                pbar = ProgressBar(100)
                response = get_current_manager().get_result(host, data, pbar=pbar)
                if response is not None:
                    get_current_manager().log(f"✅ Finished upscaling image: {i}/{len(images)}")
                    im2 = decode_b64_and_unpickle(response)
                    tensors.append(convert_image_to_hwc_tensor(im2))
                else:
                    if len(images) == 1:
                        get_current_manager().log("❌ No media generated")
                    else:
                        get_current_manager().log(f"❌ Error processing image: {i}/{len(images)}")
            except Exception as e:
                get_current_manager().log("❌ Error getting data from server.\n" + str(e))
        assert len(tensors) > 0, NO_MEDIA_GENERATED
        get_current_manager().log(IMAGE_SUCCESS_MESSAGE)
        return (host, torch.stack(tuple(tensors)),)       # HWC -> NHWC
