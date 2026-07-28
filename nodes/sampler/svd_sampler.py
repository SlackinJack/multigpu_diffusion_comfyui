import json
import torch


from comfy.utils import ProgressBar


from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


IMAGE_SUCCESS_MESSAGE = "?? Successfully created image"
MULTI_SUCCESS_MESSAGE = "?? Successfully created frames"
NO_MEDIA_GENERATED = "No media generated.\nCheck console for details."


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

        pbar = ProgressBar(100)
        response = get_current_manager().get_result(host, data, pbar=pbar)
        assert response is not None, NO_MEDIA_GENERATED
        images = decode_b64_and_unpickle(response)
        tensors = []
        for i in images:
            tensors.append(convert_image_to_hwc_tensor(i))
        get_current_manager().log(MULTI_SUCCESS_MESSAGE)
        return (host, torch.stack(tuple(tensors)),)   # HWC -> NHWC
