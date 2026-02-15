import json
import torch


from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


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
                # "clip_skip": CLIP_SKIP,
                # "denoising_start_step": DENOISING_START_STEP,
                # "denoising_end_step": DENOISING_END_STEP,
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
        # clip_skip,
        # denoising_start_step,
        # denoising_end_step,
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
            # "clip_skip":        clip_skip,
            # "denoising_start":  denoising_start_step,
            # "denoising_end":    denoising_end_step,
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
