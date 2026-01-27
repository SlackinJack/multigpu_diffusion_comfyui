import os
from ..modules.model_finder import *


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
node_dir = os.path.join(os.path.join(comfy_root, "custom_nodes"), "multigpu_diffusion_comfyui")
outputs_dir = os.path.join(comfy_root, "output")
comfy_models_dir = os.path.join(os.path.join(comfy_root, "models"))


checkpoints = getModelSubfoldersInFolder(comfy_models_dir, ["loras"])
models = getModelFilesInFolder(comfy_models_dir, ["loras"])
models_unsafe = getModelFilesInFolderUnsafe(comfy_models_dir, ["loras"])
model_configs = getModelConfigsInFolder(comfy_models_dir, ["loras"])
loras = getModelFilesInFolder(os.path.join(comfy_models_dir, "loras"))


INT_MAX = 2 ** 32 - 1
INT_MIN = -1 * INT_MAX
BOOLEAN_DEFAULT_TRUE = ("BOOLEAN", { "default": True })
BOOLEAN_DEFAULT_FALSE = ("BOOLEAN", { "default": False })
TRILEAN_WITH_DEFAULT = (["true", "false", "default"], { "default": "default" })
CONDITIONING = ("CONDITIONING",)
IMAGE = ("IMAGE",)
LATENT = ("LATENT",)


VARIANT = (["bf16", "fp16", "fp32"], { "default": "fp16" })
NPROC_PER_NODE = ("INT", { "default": 2, "min": 2, "max": INT_MAX, "step": 1 })
COMPEL_MODEL_LIST = (["sd1", "sd2", "sdxl"], { "default": "sdxl" })
SUPPORTED_MODEL_LIST = (["ad", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"], { "default": "sdxl" })


CHECKPOINT = ("MD_CHECKPOINT",)
CHECKPOINT_LIST = (checkpoints,)
MODEL = ("MD_MODEL",)
MODEL_LIST = (models,)
UNSAFE_MODEL_LIST = (models_unsafe,)
MODEL_CONFIG = ("MD_MODEL_CONFIG",)
MODEL_CONFIG_LIST = (model_configs,)
LORA = ("MD_LORA",)
LORA_LIST = (loras,)
LORA_WEIGHT = ("FLOAT", { "default": 1.000, "min": INT_MIN, "max": INT_MAX, "step": 0.001 })


HOST = ("MD_HOST",)
HOST_CONFIG = ("MD_HOST_CONFIG",)
BACKEND = (["asyncdiff", "balanced"], { "default": "asyncdiff" })
PORT = ("INT", { "default": 6000, "min": 1025, "max": 65535, "step": 1 })
MASTER_PORT = ("INT", { "default": 29400, "min": 1025, "max": 65535, "step": 1 })
PIPELINE_INIT_TIMEOUT = ("INT", { "default": 600, "min": 0, "max": INT_MAX, "step": 1 })


MODEL_QUANT_CONFIG = ("MD_MODEL_QUANT_CONFIG",)
QUANT_CONFIG = ("MD_QUANT_CONFIG",)
BNB_QUANTS = (["int4", "int8"], { "default": "int8" })
BNB_QUANT_TYPES = (["nf4", "fp4"], { "default": "nf4" })
BNB_QUANT_STORAGE_TYPES = (["bf16", "fp8", "fp16", "fp32", "fp64", "cp32", "cp64", "cp128", "int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int16", "int32", "int64","bool"], { "default": "fp16" })
TAO_QUANTS = (["int4wo", "int4dq", "int8wo", "int8dq", "uint1wo", "uint2wo", "uint3wo", "uint4wo", "uint5wo", "uint6wo", "uint7wo", "float8wo"], { "default": "int8dq" })


TORCH_CONFIG = ("MD_TORCH_CONFIG",)
TORCH_CACHE_LIMIT = ("INT", { "default": 16, "min": 0, "max": INT_MAX, "step": 1 })
TORCH_ACCUMULATED_CACHE_LIMIT = ("INT", { "default": 128, "min": 0, "max": INT_MAX, "step": 1 })


COMPILE_CONFIG = ("MD_COMPILE_CONFIG",)
COMPILE_BACKENDS = (["default", "inductor", "eager"], { "default": "default" })
COMPILE_MODES = (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], { "default": "default" })


PROMPT = ("STRING", { "default": "", "multiline": True })
RESOLUTION = ("INT", { "default": 512, "min": 8, "max": INT_MAX, "step": 8 })
SEED = ("INT", { "default": 0, "min": 0, "max": INT_MAX, "step": 1 })
STEPS = ("INT", { "default": 60, "min": 1, "max": INT_MAX, "step": 1 })
CLIP_SKIP = ("INT", { "default": 0, "min": 0, "max": INT_MAX, "step": 1 })
DENOISING_START_STEP = ("INT", { "default": 0, "min": 0, "max": INT_MAX, "step": 1 })
DENOISING_END_STEP = ("INT", { "default": INT_MAX, "min": 1, "max": INT_MAX, "step": 1 })
CFG = ("FLOAT", { "default": 7.000, "min": 0, "max": INT_MAX, "step": 0.001 })
IP_ADAPTER_SCALE = ("FLOAT", { "default": 0.5000, "min": 0.0000, "max": INT_MAX, "step": 0.0001 })
CONTROLNET_SCALE = ("FLOAT", { "default": 0.5000, "min": 0.0000, "max": INT_MAX, "step": 0.0001 })


SCHEDULER = ("MD_SCHEDULER",)
SCHEDULER_LIST = (["ddim", "ddpm", "deis", "dpm_2", "dpm_2_a", "dpm_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "euler", "euler_a", "heun", "ipndm", "lms", "pndm", "tcd", "unipc"], { "default": "ddim" })
TIMESTEP_LIST = (["default", "leading", "linspace", "trailing"], { "default": "default" })
BETA_LIST = (["default", "linear", "scaled_linear", "squaredcos_cap_v2"], { "default": "default" })
BETA_START = ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 })
BETA_END = ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 })


# SVD
DECODE_CHUNK_SIZE = ("INT", { "default": 8, "min": 1, "max": INT_MAX, "step": 1 })
NUM_FRAMES = ("INT", { "default": 25, "min": 1, "max": INT_MAX, "step": 1 })
MOTION_BUCKET_ID = ("INT", { "default": 180, "min": 1, "max": INT_MAX, "step": 1 })
NOISE_AUG_STRENGTH = ("FLOAT", { "default": 0.001, "min": INT_MIN, "max": INT_MAX, "step": 0.001 })
SCALE_PERCENTAGE = ("FLOAT", { "default": 100.000, "min": 0.001, "max": INT_MAX, "step": 0.001 })


BACKEND_CONFIG = ("MD_BACKEND_CONFIG",)
MODEL_N = ("INT", { "default": 2, "min": 1, "max": INT_MAX, "step": 1 })
STRIDE = ("INT", { "default": 1, "min": 1, "max": INT_MAX, "step": 1 })
SYNCED_STEPS = ("INT", { "default": 10, "min": 0, "max": INT_MAX, "step": 1 })
SYNCED_PERCENT = ("FLOAT", { "default": 10.000, "min": 0, "max": 100, "step": 0.001 })


ROOT_CATEGORY = "MultiGPU Diffusion"
ROOT_CATEGORY_CONFIG = f"{ROOT_CATEGORY}/Configuration"
ROOT_CATEGORY_GENERAL = f"{ROOT_CATEGORY}/General"
ROOT_CATEGORY_TOOLS = f"{ROOT_CATEGORY}/Tools"
ROOT_CATEGORY_SAMPLERS = f"{ROOT_CATEGORY}/Samplers"
ASYNCDIFF_CATEGORY = f"{ROOT_CATEGORY_SAMPLERS}/AsyncDiff"


HOST_CONFIGS = {
    "port": PORT,
    "master_port": MASTER_PORT,
    "nproc_per_node": NPROC_PER_NODE,
    "backend": BACKEND,
    "cuda_visible_devices": ("STRING", { "default": "", "multiline": False }),
}


def trilean(value):
    if value == "true": return True
    if value == "false": return False
    return None


def get_root_dir():
    global comfy_root
    return comfy_root


def get_node_dir():
    global node_dir
    return node_dir


def get_models_dir():
    global comfy_models_dir
    return comfy_models_dir
