import os
from ..modules.model_finder import *


cwd                 = os.path.dirname(__file__)
comfy_root          = os.path.dirname(os.path.dirname(cwd)) + "/../"
outputs_dir         = os.path.join(comfy_root, "output")
models_dir          = os.path.join(os.path.join(comfy_root, "models"))
checkpoints_dir     = os.path.join(models_dir, "checkpoints")
checkpoints         = getModelSubfoldersInFolder(checkpoints_dir)
ggufs               = getModelFilesInFolderGGUF(checkpoints_dir)
controlnet_dir      = os.path.join(models_dir, "controlnet")
controlnets         = getModelSubfoldersInFolder(controlnet_dir)
loras_dir           = os.path.join(models_dir, "loras")
loras               = getModelFilesInFolder(loras_dir)
vae_dir             = os.path.join(models_dir, "vae")
vaes                = getModelFilesInFolder(vae_dir)
motion_adapter_dir  = os.path.join(models_dir, "animatediff_models")
motion_adapters     = getModelSubfoldersInFolder(motion_adapter_dir)


# start unsafe loaders
ipadapter_dir       = os.path.join(models_dir, "ipadapter")
ipadapters          = getModelFilesInFolderUnsafe(ipadapter_dir)
motion_modules      = getModelFilesInFolderUnsafe(motion_adapter_dir)
# end unsafe loaders


INT_MAX                         = 2 ** 32 - 1
INT_MIN                         = -1 * INT_MAX
BOOLEAN_DEFAULT_TRUE            = ("BOOLEAN",                           { "default": True })
BOOLEAN_DEFAULT_FALSE           = ("BOOLEAN",                           { "default": False })
TRILEAN_WITH_DEFAULT            = (["true", "false", "default"],        { "default": "default" })
GENERIC_CONFIG                  = ("MD_GENERIC_CONFIG",)
ASYNCDIFF_CONFIG                = ("MD_ASYNCDIFF_CONFIG",)
DISTRIFUSER_CONFIG              = ("MD_DISTRIFUSER_CONFIG",)
XDIT_CONFIG                     = ("MD_XDIT_CONFIG",)
XDIT_USP_CONFIG                 = ("MD_XDIT_USP_CONFIG",)
CONDITIONING                    = ("CONDITIONING",)
IMAGE                           = ("IMAGE",)
LATENT                          = ("LATENT",)
CONTROLNET                      = ("MD_CONTROLNET",)
CONTROLNET_LIST                 = (controlnets,)
LORA                            = ("MD_LORA",)
LORA_LIST                       = (loras,)
CHECKPOINT                      = ("MD_CHECKPOINT",)
CHECKPOINT_LIST                 = (checkpoints,)
MODEL_GGUF                      = ("MD_GGUF",)
GGUF_LIST                       = (ggufs,)
IPADAPTER                       = ("MD_IPADAPTER",)
IPADAPTER_LIST                  = (ipadapters,)
VAE                             = ("MD_VAE",)
VAE_LIST                        = (vaes,)
MOTION_MODULE                   = ("MD_MOTION_MODULE",)
MOTION_MODULE_LIST              = (motion_modules,)
MOTION_ADAPTER                  = ("MD_MOTION_ADAPTER",)
MOTION_ADAPTER_LORA             = ("MD_MOTION_ADAPTER_LORA",)
MOTION_ADAPTER_LIST             = (motion_adapters,)
SCHEDULER                       = ("MD_SCHEDULER",)
SCHEDULER_LIST                  = ([
                                    "dpmpp_2m",
                                    "dpmpp_2m_sde",
                                    "dpmpp_sde",
                                    "dpm_2",
                                    "dpm_2_a",
                                    "euler",
                                    "euler_a",
                                    "heun",
                                    "lms",
                                    "ddim",
                                    "deis",
                                    "dpm_sde",
                                    "pndm",
                                    "tcd",
                                    "unipc"
                                ],                                      { "default": "ddim" })
TIMESTEP_LIST                   = (["default", "leading",
                                    "linspace", "trailing"],            { "default": "default" })
VARIANT                         = (["bf16", "fp16", "fp32"],            { "default": "fp16" })
QUANT                           = (["disabled", "float8",
                                    "int8", "int4", "int2"],            { "default": "disabled" })
NPROC_PER_NODE                  = ("INT",                               { "default": 2,         "min": 2,           "max": INT_MAX,     "step": 1 })
SCALE_PERCENTAGE                = ("FLOAT",                             { "default": 100.0,     "min": 0.01,        "max": INT_MAX,     "step": 0.01 })
PROMPT                          = ("STRING",                            { "default": "",        "multiline": True })
RESOLUTION                      = ("INT",                               { "default": 512,       "min": 8,           "max": INT_MAX,     "step": 8 })
SEED                            = ("INT",                               { "default": 0,         "min": 0,           "max": INT_MAX,     "step": 1 })
STEPS                           = ("INT",                               { "default": 60,        "min": 1,           "max": INT_MAX,     "step": 1 })
CLIP_SKIP                       = ("INT",                               { "default": 0,         "min": 0,           "max": INT_MAX,     "step": 1 })
DENOISE                         = ("FLOAT",                             { "default": 1.00,      "min": 0.00,        "max": 1.0,         "step": 0.01 })
DECODE_CHUNK_SIZE               = ("INT",                               { "default": 8,         "min": 1,           "max": INT_MAX,     "step": 1 })
NUM_FRAMES                      = ("INT",                               { "default": 25,        "min": 1,           "max": INT_MAX,     "step": 1 })
MOTION_BUCKET_ID                = ("INT",                               { "default": 180,       "min": 1,           "max": INT_MAX,     "step": 1 })
WARM_UP_STEPS                   = ("INT",                               { "default": 10,        "min": 0,           "max": INT_MAX,     "step": 1 })
PIPELINE_INIT_TIMEOUT           = ("INT",                               { "default": 600,       "min": 0,           "max": INT_MAX,     "step": 1 })
CFG                             = ("FLOAT",                             { "default": 7.0,       "min": 0,           "max": INT_MAX,     "step": 0.1 })
NOISE_AUG_STRENGTH              = ("FLOAT",                             { "default": 0.01,      "min": 0,           "max": INT_MAX,     "step": 0.01 })
LORA_WEIGHT                     = ("FLOAT",                             { "default": 1.00,      "min": INT_MIN,     "max": INT_MAX,     "step": 0.01 })
CONTROLNET_SCALE                = ("FLOAT",                             { "default": 1.00,      "min": INT_MIN,     "max": INT_MAX,     "step": 0.01 })
IP_ADAPTER_SCALE                = ("FLOAT",                             { "default": 1.00,      "min": INT_MIN,     "max": INT_MAX,     "step": 0.01 })

PORT                            = ("INT",                               { "default": 6000,      "min": 1025,        "max": 65535,       "step": 1 })
MASTER_PORT                     = ("INT",                               { "default": 29400,     "min": 1025,        "max": 65535,       "step": 1 })



COMPEL_MODEL_LIST               = (["sd1", "sd2", "sdxl"],              { "default": "sdxl" })


# asyncdiff
ASYNCDIFF_MODEL_LIST            = (["ad", "sd1", "sd2", "sd3",
                                    "sdup", "sdxl", "svd"],             { "default": "sdxl" })
MODEL_N                         = ("INT",                               { "default": 2,         "min": 2,           "max": 4,           "step": 1 })
STRIDE                          = ("INT",                               { "default": 1,         "min": 1,           "max": 2,           "step": 1 })


# distrifuser
DISTRIFUSER_MODEL_LIST          = (["sd1", "sd2", "sdxl"],              { "default": "sdxl" })
DISTRIFUSER_PARALLELISM_LIST    = (["naive_patch", "patch", "tensor"],  { "default": "patch" })
DISTRIFUSER_SYNC_MODE_LIST      = (["corrected_async_gn",
                                    "separate_gn", "stale_gn",
                                    "sync_gn", "full_sync", "no_sync"], { "default": "corrected_async_gn" })

# xdit
XDIT_MODEL_LIST                 = (["flux", "hy",
                                    "pixa", "pixs", "sd3", "sdxl"],     { "default": "flux" })
XDIT_USP_MODEL_LIST             = (["wan_t2v", "wan_t2i",
                                    "wan_i2v", "wan_flf2v",
                                    "wan_vace"],                        { "default": "wan_t2v" })
PIPEFUSION_PARALLEL_DEGREE      = ("INT",                               { "default": 2,         "min": 1,           "max": INT_MAX,     "step": 1 })
TENSOR_PARALLEL_DEGREE          = ("INT",                               { "default": 1,         "min": 1,           "max": INT_MAX,     "step": 1 })
DATA_PARALLEL_DEGREE            = ("INT",                               { "default": 1,         "min": 1,           "max": INT_MAX,     "step": 1 })
ULYSSES_DEGREE                  = ("INT",                               { "default": 1,         "min": 1,           "max": INT_MAX,     "step": 1 })
RING_DEGREE                     = ("INT",                               { "default": 1,         "min": 1,           "max": INT_MAX,     "step": 1 })


ROOT_CATEGORY           = "MultiGPU Diffusion"
ROOT_CATEGORY_CONFIG    = f"{ROOT_CATEGORY}/Configuration"
ROOT_CATEGORY_GENERAL   = f"{ROOT_CATEGORY}/General"
ROOT_CATEGORY_TOOLS     = f"{ROOT_CATEGORY}/Tools"
ROOT_CATEGORY_SAMPLERS  = f"{ROOT_CATEGORY}/Samplers"
ASYNCDIFF_CATEGORY      = f"{ROOT_CATEGORY_SAMPLERS}/AsyncDiff"
DISTRIFUSER_CATEGORY    = f"{ROOT_CATEGORY_SAMPLERS}/Distrifuser"
XDIT_CATEGORY           = f"{ROOT_CATEGORY_SAMPLERS}/xDiT"


GENERIC_CONFIGS = {
    "port":                             PORT,
    "master_port":                      MASTER_PORT,
    "variant":                          VARIANT,
    "width":                            RESOLUTION,
    "height":                           RESOLUTION,
    "compile_unet":                     BOOLEAN_DEFAULT_FALSE,
    "compile_vae":                      BOOLEAN_DEFAULT_FALSE,
    "compile_text_encoder":             BOOLEAN_DEFAULT_FALSE,
    "enable_vae_tiling":                BOOLEAN_DEFAULT_FALSE,
    "enable_vae_slicing":               BOOLEAN_DEFAULT_FALSE,
    "xformers_efficient":               BOOLEAN_DEFAULT_FALSE,
    "enable_model_cpu_offload":         BOOLEAN_DEFAULT_FALSE,
    "enable_sequential_cpu_offload":    BOOLEAN_DEFAULT_FALSE,
    "quantize_to":                      QUANT,
    "warm_up_steps":                    WARM_UP_STEPS,
}


GENERIC_CONFIGS_COMFY = {
    "pipeline_init_timeout":    PIPELINE_INIT_TIMEOUT,
    "keepalive":                BOOLEAN_DEFAULT_FALSE,
}


GENERIC_CONFIGS_OPTIONAL = {
    "scheduler":            SCHEDULER,
    "lora":                 LORA,
    "ip_adapter":           IPADAPTER,
    "vae":                  VAE,
    "control_net":          CONTROLNET,
    "motion_module":        MOTION_MODULE,
    "motion_adapter":       MOTION_ADAPTER,
    "motion_adapter_lora":  MOTION_ADAPTER_LORA,
}


def trilean(value):
    if value == "true": return True
    if value == "false": return False
    return None
