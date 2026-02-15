from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


class BNBQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "load_in_8bit":                     BOOLEAN_DEFAULT_FALSE,
            "load_in_4bit":                     BOOLEAN_DEFAULT_FALSE,
            "llm_int8_threshold":               ("FLOAT", { "default": 6.0 }),
            # "llm_int8_skip_modules":           ("STRING",),
            "llm_int8_enable_fp32_cpu_offload": BOOLEAN_DEFAULT_FALSE,
            "llm_int8_has_fp16_weight":         BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_compute_dtype":           VARIANT,
            "bnb_4bit_quant_type":              (["nf4", "fp4"], { "default": "nf4" }),
            "bnb_4bit_use_double_quant":        BOOLEAN_DEFAULT_FALSE,
            "bnb_4bit_quant_storage":           (["bf16", "fp8", "fp16", "fp32", "int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int16", "int32", "bool"], { "default": "fp16" }),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        kwargs["backend"] = "bitsandbytes"
        return (kwargs,)


class QTOQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "quant_type": (["float8", "int2", "int4", "int8"], { "default": "int8" }),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        kwargs["backend"] = "quanto"
        return (kwargs,)


class SNQQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
                "quant_type": (["int16", "int8", "int7", "int6", "int5", "int4", "int3", "int2", "uint16", "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "float16", "float8_em4m3fn", "float7_em3m3fn", "float6_em3m2fn", "float5_em2m2fn", "float4_em2m1fn", "float3_em1m1fn", "float2_em1m0fn", "float1_em1m0fnu"], { "default": "int8" }),
                "group_size": ("INT", { "default": 0, "min": -1, "max": INT_MAX }),
                "use_svd": TRILEAN_WITH_DEFAULT,
                "svd_rank": ("INT", { "default": 32, "min": INT_MIN, "max": INT_MAX }),
                "svd_steps": ("INT", { "default": 8, "min": INT_MIN, "max": INT_MAX }),
                "quant_conv": TRILEAN_WITH_DEFAULT,
                "use_quantized_matmul": TRILEAN_WITH_DEFAULT,
                "use_quantized_matmul_conv": TRILEAN_WITH_DEFAULT,
                "dequantize_fp32": TRILEAN_WITH_DEFAULT,
                "non_blocking": TRILEAN_WITH_DEFAULT,
                "use_static_quantization": TRILEAN_WITH_DEFAULT,
                "dynamic_loss_threshold": ("FLOAT", { "default": 0.01000, "step": 0.00001 }),
                "use_stochastic_rounding": TRILEAN_WITH_DEFAULT,
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        out = {}
        out["backend"] = "sdnq"
        for k,v in kwargs.items():
            if k in ["use_svd", "quant_conv", "use_quantized_matmul", "use_quantized_matmul_conv", "dequantize_fp32", "non_blocking", "use_static_quantization", "use_stochastic_rounding"]:
                if trilean(v) != None:
                    out[k] = trilean(v)
            else:
                out[k] = v
        return (out,)


class TAOQuantizationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "quant_type": (["int4wo", "int4dq", "int8wo", "int8dq", "uint1wo", "uint2wo", "uint3wo", "uint4wo", "uint5wo", "uint6wo", "uint7wo", "float8wo"], { "default": "int8dq" }),
                "quant_type":  ("STRING",),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL_QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        # return (f"tao,{quantize_to}",)
        kwargs["backend"] = "torchao"
        return (kwargs,)


class QuantizationConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "optional": {
            "transformer": MODEL_QUANT_CONFIG,
            "encoder":     MODEL_QUANT_CONFIG,
            "vae":         MODEL_QUANT_CONFIG,
            "tokenizer":   MODEL_QUANT_CONFIG,
            "misc":        MODEL_QUANT_CONFIG,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = QUANT_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)
