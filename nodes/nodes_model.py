from .data_types import *


class CheckpointSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "checkpoint": CHECKPOINT_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class ModelSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": MODEL_LIST, "config": MODEL_CONFIG_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class UnsafeModelSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": UNSAFE_MODEL_LIST, "config": MODEL_CONFIG_LIST } }
    RETURN_TYPES, FUNCTION, CATEGORY = MODEL, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class LoraSelector:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "lora": LORA_LIST, "weight": LORA_WEIGHT } }
    RETURN_TYPES, FUNCTION, CATEGORY = LORA, "get", ROOT_CATEGORY_GENERAL
    def get(self, lora, weight): return ({ f"{get_models_dir()}/loras/{lora}": weight },)


class MultiLoraJoiner:
    @classmethod
    def INPUT_TYPES(s): return { "optional": { "lora_1": LORA, "lora_2": LORA, "lora_3": LORA, "lora_4": LORA,
                                               "lora_5": LORA, "lora_6": LORA, "lora_7": LORA, "lora_8": LORA,
                                               "lora_9": LORA, "lora_10": LORA, "lora_11": LORA, "lora_12": LORA,
                                               "lora_13": LORA, "lora_14": LORA, "lora_15": LORA, "lora_16": LORA,
                                               "lora_17": LORA, "lora_18": LORA, "lora_19": LORA, "lora_20": LORA,
                                               "lora_21": LORA, "lora_22": LORA, "lora_23": LORA, "lora_24": LORA,
                                               "lora_25": LORA, "lora_26": LORA, "lora_27": LORA, "lora_28": LORA,
                                               "lora_29": LORA, "lora_30": LORA, "lora_31": LORA, "lora_32": LORA } }
    RETURN_TYPES, FUNCTION, CATEGORY = LORA, "join", ROOT_CATEGORY_GENERAL
    def join(self, **kwargs):
        out = {}
        for k, v in kwargs.items():
            for adapter, scale in v.items():
                out[adapter] = scale
        return (out,)
