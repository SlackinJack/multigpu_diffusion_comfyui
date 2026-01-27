from .nodes.nodes_general import *
from .nodes.nodes_host import *
from .nodes.nodes_asyncdiff import *
from .nodes.nodes_balanced import *


NODE_CLASS_MAPPINGS = {
    "CheckpointSelector": CheckpointSelector,
    "SchedulerSelector": SchedulerSelector,
    "AdvancedSchedulerSelector": AdvancedSchedulerSelector,
    "ModelSelector": ModelSelector,
    "UnsafeModelSelector": UnsafeModelSelector,
    "LoraSelector": LoraSelector,
    "MultiLoraJoiner": MultiLoraJoiner,

    "EncodePromptWithCompel": EncodePromptWithCompel,

    "AsyncDiffConfig": AsyncDiffConfig,
    "BalancedConfig": BalancedConfig,
    "BNBQuantizationConfig": BNBQuantizationConfig,
    "TAOQuantizationConfig": TAOQuantizationConfig,
    "TorchConfig": TorchConfig,
    "CompileConfig": CompileConfig,
    "QuantizationConfig": QuantizationConfig,

    "CreateHost": CreateHost,
    "CloseHost": CloseHost,
    "ApplyPipeline": ApplyPipeline,
    "OffloadPipeline": OffloadPipeline,

    # "ADSampler": ADSampler,
    "SDSampler": SDSampler,
    "SDUpscaleSampler": SDUpscaleSampler,
    "SVDSampler": SVDSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSelector": "CheckpointSelector",
    "SchedulerSelector": "SchedulerSelector",
    "AdvancedSchedulerSelector": "AdvancedSchedulerSelector",
    "ModelSelector": "ModelSelector",
    "UnsafeModelSelector": "UnsafeModelSelector",
    "LoraSelector": "LoraSelector",
    "MultiLoraJoiner": "MultiLoraJoiner",

    "EncodePromptWithCompel": "EncodePromptWithCompel",

    "AsyncDiffConfig": "AsyncDiffConfig",
    "BalancedConfig": "BalancedConfig",
    "BNBQuantizationConfig": "BNBQuantizationConfig",
    "TAOQuantizationConfig": "TAOQuantizationConfig",
    "TorchConfig": "TorchConfig",
    "CompileConfig": "CompileConfig",
    "QuantizationConfig": "QuantizationConfig",

    "CreateHost": "CreateHost",
    "CloseHost": "CloseHost",
    "ApplyPipeline": "ApplyPipeline",
    "OffloadPipeline": "OffloadPipeline",

    # "ADSampler": "ADSampler",
    "SDSampler": "SDSampler",
    "SDUpscaleSampler": "SDUpscaleSampler",
    "SVDSampler": "SVDSampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
