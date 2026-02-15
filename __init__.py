from .nodes.nodes_asyncdiff import *
from .nodes.nodes_balanced import *
from .nodes.nodes_general import *
from .nodes.nodes_host import *
from .nodes.nodes_quantization import *
from .nodes.nodes_sampler import *
from .nodes.nodes_scheduler import *


NODE_CLASS_MAPPINGS = {
    "CheckpointSelector": CheckpointSelector,
    "SchedulerSelector": SchedulerSelector,
    "AdvancedSchedulerSelector": AdvancedSchedulerSelector,
    "AdvancedFMSchedulerSelector": AdvancedFMSchedulerSelector,
    "ModelSelector": ModelSelector,
    "UnsafeModelSelector": UnsafeModelSelector,
    "LoraSelector": LoraSelector,
    "MultiLoraJoiner": MultiLoraJoiner,

    "EncodePromptWithCompel": EncodePromptWithCompel,

    "AsyncDiffConfig": AsyncDiffConfig,
    "BalancedConfig": BalancedConfig,
    "BNBQuantizationConfig": BNBQuantizationConfig,
    "QTOQuantizationConfig": QTOQuantizationConfig,
    "SNQQuantizationConfig": SNQQuantizationConfig,
    "TAOQuantizationConfig": TAOQuantizationConfig,
    "TorchConfig": TorchConfig,
    "CompileConfig": CompileConfig,
    "QuantizationConfig": QuantizationConfig,
    "GroupOffloadConfig": GroupOffloadConfig,
    "OffloadConfig": OffloadConfig,

    "CreateHost": CreateHost,
    "CloseHost": CloseHost,
    "ApplyPipeline": ApplyPipeline,
    "OffloadPipeline": OffloadPipeline,

    # "ADSampler": ADSampler,
    "SDSampler": SDSampler,
    "SDSamplerPrompt": SDSamplerPrompt,
    "SDUpscaleSampler": SDUpscaleSampler,
    "SVDSampler": SVDSampler,
    "WanSampler": WanSampler,
    "ZImageSampler": ZImageSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSelector": "CheckpointSelector",
    "SchedulerSelector": "SchedulerSelector",
    "AdvancedSchedulerSelector": "AdvancedSchedulerSelector",
    "AdvancedFMSchedulerSelector": "AdvancedFMSchedulerSelector",
    "ModelSelector": "ModelSelector",
    "UnsafeModelSelector": "UnsafeModelSelector",
    "LoraSelector": "LoraSelector",
    "MultiLoraJoiner": "MultiLoraJoiner",

    "EncodePromptWithCompel": "EncodePromptWithCompel",

    "AsyncDiffConfig": "AsyncDiffConfig",
    "BalancedConfig": "BalancedConfig",
    "BNBQuantizationConfig": "BNBQuantizationConfig",
    "QTOQuantizationConfig": "QTOQuantizationConfig",
    "SNQQuantizationConfig": "SNQQuantizationConfig",
    "TAOQuantizationConfig": "TAOQuantizationConfig",
    "TorchConfig": "TorchConfig",
    "CompileConfig": "CompileConfig",
    "QuantizationConfig": "QuantizationConfig",
    "GroupOffloadConfig": "GroupOffloadConfig",
    "OffloadConfig": "OffloadConfig",

    "CreateHost": "CreateHost",
    "CloseHost": "CloseHost",
    "ApplyPipeline": "ApplyPipeline",
    "OffloadPipeline": "OffloadPipeline",

    # "ADSampler": "ADSampler",
    "SDSampler": "SDSampler",
    "SDSamplerPrompt": "SDSampler (Using Prompt)",
    "SDUpscaleSampler": "SDUpscaleSampler",
    "SVDSampler": "SVDSampler",
    "WanSampler": "WanSampler",
    "ZImageSampler": "ZImageSampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
