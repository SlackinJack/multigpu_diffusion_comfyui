from .nodes.nodes_asyncdiff import *
from .nodes.nodes_balanced import *
from .nodes.nodes_general import *
from .nodes.nodes_host import *
from .nodes.nodes_model import *
from .nodes.nodes_quantization import *
from .nodes.nodes_sampler import *
from .nodes.nodes_scheduler import *
from .nodes.nodes_single import *


NODE_CLASS_MAPPINGS = {
    "CheckpointSelector": CheckpointSelector,
    "SchedulerSelector": SchedulerSelector,
    "AdvancedSchedulerSelector": AdvancedSchedulerSelector,
    "FlowMatchScheduler": FlowMatchScheduler,
    "AdvancedFlowMatchScheduler": AdvancedFlowMatchScheduler,
    "ModelSelector": ModelSelector,
    "UnsafeModelSelector": UnsafeModelSelector,
    "LoraSelector": LoraSelector,
    "MultiLoraJoiner": MultiLoraJoiner,

    "EncodePromptWithCompel": EncodePromptWithCompel,

    "AsyncDiffConfig": AsyncDiffConfig,
    "BalancedConfig": BalancedConfig,
    "SingleConfig": SingleConfig,
    "BitsAndBytesQuantizationConfig": BitsAndBytesQuantizationConfig,
    "QuantoQuantizationConfig": QuantoQuantizationConfig,
    "SDNQQuantizationConfig": SDNQQuantizationConfig,
    "TorchAOQuantizationConfig": TorchAOQuantizationConfig,
    "TorchConfig": TorchConfig,
    "CompileConfig": CompileConfig,
    "QuantizationConfig": QuantizationConfig,
    "GroupOffloadConfig": GroupOffloadConfig,
    "OffloadConfig": OffloadConfig,
    "AttentionBackendConfig": AttentionBackendConfig,

    "CreateHost": CreateHost,
    "CloseHost": CloseHost,
    "ApplyPipeline": ApplyPipeline,
    "SleepHost": SleepHost,
    "OffloadPipeline": OffloadPipeline,

    # "ADSampler": ADSampler,
    "FluxSampler": FluxSampler,
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
    "FlowMatchScheduler": "FlowMatchScheduler",
    "AdvancedFlowMatchScheduler": "AdvancedFlowMatchScheduler",
    "ModelSelector": "ModelSelector",
    "UnsafeModelSelector": "UnsafeModelSelector",
    "LoraSelector": "LoraSelector",
    "MultiLoraJoiner": "MultiLoraJoiner",

    "EncodePromptWithCompel": "EncodePromptWithCompel",

    "AsyncDiffConfig": "AsyncDiffConfig",
    "BalancedConfig": "BalancedConfig",
    "SingleConfig": "SingleConfig",
    "BitsAndBytesQuantizationConfig": "BitsAndBytesQuantizationConfig",
    "QuantoQuantizationConfig": "QuantoQuantizationConfig",
    "SDNQQuantizationConfig": "SDNQQuantizationConfig",
    "TorchAOQuantizationConfig": "TorchAOQuantizationConfig",
    "TorchConfig": "TorchConfig",
    "CompileConfig": "CompileConfig",
    "QuantizationConfig": "QuantizationConfig",
    "GroupOffloadConfig": "GroupOffloadConfig",
    "OffloadConfig": "OffloadConfig",
    "AttentionBackendConfig": "AttentionBackendConfig",

    "CreateHost": "CreateHost",
    "CloseHost": "CloseHost",
    "ApplyPipeline": "ApplyPipeline",
    "SleepHost": "SleepHost",
    "OffloadPipeline": "OffloadPipeline",

    # "ADSampler": "ADSampler",
    "FluxSampler": "FluxSampler",
    "SDSampler": "SDSampler",
    "SDSamplerPrompt": "SDSampler (Using Prompt)",
    "SDUpscaleSampler": "SDUpscaleSampler",
    "SVDSampler": "SVDSampler",
    "WanSampler": "WanSampler",
    "ZImageSampler": "ZImageSampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
