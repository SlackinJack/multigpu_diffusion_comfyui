from .nodes.nodes_asyncdiff import *
from .nodes.nodes_balanced import *
from .nodes.nodes_general import *
from .nodes.nodes_host import *
from .nodes.nodes_model import *
from .nodes.nodes_quantization import *
from .nodes.nodes_sampler import *
from .nodes.nodes_single import *
from .nodes.scheduler import *


NODE_CLASS_MAPPINGS = {
    "CheckpointSelector": CheckpointSelector,
    "ModelSelector": ModelSelector,
    "UnsafeModelSelector": UnsafeModelSelector,
    "LoraSelector": LoraSelector,
    "MultiLoraJoiner": MultiLoraJoiner,
    "CustomTimesteps": CustomTimesteps,
    "CustomSigmas": CustomSigmas,

    "FlowMatchScheduler": nodes_scheduler.FlowMatchScheduler,
    "FlowMatchSchedulerConfig": nodes_scheduler.FlowMatchSchedulerConfig,
    "DDIMScheduler": ddim_scheduler.DDIMScheduler,
    "DDPMScheduler": ddpm_scheduler.DDPMScheduler,
    "DEISMultistepScheduler": deis_multistep_scheduler.DEISMultistepScheduler,
    "DPMSolverMultistepScheduler": dpmsolver_multistep_scheduler.DPMSolverMultistepScheduler,
    "DPMSolverSinglestepScheduler": dpmsolver_singlestep_scheduler.DPMSolverSinglestepScheduler,
    "EulerAncestralDiscreteScheduler": euler_ancestral_discrete_scheduler.EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": euler_discrete_scheduler.EulerDiscreteScheduler,
    "HeunDiscreteScheduler": heun_discrete_scheduler.HeunDiscreteScheduler,
    "KDPM2AncestralDiscreteScheduler": kdpm2_ancestral_discrete_scheduler.KDPM2AncestralDiscreteScheduler,
    "KDPM2DiscreteScheduler": kdpm2_discrete_scheduler.KDPM2DiscreteScheduler,
    "LMSDiscreteScheduler": lms_discrete_scheduler.LMSDiscreteScheduler,
    "PNDMScheduler": pndm_scheduler.PNDMScheduler,
    "TCDScheduler": tcd_scheduler.TCDScheduler,
    "UniPCMultistepScheduler": unipc_multistep_scheduler.UniPCMultistepScheduler,

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
    "SDUpscaleSampler": SDUpscaleSampler,
    "SVDSampler": SVDSampler,
    "WanSampler": WanSampler,
    "ZImageSampler": ZImageSampler,
    "Krea2Sampler": Krea2Sampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSelector": "CheckpointSelector",
    "ModelSelector": "ModelSelector",
    "UnsafeModelSelector": "UnsafeModelSelector",
    "LoraSelector": "LoraSelector",
    "MultiLoraJoiner": "MultiLoraJoiner",
    "CustomTimesteps": "CustomTimesteps",
    "CustomSigmas": "CustomSigmas",

    "FlowMatchScheduler": "FlowMatchScheduler",
    "FlowMatchSchedulerConfig": "FlowMatchSchedulerConfig",
    "DDIMScheduler": "DDIMScheduler",
    "DDPMScheduler": "DDPMScheduler",
    "DEISMultistepScheduler": "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler": "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler": "DPMSolverSinglestepScheduler",
    "EulerAncestralDiscreteScheduler": "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler": "EulerDiscreteScheduler",
    "HeunDiscreteScheduler": "HeunDiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler": "KDPM2AncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler": "KDPM2DiscreteScheduler",
    "LMSDiscreteScheduler": "LMSDiscreteScheduler",
    "PNDMScheduler": "PNDMScheduler",
    "TCDScheduler": "TCDScheduler",
    "UniPCMultistepScheduler": "UniPCMultistepScheduler",

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
    "SDUpscaleSampler": "SDUpscaleSampler",
    "SVDSampler": "SVDSampler",
    "WanSampler": "WanSampler",
    "ZImageSampler": "ZImageSampler",
    "Krea2Sampler": "Krea2Sampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
