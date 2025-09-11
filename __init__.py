from .nodes.nodes_general import *
from .nodes.nodes_asyncdiff import *
from .nodes.nodes_distrifuser import *
from .nodes.nodes_xdit import *
from .nodes.nodes_xdit_usp import *


NODE_CLASS_MAPPINGS = {
    "GGUFSelector":                 GGUFSelector,
    "CheckpointSelector":           CheckpointSelector,
    "SchedulerSelector":            SchedulerSelector,
    "LoraSelector":                 LoraSelector,
    "MultiLoraJoiner":              MultiLoraJoiner,
    "VAESelector":                  VAESelector,
    "MotionModuleSelector":         MotionModuleSelector,
    "MotionAdapterSelector":        MotionAdapterSelector,
    "MotionAdapterLoraSelector":    MotionAdapterLoraSelector,
    "ControlNetSelector":           ControlNetSelector,
    "IPAdapterSelector":            IPAdapterSelector,
    "EncodePromptWithCompel":       EncodePromptWithCompel,

    "HostConfig":                   HostConfig,

    "AsyncDiffConfig":              AsyncDiffConfig,
    "AsyncDiffADSampler":           AsyncDiffADSampler,
    "AsyncDiffSDSampler":           AsyncDiffSDSampler,
    "AsyncDiffSDUpscaleSampler":    AsyncDiffSDUpscaleSampler,
    "AsyncDiffSVDSampler":          AsyncDiffSVDSampler,

    "DistrifuserConfig":            DistrifuserConfig,
    "DistrifuserSDSampler":         DistrifuserSDSampler,

    "xDiTConfig":                   xDiTConfig,
    "xDiTSampler":                  xDiTSampler,
    "xDiTUSPConfig":                xDiTUSPConfig,
    "xDiTUSPImageSampler":          xDiTUSPImageSampler,
    "xDiTUSPVideoSampler":          xDiTUSPVideoSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFSelector":                 "GGUFSelector",
    "CheckpointSelector":           "CheckpointSelector",
    "SchedulerSelector":            "SchedulerSelector",
    "LoraSelector":                 "LoraSelector",
    "MultiLoraJoiner":              "MultiLoraJoiner",
    "VAESelector":                  "VAESelector",
    "MotionModuleSelector":         "MotionModuleSelector",
    "MotionAdapterSelector":        "MotionAdapterSelector",
    "MotionAdapterLoraSelector":    "MotionAdapterLoraSelector",
    "ControlNetSelector":           "ControlNetSelector",
    "IPAdapterSelector":            "IPAdapterSelector",
    "EncodePromptWithCompel":       "EncodePromptWithCompel",

    "HostConfig":                   "HostConfig",

    "AsyncDiffConfig":              "AsyncDiffConfig",
    "AsyncDiffADSampler":           "AsyncDiffADSampler",
    "AsyncDiffSDSampler":           "AsyncDiffSDSampler",
    "AsyncDiffSDUpscaleSampler":    "AsyncDiffSDUpscaleSampler",
    "AsyncDiffSVDSampler":          "AsyncDiffSVDSampler",

    "DistrifuserConfig":            "DistrifuserConfig",
    "DistrifuserSDSampler":         "DistrifuserSDSampler",

    "xDiTConfig":                   "xDiTConfig",
    "xDiTSampler":                  "xDiTSampler",
    "xDiTUSPConfig":                "xDiTUSPConfig",
    "xDiTUSPImageSampler":          "xDiTUSPImageSampler",
    "xDiTUSPVideoSampler":          "xDiTUSPVideoSampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
