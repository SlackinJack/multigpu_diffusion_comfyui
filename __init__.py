from .nodes.nodes_general import *
from .nodes.nodes_asyncdiff import *
from .nodes.nodes_distrifuser import *
from .nodes.nodes_xdit import *
from .nodes.nodes_xdit_usp import *


NODE_CLASS_MAPPINGS = {
    "GGUFSelector":                 GGUFSelector,
    "ModelSelector":                ModelSelector,
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

    "PipelineConfig":               PipelineConfig,

    "AsyncDiffPipelineConfig":      AsyncDiffPipelineConfig,
    "AsyncDiffADSampler":           AsyncDiffADSampler,
    "AsyncDiffSDSampler":           AsyncDiffSDSampler,
    "AsyncDiffSDUpscaleSampler":    AsyncDiffSDUpscaleSampler,
    "AsyncDiffSVDSampler":          AsyncDiffSVDSampler,

    "DistrifuserPipelineConfig":    DistrifuserPipelineConfig,
    "DistrifuserSDSampler":         DistrifuserSDSampler,

    "xDiTPipelineConfig":           xDiTPipelineConfig,
    "xDiTSampler":             xDiTSampler,
    "xDiTUSPPipelineConfig":        xDiTUSPPipelineConfig,
    "xDiTUSPImageSampler":          xDiTUSPImageSampler,
    "xDiTUSPVideoSampler":          xDiTUSPVideoSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFSelector":                 "GGUFSelector",
    "ModelSelector":                "ModelSelector",
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

    "PipelineConfig":               "PipelineConfig",

    "AsyncDiffPipelineConfig":      "AsyncDiffPipelineConfig",
    "AsyncDiffADSampler":           "AsyncDiffADSampler",
    "AsyncDiffSDSampler":           "AsyncDiffSDSampler",
    "AsyncDiffSDUpscaleSampler":    "AsyncDiffSDUpscaleSampler",
    "AsyncDiffSVDSampler":          "AsyncDiffSVDSampler",

    "DistrifuserPipelineConfig":    "DistrifuserPipelineConfig",
    "DistrifuserSDSampler":         "DistrifuserSDSampler",

    "xDiTPipelineConfig":           "xDiTPipelineConfig",
    "xDiTSampler":                  "xDiTSampler",
    "xDiTUSPPipelineConfig":        "xDiTUSPPipelineConfig",
    "xDiTUSPImageSampler":          "xDiTUSPImageSampler",
    "xDiTUSPVideoSampler":          "xDiTUSPVideoSampler",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
