# ALL OWN CODE,  USING LIBRARIES AND ADAPTING FROM https://cloud.google.com/sdk/docs/
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sized

import pydantic


def _check_size2(field_name: str, value: Optional[Sized]) -> Optional[Sized]:
    if value is not None and len(value) != 2:
        raise ValueError(f"Value for field '{field_name}' must be of size 2, but instead got {value}")
    return value


@pydantic.dataclasses.dataclass
class VisionNerfRunConfig:
    img_hw: Optional[List[int]] = None
    chunk_size: Optional[int] = 2048
    mlp_block_num: Optional[int] = 6
    white_bkgd: Optional[bool] = True
    im_feat_dim: Optional[int] = 512
    pose_index: Optional[int] = 0

    @pydantic.validator("img_hw")
    def image_hw_validate(cls, v):
        return _check_size2("img_hw", v)
    
    def to_dict(self, experiment_name: str, mount_prefix: Path) -> Dict[str, Any]:
        return {
            "expname": experiment_name,
            "ckptdir": mount_prefix / "weights/",
            "ckpt_path": mount_prefix / "weights" / "srn_cars_500000.pth",
            "data_path": mount_prefix / "data/",
            "outdir": mount_prefix / "results/",
            "img_hw": self.img_hw or [128, 128],
            "chunk_size": self.chunk_size,
            "mlp_block_num": self.mlp_block_num,
            "white_bkgd": self.white_bkgd,
            "im_feat_dim": self.im_feat_dim,
            "data_range": [0, 1],
            "pose_index": self.pose_index
        }


@pydantic.dataclasses.dataclass
class NvDiffrecRunConfig:
    random_textures: Optional[bool] = True
    iter: Optional[int] = 500
    save_interval: Optional[int] = 100
    texture_res: Optional[List[int]] = None
    train_res: Optional[List[int]] = None
    batch: Optional[int] = 8
    learning_rate: Optional[List[float]] = None
    ks_min: Optional[List[float]] = None
    dmtet_grid: Optional[int] = 128
    mesh_scale: Optional[float] = 2.3
    laplace_scale: Optional[int] = 3000
    display: Optional[List[Dict[str, Any]]] = None
    layers: Optional[int] = 8
    background: Optional[str] = "white"

    @pydantic.validator("texture_res")
    def texture_res_validate(cls, v):
        return _check_size2("texture_res", v)

    @pydantic.validator("train_res")
    def train_res_validate(cls, v):
        return _check_size2("train_res", v)
    
    @pydantic.validator("learning_rate")
    def learning_rate_validate(cls, v):
        return _check_size2("learning_rate", v)

    def to_dict(self, experiment_name: str) -> Dict[str, Any]:
        return {
            "ref_mesh": str(Path("/mnt") / "visionnerf" / "results" / experiment_name),
            "random_textures": self.random_textures,
            "iter": self.iter,
            "save_interval": self.save_interval,
            "texture_res": self.texture_res or [ 128, 128 ],
            "train_res": self.train_res or [128, 128],
            "batch": self.batch,
            "learning_rate": self.learning_rate or [0.03, 0.01],
            "ks_min" : self.ks_min or [0, 0.25, 0.0],
            "dmtet_grid" : self.dmtet_grid,
            "mesh_scale" : self.mesh_scale,
            "laplace_scale" : self.laplace_scale,
            "display": self.display or [{"latlong" : True}, {"bsdf" : "kd"}, {"bsdf" : "ks"}, {"bsdf" : "normal"}],
            "layers" : self.layers,
            "background" : self.background,
            "out_dir": ""
        }


@pydantic.dataclasses.dataclass
class PipelineRequest:
    runId: str
    inputDataBucket: str
    inputDataPrefix: str
    outputDataBucket: str
    visionnerfWeights: str
    visionnerfRunConfig: Optional[VisionNerfRunConfig] = None
    nvdiffrecRunConfig: Optional[NvDiffrecRunConfig] = None

    @pydantic.validator("inputDataPrefix")
    def prefix_validate(cls, v):
        if not v.endswith("/"):
            raise ValueError(f"inputDataPrefix '{v}' should end with forward slash '/'.")
        return v

    @pydantic.validator("visionnerfWeights")
    def visionnerf_weights_validate(cls, v):
        if not v.endswith(".pth"):
            raise ValueError(f"Vision nerf weights '{v}' should end with a .pth file extension")
        return v

    def visionnerf_run_config(self) -> VisionNerfRunConfig:
        return self.visionnerfRunConfig or VisionNerfRunConfig()
    
    def nvdiffrec_run_config(self) -> NvDiffrecRunConfig:
        return self.nvdiffrecRunConfig or NvDiffrecRunConfig()
