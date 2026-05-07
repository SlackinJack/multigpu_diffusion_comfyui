# multigpu_diffusion_comfyui

ComfyUI nodes to run [multigpu_diffusion](https://github.com/SlackinJack/multigpu_diffusion).
(Basically, a ComfyUI front-end to Diffusers, with support for multi-GPU environments.)


## Notes:
- Does not (and will never) use native sampling.
- Windows and macOS are (probably) not supported.


## Usage:
1. If applicable, activate your ComfyUI venv.
2. `cd` to `custom_nodes/multigpu_diffusion_comfyui`.
3. (Review and) run `setup.sh`.


## Test Environment:
- ComfyUI 0.9.1
- 4x Nvidia Tesla T4
- Ubuntu Server 26.04
- Python 3.14.4
