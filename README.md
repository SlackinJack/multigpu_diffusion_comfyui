# multigpu_diffusion_comfyui

ComfyUI nodes to run [multigpu_diffusion](https://github.com/SlackinJack/multigpu_diffusion).
(Basically, a ComfyUI front-end to Diffusers, with support for multi-GPU environments.)


## Notes:
- Requires **Diffusers-format** checkpoints/models.
- Does not (and will never) use native sampling.
- Windows and macOS are not (and probably never will be) supported.
- **This node uses psutil/subprocess to manage the host scripts.** You can find the calls in `modules/host_manager.py`.


## Setup:
1. `cd` to `custom_nodes/multigpu_diffusion_comfyui`.
2. (Review and) run `setup.sh`.


## Debugging:
- If you run into issues, you can manually curl to the hosts (e.g., `curl localhost:{port}/close` to manually shut down the host).


## Test Environment:
- ComfyUI 0.9.1
- 4x Nvidia Tesla T4
- Ubuntu Server 26.04
- Python 3.14.4
