# OW-OVD Pest Detection

This repository is a customized OW-OVD workspace for open-world pest detection. It was pulled from the original author repository and adapted for local experiments, demo runs, ablation studies, and notebook-based evaluation.

The repo centers on a Gradio demo, MMYOLO-based configs, and a set of notebooks/scripts used to run and visualize open-world detection experiments on pest datasets such as IP102.

## Requirements

- Kaggle is the primary runtime environment for this repo.
- Python 3.10+ with GPU acceleration available in the notebook runtime.
- Internet access for the first run, or pre-uploaded wheels/packages in Kaggle Input if you want an offline-first setup.
- PyTorch 2.4.0 + CUDA 12.1, torchvision 0.19.0 + CUDA 12.1, MMCV, MMDetection, MMYOLO, MMEngine, Gradio, Pillow, NumPy, and the supporting evaluation packages used by the notebooks.

## Setup

1. Clone MMYOLO into `third_party/mmyolo`.

```bash
git clone https://github.com/open-mmlab/mmyolo.git third_party/mmyolo
```

2. Install the runtime dependencies used by the Kaggle notebooks.

```bash
pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -q matplotlib pycocotools terminaltables mmengine prettytable wcwidth open_clip_torch transformers
pip install -q "mmdet>=3.1.0" --no-deps
pip install -q --no-build-isolation --no-deps third_party/mmyolo
```

3. Patch the MMCV version ceiling if your notebook flow requires it.

```bash
python patch_versions.py
```
## Training
```bash
!PYTHONPATH=. torchrun --nproc_per_node=2 third_party/mmyolo/tools/train.py configs/custom/ip102_t1.py --launcher pytorch
```
## Testing
```bash
!PYTHONPATH=. python third_party/mmyolo/tools/test.py configs/custom/ip102_t1.py "best_model_path"
```
## Run the demo

Start the demo with the launcher or call the app directly.

```bash
run_demo.bat
```

## Notes

- The project is research-oriented, so some paths and scripts are tailored to the author's original setup.
- If you adapt the repo to another machine, update the Python path and checkpoint paths in the batch files.

## License

This repository inherits the licensing terms of the original upstream project and any included third-party components. Check the upstream source and bundled dependencies before redistribution.
