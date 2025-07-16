# VFIMAMBA RECURSIVE

Coming Soon

## Instalation

```bash
pip install torch torchvision torchaudio
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
pip install opencv-python imageio timm

```

## :sunglasses: Play with Demos

1. Download the [model checkpoints](https://huggingface.co/MCG-NJU/VFIMamba_ckpts/tree/main) and put the `ckpt` folder into the root dir. We also support directly importing model weights from HuggingFace. Please refer to hf_demo_2x.py.
2. Run the following commands to generate 2x and Nx (arbitrary) frame interpolation demos:

We provide two models, an efficient version (VFIMamba-S) and a stronger one (VFIMamba).
You can choose what you need by changing the parameter `model`.

### Manually Load

```shell
python inference_img.py --n 8 --input imgs/pth --model **model[VFIMamba_S/VFIMamba]** # for 8x interpolation
```

Images should be named in numerical order: 0.png 8.png etc (see example folder)

## :runner: Evaluation

1. Download the dataset you need:

   - [Vimeo90K dataset](http://toflow.csail.mit.edu/)
   - [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow)
   - [Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py)
   - [SNU-FILM dataset](https://myungsub.github.io/CAIN/)
   - [X4K1000FPS dataset](https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0)

2. Download the [model checkpoints](https://huggingface.co/MCG-NJU/VFIMamba_ckpts/tree/main) and put the `ckpt` folder into the root dir. We also support directly importing model weights from HuggingFace. Please refer to hf_demo_2x.py.

For all benchmarks:

```shell
python benchmark/**dataset**.py --model **model[VFIMamba_S/VFIMamba]** --path /where/is/your/**dataset**
```

You can also test the inference time of our methods on the $H\times W$ image with the following command:

```shell
python benchmark/TimeTest.py --model **model[VFIMamba_S/VFIMamba]** --H **SIZE** --W **SIZE**
```

## :muscle: Citation

If you think this project is helpful in your research or for application, please feel free to leave a star⭐️ and cite our paper:

```
@misc{zhang2024vfimambavideoframeinterpolation,
      title={VFIMamba: Video Frame Interpolation with State Space Models},
      author={Guozhen Zhang and Chunxu Liu and Yutao Cui and Xiaotong Zhao and Kai Ma and Limin Wang},
      year={2024},
      eprint={2407.02315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.02315},
}
```

## :heartpulse: License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [EMA-VFI](https://github.com/whai362/PVT), [MambaIR](https://github.com/csguoh/MambaIR?tab=readme-ov-file#installation) and [SGM-VFI](https://github.com/MCG-NJU/SGM-VFI). Please also follow their licenses. Thanks for their awesome works.
