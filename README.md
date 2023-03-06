<p align="center">
<img src="./vall-e.png" width="500px"></img>
</p>

# VALL-E

An unofficial PyTorch implementation of [VALL-E](https://valle-demo.github.io/) by [enhuiz](https://github.com/enhuiz/vall-e), based on the [EnCodec](https://github.com/facebookresearch/encodec) tokenizer.


### Requirements

Since the trainer is based on [DeepSpeed](https://github.com/microsoft/DeepSpeed#requirements), you will need to have a GPU that DeepSpeed has developed and tested against, as well as a CUDA or ROCm compiler pre-installed to install this package.

### Install

```
git clone https://github.com/liuhaozhe6788/vall-e.git
```

Note that the code is only tested under `Python 3.9.0` `CUDA11.1`.

### Train

1. Put your data into a folder, e.g. `data/your_data`. Audio files should be named with the suffix `.wav` and text files with `.normalized.txt`.

2. Quantize the data:

```
python run_qnt.py data/your_data
```

3. Generate phonemes based on the text:

```
python run_g2p.py  data/your_data
```

4. Customize your configuration by creating `config/your_data/ar.yml` and `config/your_data/nar.yml`. Refer to the example configs in `config/test` and `vall_e/config.py` for details. You may choose different model presets, check `vall_e/vall_e/__init__.py`.

5. Train the AR or NAR model using the following scripts:

```
python run_train.py yaml=config/your_data/ar_or_nar.yml
```

You may quit your training any time by just typing `quit` in your CLI. The latest checkpoint will be automatically saved.

### Export

Both trained models need to be exported to a certain path. To export either of them, run:

```
python run_export.py zoo/ar_or_nar.pt yaml=config/your_data/ar_or_nar.yml
```

This will export the latest checkpoint.

### Synthesis

```
python run.py <text> <ref_path> <out_path> --ar-ckpt zoo/ar.pt --nar-ckpt zoo/nar.pt
```

### Open tensorboard

```
tensorboard --logdir logs/your_data --host localhost --port 8088
```

## TODO

- [x] AR model for the first quantizer
- [x] Audio decoding from tokens
- [x] NAR model for the rest quantizers
- [x] Trainers for both models
- [x] Implement AdaLN for NAR model.
- [x] Sample-wise quantization level sampling for NAR training.
- [ ] Pre-trained checkpoint and demos on LibriTTS
- [x] Synthesis CLI
- [x] Plot train and validation loss

## Notice

- [EnCodec](https://github.com/facebookresearch/encodec) is licensed under CC-BY-NC 4.0. If you use the code to generate audio quantization or perform decoding, it is important to adhere to the terms of their license.

## Citations

```bibtex
@article{wang2023neural,
  title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```
