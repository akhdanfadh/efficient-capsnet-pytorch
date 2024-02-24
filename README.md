# Efficient-CapsNet on Pytorch

[Efficient-CapsNet](https://www.nature.com/articles/s41598-021-93977-0) (Mazzia et al., 2021) is a novel architecture for Capsule Networks that improves the routing algorithm and reduces the number of parameters.

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-93977-0/MediaObjects/41598_2021_93977_Fig2_HTML.png?as=webp" width=75%> <br>
  Schematic representation of the overall architecture of Efficient-CapsNet.
</p>

This project is a faithful PyTorch implementation of the paper with additional features such as logging and monitoring with tensorboard, and a configuration file for easy hyperparameter tuning.
The code is based on authors' original implementation in Tensorflow [here](https://github.com/EscVM/Efficient-CapsNet), and has been tested to match it numerically.

## Installation

<details><summary>Python 3 dependencies</summary>

- pyyaml
- torch
- torchvision
- opencv-python
- pandas
- tensorboard
</details>

We recommend using a virtual environment to install the required packages, such as `conda`.
```bash
git clone git@github.com:akhdanfadh/efficient-capsnet-pytorch.git
cd efficient-capsnet-pytorch

conda create -n efficient-capsnet python=3.10
conda activate efficient-capsnet
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorboard pyyaml pandas opencv
```


This project has been tested on WSL Ubuntu 22.04 with PyTorch 2.2 & CUDA 12.2 on a 3090.


## Usage

Modify the `config.yaml` file to your needs and run the training script as follows. Please check the config file for the available options.
```bash
python train.py -c config.yaml -i run_id
```

For monitoring the training process, we use tensorboard. To start tensorboard, run the following command after training or in a separate terminal for live-monitoring.
```bash
tensorboard --logdir=saved
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Code structure and training loop are based on the [pytorch-template](https://github.com/victoresque/pytorch-template) repository with lots of adjustments.
The repo help us to understand organizing a deep learning project thoroughly.

## Citation

Kudos to the authors of the paper for their amazing work. If you find this code useful, please consider citing the original work:
```
@article{mazzia2021efficient,
    title={Efficient-CapsNet: capsule network with self-attention routing},
    author={Mazzia, Vittorio and Salvetti, Francesco and Chiaberge, Marcello},
    year={2021},
    journal={Scientific reports},
    publisher={Nature Publishing Group},
    volume={11}
}
```
