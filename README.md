# EMG Keystroke Decoding - UCLA ECE C147/C247 Final Project

This repository contains our final project for UCLA ECE C147/C247 (Neural Networks 
& Deep Learning, Winter 2026). We investigate the decoding of QWERTY keystrokes 
from surface electromyography (sEMG) signals using the `emg2qwerty` framework 
released by Meta Reality Labs. All experiments are conducted on a single subject 
(ID: `#89335547`) from the `emg2qwerty` dataset.

---

## Project Overview

Surface EMG signals recorded from both wrists provide a non-invasive window into 
finger movement intent during typing. The core task is to decode a sequence of 
QWERTY keystrokes from 32-channel sEMG recordings using deep learning models 
trained with Connectionist Temporal Classification (CTC) loss, evaluated via 
Character Error Rate (CER).

We systematically compare several neural architectures and conduct ablation studies 
to understand what drives decoding performance:

| Model | Description |
|---|---|
| TDS Conv (Baseline) | Provided Time-Depth Separable convolutional baseline |
| Vanilla RNN | Simple unidirectional recurrent network |
| BiLSTM | Bidirectional LSTM encoder |
| Transformer | Multi-head self-attention encoder |
| CNN + BiLSTM | Proposed hybrid convolutional-recurrent architecture |

Our best standalone sequential model is the **BiLSTM**, and our best overall model 
is the **CNN + BiLSTM** configuration selected after systematic architecture and 
hyperparameter tuning.

---

## Repository Structure
```text
.
├── config/                          # Hydra and model configuration YAML files
├── emg2qwerty/                      # Modified project source code
│   ├── lightning.py                 # PyTorch Lightning modules
│   ├── modules.py                   # Model architecture definitions
│   ├── data.py                      # Dataset and data loading utilities
│   └── transforms.py                # Data transforms and augmentation
├── figures/                         # Saved plots and experiment figures
├── models/                          # Model definitions or saved checkpoints
├── scripts/                         # Helper scripts for running experiments
│
├── Colab_setup.ipynb                # Environment setup and baseline walkthrough
│
│── Baseline and Architecture Experiments
├── tds_conv.ipynb                   # TDS Conv baseline (150 epochs)
├── tds_conv_50_epochs.ipynb         # Shorter TDS Conv baseline run (50 epochs)
├── rnn.ipynb                        # Vanilla RNN experiment
├── bilstm.ipynb                     # Bidirectional LSTM experiment
├── transformer.ipynb                # Transformer encoder experiment
│
│── CNN-BiLSTM Ablation Studies
├── hidden_size_lstm_depth.ipynb     # BiLSTM hidden size and layer depth study
├── cnn_depth.ipynb                  # CNN depth ablation (1, 2, 3 layers)
├── kernel_size.ipynb                # Convolutional kernel size ablation
│
│── Optimization and Augmentation
├── dropout_lr_schedule.ipynb        # Grid search over dropout, LR, and scheduler
├── 250_epoch_model.ipynb            # Extended training with augmentation
│
│── Final Model
├── final_model.ipynb                # Final CNN + BiLSTM model (150 epochs)
│
│── Analysis Notebooks
├── bilstm_analysis.ipynb            # Metrics and plots for BiLSTM
├── cnn_bilstm_analysis.ipynb        # Metrics and plots for CNN + BiLSTM
├── final_model_analysis.ipynb       # Summary metrics and plots for final model
├── 250_epoch_model_analysis.ipynb   # Analysis for extended 250-epoch run
│
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Python package dependencies
└── README.md
```
---
## Notebook Descriptions

### Baseline and Architecture Comparison

| Notebook | Description |
|---|---|
| `tds_conv.ipynb` | Trains and evaluates the provided TDS Conv baseline for 150 epochs. Establishes reference validation and test CER values used for all comparisons. |
| `tds_conv_50_epochs.ipynb` | Shortened 50-epoch run of the TDS Conv model used for fast comparisons during hyperparameter sweeps. |
| `rnn.ipynb` | Trains a 3-layer unidirectional RNN encoder with hidden size 512. Establishes a lower-bound recurrent baseline. |
| `bilstm.ipynb` | Trains a 3-layer BiLSTM encoder with hidden size 512. Best-performing standalone sequential model. |
| `transformer.ipynb` | Trains a Transformer encoder with multi-head self-attention. Explores global temporal context for sEMG decoding. |

### CNN-BiLSTM Ablation Studies

| Notebook | Description |
|---|---|
| `hidden_size_lstm_depth.ipynb` | Evaluates BiLSTM hidden sizes of 128, 256, and 512 to identify the effect of recurrent capacity on CER. |
| `cnn_depth.ipynb` | Compares CNN frontend depths of 1, 2, and 3 convolutional blocks to assess the benefit of deeper spatial feature hierarchies. |
| `kernel_size.ipynb` | Compares convolutional kernel sizes of 3, 5, 7, and 9 to study the effect of temporal receptive field size on decoding performance. |

### Optimization and Augmentation

| Notebook | Description |
|---|---|
| `dropout_lr_schedule.ipynb` | Full grid search (27 configurations) over dropout (0, 0.2, 0.4), learning rate (1e-3, 5e-4, 1e-4), and LR scheduler (none, cosine annealing, reduce-on-plateau). All runs trained for 50 epochs. |
| `250_epoch_model.ipynb` | Extended 250-epoch training run using the best architecture and hyperparameter configuration, with additional data augmentation strategies including Gaussian noise injection, amplitude scaling, channel dropout, and temporal masking. |

### Final Model

| Notebook | Description |
|---|---|
| `final_model.ipynb` | Trains the final selected CNN + BiLSTM configuration for 150 epochs using the optimal architecture and hyperparameters identified through ablation and tuning. This is the primary model for final evaluation. |

### Analysis Notebooks

| Notebook | Description |
|---|---|
| `bilstm_analysis.ipynb` | Generates training curves, CER breakdowns, and error analysis plots for the BiLSTM model. |
| `cnn_bilstm_analysis.ipynb` | Generates metrics and visualizations for the CNN + BiLSTM model across ablation experiments. |
| `final_model_analysis.ipynb` | Produces all summary metrics and figures reported in the final paper for the selected model. |
| `250_epoch_model_analysis.ipynb` | Analysis and plots for the extended 250-epoch training experiment. |

---

## Guiding Tips + FAQs
_Last updated 2/13/2025_
- Read through the Project Guidelines to ensure that you have a clear understanding of what we expect
- Familiarize yourself with the prediction task and get a high-level understanding of their base architecture (it would be beneficial to read about CTC loss)
- Get comfortable with the codebase
  - ```lightning.py``` + ```modules.py``` - where most of your model architecture development will take place
  - ```data.py``` - defines PyTorch dataset (likely will not need to touch this much)
  - ```transforms.py``` - implement more data transforms and other preprocessing techniques
  - ```config/*.yaml``` - modify model hyperparameters and PyTorch Lightning training configuration
    - **Q: How do we update these configuration files?** A: Note the structure of YAML files include basic key-value pairs (i.e. ```<key>: <value>```) and hierarchical structure. So, for instance, if we wanted to update the ```mlp_features``` hyperparameter of the ```TDSConvCTCModule```, we would change the value at line 5 of ```config/model/tds_conv_ctc.yaml``` (under ```module```). _Read more details [here](https://pytorch-lightning.readthedocs.io/en/1.3.8/common/lightning_cli.html)._
    - **Q: Where do we configure data splitting?** A: Refer to ```config/user/single_user.yaml```. Be careful with your edits, so that you don't accidentally move the test data into your training set.

# emg2qwerty
[ [`Paper`](https://arxiv.org/abs/2410.20081) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2qwerty) ]

A dataset of surface electromyography (sEMG) recordings while touch typing on a QWERTY keyboard with ground-truth, benchmarks and baselines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/71a9f361-7685-4188-83c3-099a009b6b81" height="80%" width="80%" alt="alt="sEMG recording" >
</p>

## Setup

### 1. Clone the Repository
```shell
git clone https://github.com//emg2qwerty.git ~/emg2qwerty
cd ~/emg2qwerty
```

### 2. Create and Activate the Conda Environment
```shell
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .
```

### 3. Download and Link the Dataset
```shell
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

### 4. Google Colab Users

Open `Colab_setup.ipynb` for a step-by-step walkthrough of environment setup and 
baseline model training within a Colab environment, including GPU configuration 
and Google Drive integration.

---

## Data

The dataset consists of 1,136 files in total - 1,135 session files spanning 108 users and 346 hours of recording, and one `metadata.csv` file. Each session file is in a simple HDF5 format and includes the left and right sEMG signal data, prompted text, keylogger ground-truth, and their corresponding timestamps. `emg2qwerty.data.EMGSessionData` offers a programmatic read-only interface into the HDF5 session files.

To load the `metadata.csv` file and print dataset statistics,

```shell
python scripts/print_dataset_stats.py
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012947-66cab4c4-963c-4f1a-af12-47fea1681f09.png" alt="Dataset statistics" height="50%" width="50%">
</p>

To re-generate data splits,

```shell
python scripts/generate_splits.py
```

The following figure visualizes the dataset splits for training, validation and testing of generic and personalized user models. Refer to the paper for details of the benchmark setup and data splits.

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012465-504eccbf-8eac-4432-b8aa-0e453ad85b49.png" alt="Data splits">
</p>

To re-format data in [EEG BIDS format](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html),

```shell
python scripts/convert_to_bids.py
```

## Training

Generic user model:

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun
```

Personalized user models:

```shell
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1
```

If you are using a Slurm cluster, include "cluster=slurm" override in the argument list of above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

## Testing

Greedy decoding:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Beam-search decoding with 6-gram character-level language model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

The 6-gram character-level language model, used by the first-pass beam-search decoder above, is generated from [WikiText-103 raw dataset](https://huggingface.co/datasets/wikitext), and built using [KenLM](https://github.com/kpu/kenlm). The LM is available under `models/lm/`, both in the binary format, and the human-readable [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). These can be regenerated as follows:

1. Build kenlm from source: <https://github.com/kpu/kenlm#compiling>
2. Run `./scripts/lm/build_char_lm.sh <ngram_order>`

## License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

## Citing emg2qwerty

```
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```
