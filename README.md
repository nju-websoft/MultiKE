### Purpose of the code

The purpose consists out of the following goals:
- Transform the original implementation of the _MultiKE_ method from the papaer
written in `tensorflow 1`: reimplement the approach in `pytorch`

- Modify the initial approach of using a _TransE_ score with 2 scores from _MDE_

- Starting two versions of the code (unmodified in `tensor`, reimplemented end edited with `MDE`)
provides the differences.

#### Links to the articles: 

- MultiKE: https://www.ijcai.org/Proceedings/2019/0754.pdf
  - Git repository: https://github.com/nju-websoft/MultiKE

- MDE: https://ecai2020.eu/papers/1271_paper.pdf
  - repository: https://github.com/mlwin-de/MDE_adv

### How to run code?

#### Requirements

The required tools are:
- tensorflow 1.x
- pytorch
- python3.7

Setting up the environment to run the code:

- Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
- Create an environment for python3.7
  - run `conda create -n myenv python=3.7`
  - choose the environment `conda activate myenv`
- Install tensorflow 1.x : `conda install -c conda-forge tensorflow=1.14`
- Install pytorch:
  - Go to the link :https://pytorch.org/get-started/locally/
  - Set up the parameters interactively. E.g. for Linux it is `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

Download the following archive, extract and place it somewhere *next to / in* the repository directory - you will need it later.

- Filename to download : `wiki-news-300d-1M.vec.zip`
- Link: https://fasttext.cc/docs/en/english-vectors.html

You're set up.

#### Running

**Important instructions about the set up:**

The configuration of the run is provided in the `args.json` file.
Please, open it and edit accordingly:
- The file location paths
```json
  "training_data": "path/to/training/data",
  "output": "path/to/output/results/",
  "word2vec_path": "/path/to/wiki-news-300d-1M.vec",
  "dataset_division": "631/",
```
  - `dataset_division` path is relative to the `training_data`
  - by default, training data is contained in `BootEA_DBP_WD_100K`. This lies in `/data` directory of this and the original repository.
  - You should've downloaded `wiki-news-300d-1M.vec` by now. Please specifiy the path to it in `word2vec_path`
  - The original data is very big and most computers cannot handle it due to insufficient RAM resource. To run with smaller dataset, address the
  corresponding section of this readme below.

**General instructions:**

- Running original `tensorflow` version: please read the original readme's.
Also placed in the same directory as this file: `ORIGINAL_PAPER_README.md`
Recommended to use the original repository without any modification done in this lab.

- Running the modified with pytorch code:
```
python code/main.py -m ITC -d mock_data > log.txt
```

after run, see the log in the created `log.txt`

**Running with smaller dataset**

- The repository contains the directory `/mock_data` - it can be specified as `training_data` field in `args.json` file.
- please, adjust the following fields of learning rates in `args.json` (due to the significantly smaller dataset):
  ```
    "learning_rate": 0.1,
    "relation_learning_rate": 0.1,
    "ITC_learning_rate": 0.1
  ```
- then run as described above.
- note that it's possible to generate other 'small' datasets with scripts provided in the same directory. You can address the README in this directory.
  - If the RAM is still not sufficient, you need to regenerate the dataset (perhaps even a couple of times, as it runs with randomizations).
  - for different datasets different learning rates (`args.json`) work out differently.

### Scenarios of runs
#### Ideal scenario

You have sufficient RAM resource, you run both pytorch and tensorflow versions and see that pytorch behaves
comparably - losses do not differ much, matched entities/relations/attributes do not differ much.

#### Smaller dataset

You see that the decreasing loss in the log.
Running tensorflow version also provides the same loss decrease.

Depending on the `args.json`, matching results can differ.

### Log file contains

Several sections:
#### Literal encoder training
```
epoch 1 of literal encoder, loss: 238166.6931, time: 10.3270s
epoch 2 of literal encoder, loss: 146460.3867, time: 10.1467s
```
#### Alignment training:
```
epoch 1:
epoch 1 of rv, avg. loss: 37973.5807, time: 47.1617s
epoch 1 of ckgrtv, avg. loss: 6951.0853, time: 6.1715s
epoch 1 of av, avg. loss: 1050.8782, time: 65.9305s
epoch 1 of ckgatv, avg. loss: 6932.3308, time: 19.2376s
epoch 1 of cnv, avg. loss: 4990.2788, time: 1.6387s
epoch 2:
epoch 2 of rv, avg. loss: 37985.6638, time: 44.2177s
epoch 2 of ckgrtv, avg. loss: 6951.1517, time: 6.1280s
epoch 2 of av, avg. loss: 1050.8535, time: 65.3677s
epoch 2 of ckgatv, avg. loss: 6932.3688, time: 19.1464s
epoch 2 of cnv, avg. loss: 4983.1904, time: 1.6418s
epoch 3:
```

Loss should decrease over time
