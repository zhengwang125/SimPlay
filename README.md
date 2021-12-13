# SimPlay

This is the implementation of "Similar Sports Play Retrieval with Deep Reinforcement Learning" (TKDE'2022).

## Requirements

* Linux Ubuntu OS (16.04 is tested)
* Python (3.6 is tested)
* [Tensorflow-GPU](https://www.tensorflow.org/install/gpu) (1.8.0 is tested)

Please refer to the source code to install the required packages such as matplotlib in Python. You can install packages with conda in a shell as

```bash
conda install matplotlib
```

## Dataset

The dataset is a real-world soccer player tracking data collected by STATS. Download the dataset by requesting [STATS Artificial Intelligence](https://www.stats.com/artificial-intelligence/) and put the compressed `*.kpl` files into `SoccerData`. Note that the intermediate results generated by the algorithm will also be saved in this folder.

### Data format

The training kpl file contains around 7500 sequences. In each sequence, it consists of the tracking data of three parts: 11 defense players, 11 attacking players and a ball. All of these have two fields horizontal and vertical coordinates obtained at a sampling frequency of 10Hz. If you want to test your own data, please also refer to this format. More details can be found in official data instruction.

## Running Procedures

### Preparing play2vec 

Please refer to the previous [repository](https://github.com/zhengwang125/play2vec) to prepare the play2vec model for the next procedures. 
It needs running the following files under the `play2vec` folder to obtain the model:
[`preprocess.py`](./play2vec/preprocess.py),[`viz.py`](./play2vec/viz.py),[`ogm.py`](./play2vec/ogm.py),[`corrupted_noise.py`](./play2vec/corrupted_noise.py),[`corrupted_drop.py`](./play2vec/corrupted_drop.py),[`building.py`](./play2vec/building.py),[`embedding.py`](./play2vec/embedding.py),[`dae.py`](./play2vec/dae.py) 
and obtain the vector representations of all plays by running [`estimate.py`](./play2vec/estimate.py). we provide the trained play2vec model in the `model_1` folder.
```bash
python preprocess.py #data visualization
python viz.py #data visualization
python ogm.py #maps the coordinates to grids
python corrupted_noise.py #adding noise
python corrupted_drop.py #adding drop
python building.py #building corpus
python embedding.py #token representation learning
python dae.py #play representation learning
python estimate.py #evaluation
```

### Running SimPlay Algorithms
Copy required pickles and the trained play2vec model produced in the last step into `SoccerData` folder under `algorithms` folder.
To preprocess the data for training and testing, running [`rl_preprocess.py`](./algorithms/rl_preprocess.py), and then the generated data will be stored under `subt_data` folder.
The implementations of ExactS, SizeS, (PSS, POS, POS-D), RLS and RLS-Skip are stored in [`exact.py`](./algorithms/exact.py),[`fixed_length.py`](./algorithms/fixed_length.py),
[`heuristic.py`](./algorithms/heuristic.py),[`rl_main.py`](./algorithms/rl_main.py) and [`rl_main_skip.py`](./algorithms/rl_main_skip.py), respectively. 
We provide the testing codes for these algorithms in [`subt_testing.py`](./algorithms/subt_testing.py). In addition, we provide the trained RLS and RLS-Skip models in the `model_collect` folder.
```bash
python rl_preprocess.py #rl data preprocessing
python exact.py #ExactS
python fixed_length.py #SizeS
python heuristic.py #opt = 'PSS', 'POS' or 'POS-D'
python rl_main.py #RLS, your model will be trained and saved in 'model_collect/test'
python rl_main_skip.py #RLS-Skip, your model will be trained and saved in 'model_collect/skip'
python subt_testing.py #evaluation
```


### Running Game Simplification
Copy the raw `*.kpl` files into `SoccerData` folder under `simplification` folder. To obtain required inputs (simplified versions) for SimPlay algorithms, running the following codes step by step:
[`ogm.py`](./simplification/ogm.py),[`corrupted_noise.py`](./simplification/corrupted_noise.py),[`corrupted_drop.py`](./simplification/corrupted_drop.py),
[`building.py`](./simplification/building.py),[`embedding.py`](./simplification/embedding.py),[`dae.py`](./simplification/dae.py), [`estimate.py`](./simplification/estimate.py).
We provide three simplification algorithms (i.e., Uniform, Top-Down and Bottom-Up), which are implemented in [`ogm.py`](./simplification/ogm.py).
Before running the game simplification, we provide the "map back" function in [`mapback.py`](./simplification/mapback.py), which maps back the simplified versions to the database of original games for evaluation.
Then, referring the last step (e.g., copy the required pickles and the trained play2vec/RL models) and running the following to see the simplified learning-based versions. 
```bash
python ogm.py #maps the coordinates to grids
python corrupted_noise.py #adding noise
python corrupted_drop.py #adding drop
python building.py #building corpus
python embedding.py #token representation learning
python dae.py #play representation learning
python estimate.py #evaluation
python mapback.py #mapback to original for evaluation
python rl_preprocess.py #data preprocessing for RL-based algorithms
python subt_testing.py #evaluation
```

### Running Index
Copy required data/models into the corresponding folders under `index` folder. To see the effect of without inedex, random index and learned index, 
it first needs to prepare the data that is used for training/evalution in [`prepare.py`](./index/prepare.py). The implementation of the metric learning-based (triplet networks) index is in [`train_index.py`](./index/train_index.py), and the best model will be saved in `model` folder. Then, evaluating different index structures by running the following codes:
[`without_index.py`](./index/without_index.py),[`random_index.py`](./index/random_index.py), [`learned_index.py`](./index/learned_index.py). We provide the learned index model in the `model` folder.
```bash
python prepare.py #data preprocessing for building index for SimPlay
python without_index.py #without index evaluation
python train_index.py #training for metric learning based index
python random_index.py #random index evaluation
python learned_index.py #learned index evaluation
```
