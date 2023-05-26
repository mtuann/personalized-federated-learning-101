# Benchmark for Personalized Federated Learning
This code is based on the setup for Personalized Federated Learning (pFL) from the repositories:
- ***pFL-Bench*** -- [pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench)
- ***FL-bench*** -- [Federated Learning Benchmark](https://github.com/KarhouTam/FL-bench)
- []()


## 1. Environment Preparation

1. Set up the environment by installing the dependencies from the `environment.yml` file:

```bash
make install
```
2. Download the data and pretrained models for MNIST, CIFAR-10, and Tiny-ImageNet from the [DBA Github repository](https://github.com/AI-secure/DBA).



## Method 🧬

### Regular FL Methods

### Personalized FL Methods
- ***pFedSim (My Work⭐)*** -- [pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning](https://arxiv.org/abs/2305.15706) (ArXiv'23)

- ***Local-Only*** -- Local training only (without communication).

- ***FedMD*** -- [FedMD: Heterogenous Federated Learning via Model Distillation](http://arxiv.org/abs/1910.03581) (NIPS'19)

- ***APFL*** -- [Adaptive Personalized Federated Learning](http://arxiv.org/abs/2003.13461) (ArXiv'20)

- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) (ArXiv'20)

- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization](http://arxiv.org/abs/2102.07623) (ICLR'21)

- ***FedPer*** -- [Federated Learning with Personalization Layers](http://arxiv.org/abs/1912.00818) (AISTATS'20)

- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078) (ICML'21)

- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) (NIPS'20)

- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes](http://arxiv.org/abs/2006.08848) (NIPS'20)

- ***Ditto*** -- [Ditto: Fair and Robust Federated Learning Through Personalization](http://arxiv.org/abs/2012.04221) (ICML'21)

- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks](http://arxiv.org/abs/2103.04628) (ICML'21)
  
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) (CVPR'22)

- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) (ArXiv'19)

- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization](http://arxiv.org/abs/2012.08565) (ICLR'21)

- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification](https://arxiv.org/abs/2106.06042) (ICLR'22)

- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare](https://arxiv.org/abs/2112.00734) (IEEE'22)

- ***kNN-Per*** -- [Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360) (ICML'22)

- ***MetaFed*** -- [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](http://arxiv.org/abs/2206.08516) (IJCAI'22)



## Easy Run 🏃‍♂️
```shell
# partition the CIFAR-10 according to Dir(0.1) for 100 clients
cd data/utils
python run.py -d cifar10 -a 0.1 -cn 100
cd ../../

# run FedAvg under default setting.
cd src/server
python fedavg.py
```

## Run with Customized Settings 🏃‍♂️
```shell
# run FedAvg under customized setting.
```

### Monitor 📈 (optional and recommended 👍)
1. Wandb

## Arguments 🔧

About the default values and hyperparameters of advanced FL methods, go check [`src/config/args.py`](https://github.com/mtuann/personalized-federated-learning-101/tree/master/src/config/args.py) for full details.

📢 All arguments have default value.

| General Argument          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--dataset`, `-d`         | The name of dataset that experiment run on.                                                                   |
| `--model`, `-m`           | The model backbone experiment used.                                                                           |
| `--seed`                  | Random seed for running experiment.                                                                           |
| `--join_ratio`, `-jr`     | Ratio for (client each round) / (client num in total).                                                        |
| `--global_epoch`, `-ge`   | Global epoch, also called communication round.                                                                |
| `--local_epoch`, `-le`    | Local epoch for client local training.                                                                        |
| `--finetune_epoch`, `-fe` | Epoch for clients fine-tunning their models before test.                                                      |
| `--test_gap`, `-tg`       | Interval round of performing test on clients.                                                                 |
| `--eval_test`, `-ee`      | Non-zero value for performing evaluation on joined clients' testset before and after local training.          |
| `--eval_train`, `-er`     | Non-zero value for performing evaluation on joined clients' trainset before and after local training.         |
| `--local_lr`, `-lr`       | Learning rate for client local training.                                                                      |
| `--momentum`, `-mom`      | Momentum for client local opitimizer.                                                                         |
| `--weight_decay`, `-wd`   | Weight decay for client local optimizer.                                                                      |
| `--verbose_gap`, `-vg`    | Interval round of displaying clients training performance on terminal.                                        |
| `--batch_size`, `-bs`     | Data batch size for client local training.                                                                    |
| `--use_cuda`              | Non-zero value indicates that tensors are in gpu.                                                             |
| `--visible`               | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                         |
| `--save_log`              | Non-zero value for saving algorithm running log in `FL-bench/out/{$algo}`.                                    |
| `--save_model`            | Non-zero value for saving output model(s) parameters in `FL-bench/out/{$algo}`.                               |
| `--save_fig`              | Non-zero value for saving the accuracy curves showed on Visdom into a `.jpeg` file at `FL-bench/out/{$algo}`. |
| `--save_metrics`          | Non-zero value for saving metrics stats into a `.csv` file at `FL-bench/out/{$algo}`.                         |


## Supported Datasets 🎨

This benchmark only support algorithms to solve image classification task for now.


Regular Image Datasets

- *MNIST* (1 x 28 x 28, 10 classes)

- *CIFAR-10/100* (3 x 32 x 32, 10/100 classes)

- *EMNIST* (1 x 28 x 28, 62 classes)

- *FashionMNIST* (1 x 28 x 28, 10 classes)

Medical Image Datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)


## Acknowledgement 🤗
