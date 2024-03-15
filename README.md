# Data Distillation for Efficient In-Context Learning

This repository focuses on tools and scripts for data distillation in the context of efficient in-context learning. 
Our work builds upon the  [MetaICL](https://github.com/facebookresearch/MetaICL) codebase.


## Dependencies
- For data preprocessing, ensure you have `datasets==1.4.0` installed. However, this version isn't compatible with the Transformers version used for training and inference.
- We recommend setting up two separate environments: one for data preprocessing and another for model training/inference.

## Data Preprocessing

### Pretrain C4 dataset
We utilize the validation set of [C4](https://huggingface.co/datasets/c4/viewer/en/validation) dataset, select "**en**" subset of validation split.
You can also check our [preprocessed data](https://huggingface.co/datasets/bigheiniuJ/MyC4Validation) on Huggingface datasets. 


### Meta-train and Meta-test dataset
For details on downloading and preprocessing, kindly refer to the [MetaICL](https://github.com/facebookresearch/MetaICL) documentation.
You can also check our [preprocessed data](https://huggingface.co/datasets/bigheiniuJ/EvalMetaICLAll) on Huggingface datasets.

## Model Checkpoint
The model checkpoint is available in [Google Drive](https://drive.google.com/drive/folders/1Rpd08mp-Qvuup4YbmuFwDdWHH3OC2OFO?usp=sharing).

## Data Distillation Training
Inside [src](./src) directory, you will find:
- [dataset_distill.py](./src/dataset_distill.py) - This houses both the pretrain C4 dataset class and the meta-train/meta-test dataset class.
- [model_distill.py](./src/model_distill.py)- This manages the interaction between the large language model and the context distillation model.
- [SmallModel.py](./src/SmallModel.py)- This file contains the implementation of the context distillation model.
 


### Pre-training:
```shell
cd scripts
sh c4_pretrain.sh
```

### FineTuning
```shell
cd scripts
sh finetune.sh
```

## License
MetaICL is CC-BY-NC 4.0 licensed.

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{
li2024mend,
title={{MEND}: Meta Demonstration Distillation for Efficient and Effective In-Context Learning},
author={Yichuan Li and Xiyao Ma and Sixing Lu and Kyumin Lee and Xiaohu Liu and Chenlei Guo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=2Y5kBPtU0o}
}
```


