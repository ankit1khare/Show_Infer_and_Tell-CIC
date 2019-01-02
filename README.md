# Ultra_Context_Image_Captioner
Code for the paper, "Ultra-Context: Maximising the Context for Better Image Caption Generation"

## Code supports:
- Self critical training
- Ensemble 
- Multi-GPU training
- Adam optimizer with reduce on plateau (learning rate reduction when plateaued)

## Requirements
- Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)
- PyTorch 0.4 (along with torchvision)
- cider

## Pretrained models (using resnet101 feature)
Pretrained models are provided [here](https://drive.google.com/open?id=0B7fNdx_jAqhtdE1JRXpmeGJudTg). 
Performance to be expected from pretrained models:
1. Ultra_CTX model (Cider: 124.2)
2. CTX_initemb (Cider: 122.7)]

## Download COCO captions and preprocess them
Extract `dataset_coco.json` from the zip file included in this repository and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

The code is based on [ruotianluo](https://github.com/ruotianluo/self-critical.pytorch) and [neuratalk2](https://github.com/karpathy/neuraltalk2). Please refer them for more details on setup.

## Notes about Training
Ultra_CTX model with LSTM size 2048 units is over 2 GB. With a batch size of 10 it take 3GB of GPU space. 

## Evaluate on Karpathy's test split and COCO split
Code includes COCO split option and evaluation on COCO server.

## Reference
Please consider citing:
```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```

## Acknowledgements
Thanks to google cloud for giving free trial credits, the original [neuraltalk2](https://github.com/karpathy/neuraltalk2), awesome PyTorch team and [rluo](https://github.com/ruotianluo/self-critical.pytorch).

