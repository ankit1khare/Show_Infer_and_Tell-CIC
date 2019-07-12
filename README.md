# Ultra_Context_Image_Captioner
Code for the paper, "Show, Infer and Tell: Contextual Inference for Creative Captioning"

## Code supports:
- Self Critical Sequence Training
- Model Ensemble Averaging 
- Multi-GPU training
- Adam optimizer with reduce on plateau (learning rate reduction when plateaued)

## Requirements
- Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)
- PyTorch 0.4 (along with torchvision)
- cider

## Download COCO captions and preprocess them
Extract `dataset_coco.json` from the zip file included in this repository and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

The code is based on [ruotianluo](https://github.com/ruotianluo/self-critical.pytorch) and [neuratalk2](https://github.com/karpathy/neuraltalk2). Please refer them for more details on setup.

## Notes about Training
Final model with LSTM size 2048 units is over 2 GB. With a batch size of 10 it take 3GB of GPU space. 

## Evaluate on Karpathy's test split and COCO split
Code includes COCO split option and evaluation script for testing on COCO server.

## Reference
If you use the code, please consider citing:
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

