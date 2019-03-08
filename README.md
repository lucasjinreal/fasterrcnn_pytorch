# FasterRCNN

this is a modified version of FasterRCNN written by *jwyang*, original location [here](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0). It the fastest version in pytorch as it using a lot of accerleration to such as CUDA ROI, CUDA nms etc.

I did those enhancement and edit to the original:

- remove all redundant codes relate to python2;
- make all packages to relative import so that it much more stable and less couple;
- only with littel edit can do self-dataset training.


## Install

2 steps to install:

```
sudo pip3 install -r requirements.txt
cd lib
# this step must run, otherwise lib can not run correctly
python3 setup.py build
```


## Train

To train on a new dataset, recommend convert your data into VOC format, or simply label your data in VOC format. I suppose you orginise your data in this format:

```
- VOC_Like_DATA
    - JPEGImages
    - ImageSets
    - Annotations
```

In which `ImageSets` contains your `train.txt` and `val.txt` for train and val. And `JPEGImages` contains row images, `Annotations` contains all labels.

Now you should only change one place inside: `lib/model/utils/config.py`:

```
__C.DATA_DIR = '/media/jintain/sg/permanent/datasets/VOCdevkit/VOC2012'
```
to your own self data dir. then just kick off train!


## Inference

To do inference with trained model (we does not provide evaluation scripts, just directly using for check model performance by predict).

to be done.