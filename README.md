# PyTorch-ENet

PyTorch implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147), ported from the lua-torch implementation [ENet-training](https://github.com/e-lab/ENet-training) created by the authors.

This implementation has been tested on the CamVid and Cityscapes datasets. Currently, a pre-trained version of the model trained in CamVid and Cityscapes is available [here](https://github.com/davidtvs/PyTorch-ENet/tree/master/save).


|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Epochs | Mean IoU (%) | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------:|:------------:|:---------------:|:---------------------------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |      480x360     |     10     |   300  |     42.49    |       7.4       |                 1                 |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          19          |     1024x512     |      2     |   300  |     45.77    |       4.3       |                 25                |

<sup>1</sup> When referring to the number of classes, the void/unlabelled class is always excluded.<br/>
<sup>2</sup> Just for reference since changes in implementation, datasets, and hardware can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.


## Installation

1. Python 3 and pip.
2. Set up a virtual environment (optional, but recommended).
3. Install dependencies using pip: ``pip install -r requirements.txt``.


## Usage

Run [``main.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [-h] [--mode {train,test,full}] [--resume RESUME]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--learning_rate LEARNING_RATE] [--lr_decay LR_DECAY]
               [--lr_decay_epochs LR_DECAY_EPOCHS]
               [--weight_decay WEIGHT_DECAY] [--dataset {camvid,cityscapes}]
               [--dataset_dir DATASET_DIR] [--height HEIGHT] [--width WIDTH]
               [--weighing {enet,mfb,none}]
               [--ignore_unlabelled IGNORE_UNLABELLED] [--workers WORKERS]
               [--print_step PRINT_STEP] [--imshow_batch IMSHOW_BATCH]
               [--cuda CUDA] [--name NAME] [--save_dir SAVE_DIR]
```

For help on the optional arguments run: ``python main.py -h``


### Examples: Training

```
python main.py -m train --save_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


### Examples: Resuming training

```
python main.py -m train --resume True --save_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


### Examples: Testing

```
python main.py -m test --save_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


## Project structure

### Folders

- [``data``](https://github.com/davidtvs/PyTorch-ENet/tree/master/data): Contains code to load the supported datasets.
- [``metric``](https://github.com/davidtvs/PyTorch-ENet/tree/master/metric): Evaluation-related metrics.
- [``models``](https://github.com/davidtvs/PyTorch-ENet/tree/master/models):
ENet model definition.
- [``save``](https://github.com/davidtvs/PyTorch-ENet/tree/master/save): By default, ``main.py`` will save models in this folder. The pre-trained models can also be found here.

### Files

- [``args.py``](https://github.com/davidtvs/PyTorch-ENet/tree/master/args): Contains all command-line options.
- [``main.py``](https://github.com/davidtvs/PyTorch-ENet/tree/master/main): Main script file used for training and/or testing the model.
- [``test.py``](https://github.com/davidtvs/PyTorch-ENet/tree/master/test): Defines the ``Test`` class which is responsible for testing the model.
- [``train.py``](https://github.com/davidtvs/PyTorch-ENet/tree/master/train): Defines the ``Train`` class which is responsible for training the model.
- [``transforms.py``](https://github.com/davidtvs/PyTorch-ENet/tree/master/transforms): Defines image transformations to convert an RGB image encoding classes to a ``torch.LongTensor`` and vice versa.
