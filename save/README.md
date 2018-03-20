# Pre-trained models

|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Epochs | Mean IoU (%) | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------:|:------------:|:---------------:|:---------------------------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |      480x360     |     10     |   300  |     42.48    |       7.4       |                 1                 |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          19          |     1024x512     |      2     |   300  |     42.35    |       4.3       |                 25                |

<sup>1</sup> When referring to the number of classes, the void/unlabelled class is always excluded.

<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz.

<sup>3</sup> Test set.

<sup>4</sup> Validation set.

## Per-class IoU: CamVid

|     |  Sky | Building | Pole | Road | Pavement | Tree | Sig Symbol | Fence |  Car | Pedestrian | Bicyclist |
|:---:|:----:|:--------:|:----:|:----:|:--------:|:----:|:----------:|:-----:|:----:|:----------:|:---------:|
| IoU (%) | 88.6 |   64.2   | 17.7 | 85.1 |   61.8   | 56.9 |    16.2    |  11.7 | 59.1 |    22.3    |    26.2   |

## Per-class IoU: Cityscapes

|         | Road | Sidewalk | Building | Wall | Fence | Pole | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky | Person | Rider |  Car | Truck |  Bus | Train | Motorcicycle | Bicycle |
|:-------:|:----:|:--------:|:--------:|:----:|:-----:|:----:|:-------------:|:------------:|:----------:|:-------:|:----:|:------:|:-----:|:----:|:-----:|:----:|:-----:|:------------:|:-------:|
| IoU (%) | 84.1 |   53.8   |   81.1   | 13.2 |  16.6 | 34.1 |      18.8     |     30.0     |    84.6    |   31.5  | 87.2 |  46.6  |  18.6 | 77.8 |  38.2 | 49.6 |  28.4 |      8.4     |   44.3  |
