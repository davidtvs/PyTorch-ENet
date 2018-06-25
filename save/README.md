# Pre-trained models

|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Epochs | Mean IoU (%) | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------:|:------------:|:---------------:|:---------------------------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |      480x360     |     10     |   300  |     48.72<sup>3</sup>     |       7.4       |                 1                 |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          19          |     1024x512     |      2     |   300  |     55.71<sup>4</sup>     |       4.3       |                 25                |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.


## Per-class IoU: CamVid

|     |  Sky | Building | Pole | Road | Pavement | Tree | Sig Symbol | Fence |  Car | Pedestrian | Bicyclist |
|:---:|:----:|:--------:|:----:|:----:|:--------:|:----:|:----------:|:-----:|:----:|:----------:|:---------:|
| IoU (%) | 89.6 |   67.6   | 20.0 | 87.4 |   61.8   | 62.3 |    16.8    |  12.0 | 59.9 |    27.3    |    31.2   |

## Per-class IoU: Cityscapes

|         | Road | Sidewalk | Building | Wall | Fence | Pole | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky | Person | Rider |  Car | Truck |  Bus | Train | Motorcycle | Bicycle |
|:-------:|:----:|:--------:|:--------:|:----:|:-----:|:----:|:-------------:|:------------:|:----------:|:-------:|:----:|:------:|:-----:|:----:|:-----:|:----:|:-----:|:----------:|:-------:|
| IoU (%) | 95.5 |   70.1   |   84.4   | 33.6 |  35.5 | 43.7 |      37.3     |     50.8     |    86.9    |   48.2  | 89.6 |  58.5  |  39.9 | 86.9 |  38.0 | 49.5 |  31.5 |    22.2    |   56.4  |
