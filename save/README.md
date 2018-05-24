# Pre-trained models

|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Epochs | Mean IoU (%) | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------:|:------------:|:---------------:|:---------------------------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |      480x360     |     10     |   300  |     42.49    |       7.4       |                 1                 |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          19          |     1024x512     |      2     |   300  |     45.77    |       4.3       |                 25                |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.


## Per-class IoU: CamVid

|     |  Sky | Building | Pole | Road | Pavement | Tree | Sig Symbol | Fence |  Car | Pedestrian | Bicyclist |
|:---:|:----:|:--------:|:----:|:----:|:--------:|:----:|:----------:|:-----:|:----:|:----------:|:---------:|
| IoU (%) | 89.0 |   65.0   | 18.3 | 84.7 |   59.0   | 59.2 |    16.4    |  11.4 | 55.3 |    22.6    |    29.0   |

## Per-class IoU: Cityscapes

|         | Road | Sidewalk | Building | Wall | Fence | Pole | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky | Person | Rider |  Car | Truck |  Bus | Train | Motorcycle | Bicycle |
|:-------:|:----:|:--------:|:--------:|:----:|:-----:|:----:|:-------------:|:------------:|:----------:|:-------:|:----:|:------:|:-----:|:----:|:-----:|:----:|:-----:|:----------:|:-------:|
| IoU (%) | 83.0 |   56.1   |   75.3   | 28.6 |  24.8 | 37.2 |      29.8     |     38.8     |    83.4    |   42.2  | 82.7 |  50.0  |  34.5 | 81.6 |  35.6 | 43.7 |  13.9 |    18.3    |   50.3  |
