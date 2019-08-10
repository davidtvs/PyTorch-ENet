# Pre-trained models

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |     480x360      |     10     |  300   | 52.85<sup>3</sup> |       4.2        |                 1                 |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     1024x512     |     4      |  300   | 60.07<sup>4</sup> |       5.4        |                24                 |

## Per-class IoU: CamVid<sup>3</sup>

|         |  Sky  | Building | Pole  | Road  | Pavement | Tree  | Sig Symbol | Fence |  Car  | Pedestrian | Bicyclist |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: | :--------: | :---: | :---: | :--------: | :-------: |
| IoU (%) | 90.0  |   70.3   | 22.0  | 92.0  |   75.5   | 64.5  |    21.2    | 15.2  | 66.7  |    30.6    |   33.2    |

## Per-class IoU: Cityscapes<sup>4</sup>

|         | Road  | Sidewalk | Building | Wall  | Fence | Pole  | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky  | Person | Rider |  Car  | Truck |  Bus  | Train | Motorcycle | Bicycle |
| :-----: | :---: | :------: | :------: | :---: | :---: | :---: | :-----------: | :----------: | :--------: | :-----: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
| IoU (%) | 96.0  |   73.4   |   86.0   | 41.1  | 39.3  | 46.2  |     42.5      |     55.5     |    87.9    |  52.0   | 89.9  |  61.7  | 41.9  | 87.5  | 53.9  | 60.1  | 39.2  |    28.3    |  58.9   |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.
