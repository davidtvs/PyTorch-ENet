# Pre-trained models

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |     480x360      |     10     |  300   | 51.08<sup>3</sup> |       4.2        |                 1                 |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     1024x512     |     4      |  300   | 59.03<sup>4</sup> |       5.4        |                20                 |

## Per-class IoU: CamVid<sup>3</sup>

|         |  Sky  | Building | Pole  | Road  | Pavement | Tree  | Sign Symbol | Fence |  Car  | Pedestrian | Bicyclist |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: | :---------: | :---: | :---: | :--------: | :-------: |
| IoU (%) | 89.8  |   68.2   | 19.9  | 90.1  |   71.6   | 62.7  |    17.8     | 15.1  | 65.9  |    25.8    |   34.3    |

## Per-class IoU: Cityscapes<sup>4</sup>

|         | Road  | Sidewalk | Building | Wall  | Fence | Pole  | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky  | Person | Rider |  Car  | Truck |  Bus  | Train | Motorcycle | Bicycle |
| :-----: | :---: | :------: | :------: | :---: | :---: | :---: | :-----------: | :----------: | :--------: | :-----: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
| IoU (%) | 96.1  |   73.9   |   86.0   | 39.5  | 41.8  | 45.6  |     43.6      |     54.9     |    88.1    |  53.5   | 90.1  |  62.5  | 41.4  | 88.3  | 41.3  | 56.9  | 34.6  |    24.1    |  59.6   |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an AMD Ryzen 5 3600 3.6GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.
