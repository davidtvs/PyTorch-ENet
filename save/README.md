# Pre-trained models

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |     480x360      |     10     |  300   | 52.1<sup>3</sup> |       4.2        |                 1                 |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     1024x512     |     4      |  300   | 59.5<sup>4</sup> |       5.4        |                20                 |

## Per-class IoU: CamVid<sup>3</sup>

|         |  Sky  | Building | Pole  | Road  | Pavement | Tree  | Sign Symbol | Fence |  Car  | Pedestrian | Bicyclist |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: | :---------: | :---: | :---: | :--------: | :-------: |
| IoU (%) | 90.2  |   68.6   | 22.6  | 91.5  |   73.2   | 63.6  |    19.3     | 16.7  | 65.1  |    27.2    |   35.0    |

## Per-class IoU: Cityscapes<sup>4</sup>

|         | Road  | Sidewalk | Building | Wall  | Fence | Pole  | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky  | Person | Rider |  Car  | Truck |  Bus  | Train | Motorcycle | Bicycle |
| :-----: | :---: | :------: | :------: | :---: | :---: | :---: | :-----------: | :----------: | :--------: | :-----: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
| IoU (%) | 96.1  |   73.3   |   85.8   | 44.1  | 40.5  | 45.3  |     42.5      |     53.9     |    87.9    |  53.5   | 90.1  |  62.3  | 44.3  | 87.6  | 46.6  | 58.2  | 34.8  |    25.8    |  57.9   |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an AMD Ryzen 5 3600 3.6GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.
