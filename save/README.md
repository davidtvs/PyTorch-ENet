# Pre-trained models

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |     480x360      |     10     |  300   | 53.12<sup>3</sup> |       4.2        |                 1                 |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     1024x512     |     4      |  300   | 60.94<sup>4</sup> |       5.4        |                24                 |

## Per-class IoU: CamVid<sup>3</sup>

|         |  Sky  | Building | Pole  | Road  | Pavement | Tree  | Sig Symbol | Fence |  Car  | Pedestrian | Bicyclist |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: | :--------: | :---: | :---: | :--------: | :-------: |
| IoU (%) | 89.8  |   68.6   | 20.0  | 91.9  |   73.4   | 63.9  |    20.9    | 17.3  | 71.8  |    31.5    |   35.1    |

## Per-class IoU: Cityscapes<sup>4</sup>

|         | Road  | Sidewalk | Building | Wall  | Fence | Pole  | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky  | Person | Rider |  Car  | Truck |  Bus  | Train | Motorcycle | Bicycle |
| :-----: | :---: | :------: | :------: | :---: | :---: | :---: | :-----------: | :----------: | :--------: | :-----: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
| IoU (%) | 96.1  |   74.0   |   85.9   | 45.2  | 43.3  | 46.0  |     45.4      |     55.5     |    87.8    |  51.4   | 90.1  |  62.7  | 42.8  | 88.0  | 53.1  | 60.4  | 45.9  |    25.7    |  58.8   |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.
