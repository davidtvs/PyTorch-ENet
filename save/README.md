# Pre-trained models

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          11          |     480x360      |     10     |  300   | 52.46<sup>3</sup> |       4.2        |                 1                 |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     1024x512     |     4      |  300   | 59.40<sup>4</sup> |       5.4        |                24                 |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> These are just for reference. Implementation, datasets, and hardware changes can lead to very different results. Reference hardware: Nvidia GTX 1070 and an Intel Core i5-4570 3.2GHz. You can also train for 100 epochs or so and get similar mean IoU (± 2%).<br/>
<sup>3</sup> Test set.<br/>
<sup>4</sup> Validation set.


## Per-class IoU: CamVid

|         |  Sky  | Building | Pole  | Road  | Pavement | Tree  | Sig Symbol | Fence |  Car  | Pedestrian | Bicyclist |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: | :--------: | :---: | :---: | :--------: | :-------: |
| IoU (%) | 90.4  |   69.7   | 19.2  | 91.5  |   73.6   | 64.1  |    18.0    | 16.1  | 69.3  |    28.5    |   36.7    |

## Per-class IoU: Cityscapes

|         | Road  | Sidewalk | Building | Wall  | Fence | Pole  | Traffic light | Traffic Sign | Vegetation | Terrain |  Sky  | Person | Rider |  Car  | Truck |  Bus  | Train | Motorcycle | Bicycle |
| :-----: | :---: | :------: | :------: | :---: | :---: | :---: | :-----------: | :----------: | :--------: | :-----: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
| IoU (%) | 95.8  |   72.3   |   85.5   | 40.3  | 40.9  | 45.0  |     43.1      |     54.1     |    87.6    |  52.9   | 90.8  |  61.1  | 42.3  | 87.7  | 49.0  | 59.4  | 37.9  |    25.5    |  57.5   |
