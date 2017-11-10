# POSEidon: face-from-Depth for Driver Pose Estimation
Implementation in **Keras** and **Theano** of my paper published in **CVPR 2017** (Borghi *et al.*).

## How To

### Configuration
The model has been tested with the following configuration:
- Python 2.7
- OpenCV 2.4.11
- Keras 1.0.6
- Theano

### Dataset
In order to run the code, it is necessary download the dataset from [here](http://gazecapture.csail.mit.edu/) and following these steps:
- Create a folder for each run of each subject, naming from the number 000 to 100
- In each sub folder, a file named angles.txt is required.

### Train and Test
 The command to train the entwork is
```
python3 train.py
```
and to testis
```
python3 test.py
```
You must pass the following arguments:
- ```gpu/cpu``` to run the script on gpu o cpu
- ```1...n``` to run the script on your n-th gpu

## Results
Results are obtained on two types of input images: 224x224 (as in the original paper) and 64x64 (useful to fast training and testing, with low GPU memory requirements).
Results are expressed in term of *Mean Absolute Error* (MAE) and *Standard Deviation* (STD). 
Finally, the network has been tested on both datasets.

### Small dataset 
This dataset consist of 48k train images and 5k test images. All face detections are valid and all coordinates are positives.

| Input Size   | MAE               | STD              | Loss MSE  |
| :---:        |     :---:         |            :---: | :---:     |
|64x64         | 1.00, 1.10        | 1.21, 1.28       |  2.662    |
|224x224       | 1.42, 1.47        | 1.48, 1.55       |  4.410    |

### Original dataset

| Input Size   | MAE            | STD           |Loss MSE  |
| :---:         |     :---:     |         :---: |:---:     |
|64x64         | 1.45, 1.67     | 1.43, 1.62    |    3.942 |
|224x224       | -              | -             |          |

## Notes
### Data
Even after the sanity check about the detection validity, a lot of detections have negative coordinates.
Authors' response: *maybe* they used the last data about detections extracted.
So, I implemented two solutions. The first is called ```load_batch_from_names_random```, that loads a batch of data randomly, discarding negative coordinates. The second is ```load_batch_from_names_fixed```, that retrieves data from the last valid detection. I have used the first one for the experiments.

### Path
In the code are present some hardcoded paths, at the beginning of ```train``` and ```test``` scripts.

## Acknowledgements
This work is partially inspired by the work of [hugochan](https://github.com/hugochan).

