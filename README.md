# NASA
Neighborhood-Adaptive Structure Augmented Metric Learning -- AAAI2022 oral

Pytorch implementation of NASA

### How to use ?

### 1. Install requirements on your environment.
* python=3.6.8
* pytorch=1.4.0
* numpy=1.16.0
* tqdm=4.54.1
* scipy
* Pillow

### 2. Preparation.
```
mkdir ../MyDataset
```
* Download data.
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))

* Extract the compressed file (tgz or zip) into `../MyDataset/`, e.g., for Cars-196, put the files in the `../MyDataset/Cars196`.
* Download pre-train model.
  - BN([Model](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth))
  
### 3. Train.
```
python demo.py
```
