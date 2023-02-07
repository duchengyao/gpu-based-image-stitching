## 0x00 Intro

> This project is a simple implementation of opencv for the following papers.
> 
> Du, Chengyao, et al. (2020). GPU based parallel optimization for real time panoramic video stitching. Pattern Recognition Letters, 133, 62-69.

Fast panorama stitching method using UMat.

Speed of 4 cameras at 4k resolution is greater than 200fps in 1080ti.

This project does not provide a dataset so it cannot be used out of the box.

## 0x01 Quick Start
* Modify calibration file.

* Create Dataset(or use RTSP stream input by modifying the code slightly). 
```
├── results
└── datasets
    └── air-4cam-mp4
        ├── 00.MP4
        ├── 01.MP4
        ├── 02.MP4
        └── 03.MP4
```

* Run
```
$ mkdir build && cd build
$ cmake ..
$ make
$ ./image-stitching
```

## 0x02 Exposure Refine

origin-stitching  
![](assets/01.origin-stitching.png)

exposure-mask  
![](assets/02.exposure-mask.png)

exposure-mask-refine  
![](assets/03.exposure-mask-refine.png)

apply-mask
![](assets/04.apply-mask.png)

final-stitching
![](assets/05.final-stitching.png)

