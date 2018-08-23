# nuclei_instance_segmentation

### Model

![](images/U-Net-V2-white-bg.png)

### Results

![](images/loss.png)

![](images/meanIOU.png)

![](images/meanIOU_seed.png)

### Evaluaiton

With learned landscape in test time:

```
validation loss: 0.5188
validation mean IoU: 0.8843
validation mean IoU (seed): 0.6904

AP@0.50 = 0.9176
AR@0.50 = 0.7556
-----
AP@0.55 = 0.9003
AR@0.55 = 0.7414
-----
AP@0.60 = 0.8827
AR@0.60 = 0.7268
-----
AP@0.65 = 0.8591
AR@0.65 = 0.7075
-----
AP@0.70 = 0.8275
AR@0.70 = 0.6814
-----
AP@0.75 = 0.7830
AR@0.75 = 0.6448
-----
AP@0.80 = 0.6958
AR@0.80 = 0.5730
-----
AP@0.85 = 0.5535
AR@0.85 = 0.4558
-----
AP@0.90 = 0.3442
AR@0.90 = 0.2835
-----
AP@0.95 = 0.0828
AR@0.95 = 0.0681
-----
mAP@[.5:.95] = 0.6847
mAR@[.5:.95] = 0.5638
```

Without learned landscape in test time:

```
validation loss: 0.5188
validation mean IoU: 0.8843
validation mean IoU (seed): 0.6904

AP@0.50 = 0.9125
AR@0.50 = 0.7514
-----
AP@0.55 = 0.8963
AR@0.55 = 0.7380
-----
AP@0.60 = 0.8742
AR@0.60 = 0.7199
-----
AP@0.65 = 0.8514
AR@0.65 = 0.7011
-----
AP@0.70 = 0.8187
AR@0.70 = 0.6741
-----
AP@0.75 = 0.7687
AR@0.75 = 0.6329
-----
AP@0.80 = 0.6800
AR@0.80 = 0.5600
-----
AP@0.85 = 0.5377
AR@0.85 = 0.4428
-----
AP@0.90 = 0.3343
AR@0.90 = 0.2753
-----
AP@0.95 = 0.0820
AR@0.95 = 0.0675
-----
mAP@[.5:.95] = 0.6756
mAR@[.5:.95] = 0.5563
```
