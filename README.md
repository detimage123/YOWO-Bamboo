# YOWO-Bamboo: An enhanced model for giant panda action recognition 
We proposed an efficient real-time detection model YOWO-Bamboo for giant panda action recognition, which has demonstrated remarkable improvements over the YOWO-Plus by optimizing backbone and loss functions. The mean Average Precision (mAP) of our enhanced model was increased from 61.5% to 66.5%, achieving higher recognition accuracy.

# Improvement
- 2D backbone: The ConvNeXt network is utilized owing to its simplicity and efficiency, leveraging the large model scale from the series along with ImageNet-22K pre-trained weights at 224x224 resolution.

- Loss functions: For the confidence and bounding box regression loss functions, we opt for Huber Loss and Distance Intersection over Union (DIoU) loss to optimize object localization and prediction.


# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yowo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yowo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

# Visualization

![image](./img_files/panda_1.gif)


# Dataset
Our giant panda dataset, conforming to the AVA format, is a comprehensive compilation encompassing two integral parts: a training set and a validation set. The training set contains 1.5K videos, 1.35M frames, and 48K boxes, while the validation set has 176 videos, 158K frames, and 1.4K boxes. Panda’s behaviors are systematically classified into 22 diverse action types, encompassing continuous postures such as "continuous standing", daily activities like "walking", estrus-specific behaviors like "licking the vulva", and overarching activities designated as "locomotion". 

# Experiment

* AVA v2.2(our giant panda dataset)

|     Model      |    mAP    |   FPS   |    weight    |
|----------------|-----------|---------|--------------|
|    YOWO-Plus   |   61.5    |    34   |       [github](https://github.com/detimage123/download/blob/master/yowo_epoch_10.pth)      |
|    YOWOv2-N    |   31.7    |    -    |       -      |
|    YOWOv2-T    |   24.8    |    -    |       -      |
|    YOWOv2-M    |   35.1    |    -    |       -      |
|    YOWOv2-L    |   53.1    |    -    |       -      |
| YOWO-B(YOLOv2) |   65.4    |    34   |  |
|YOWO-B(ConvNeXt)|   66.5    |    30   |  |

## Train YOWO-Bamboo
* UCF101-24

```Shell
python train.py --cuda -d ucf24 -v yowo --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ucf.sh
```

* AVA
```Shell
python train.py --cuda -d ava_v2.2 -v yowo --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ava.sh
```

##  Test YOWO-Bamboo
* UCF101-24
For example:

```Shell
python test.py --cuda -d ucf24 -v yowo --weight path/to/weight --show
```

* AVA
For example:

```Shell
python test.py --cuda -d ava_v2.2 -v yowo --weight path/to/weight --show
```

##  Test YOWO-Bamboo on AVA video
For example:

```Shell
python test_video_ava.py --cuda -d ava_v2.2 -v yowo --weight path/to/weight --video path/to/video --show
```

Note that you can set ```path/to/video``` to other videos in your local device, not AVA videos.

## Evaluate YOWO-Bamboo
* UCF101-24
For example:

```Shell
# Frame mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --weight path/to/weight \
        --cal_frame_mAP \
```
```Shell
# Video mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --weight path/to/weight \
        --cal_video_mAP \
```


* AVA
Run the following command to calculate frame mAP@0.5 IoU:

```Shell
python eval.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo \
        --weight path/to/weight
```

Our YOWO-Bamboo's result of frame mAP@0.5 IoU on AVAv2.2(our giant panda dataset):
```Shell
AP@0.5IOU/climbing: 0.5757113617658555
AP@0.5IOU/Continued bipedal standing: 0.2649716081613984
AP@0.5IOU/Continued lying: 0.8984453325547908
AP@0.5IOU/continued prone: 0.8905966092013693
AP@0.5IOU/continued sitting: 0.9316373376049558
AP@0.5IOU/Continued standing: 0.6855850034354113
AP@0.5IOU/curl up: 0.014272047482379585
AP@0.5IOU/Drinking: 0.6272800219728261
AP@0.5IOU/Eating: 0.8651951536485323
AP@0.5IOU/Excretion: 0.047997439769149976
AP@0.5IOU/Exploring: 0.6344390447724938
AP@0.5IOU/Grooming: 0.76204946299934
AP@0.5IOU/Lick the chest and abdomen: 1.0
AP@0.5IOU/Lick the vulva: 1.0
AP@0.5IOU/Locomotive: 0.8094823905218087
AP@0.5IOU/Marking: 0.8710604453870625
AP@0.5IOU/Playing: 0.7815101075387594
AP0.5IOU/Pregnancy grasping: 0.26234469581801595
AP@0.5IOU/Resting: 0.7501787270419815
AP@0.5IOU/Rub Yin: 0.4
AP@0.5IOU/Rubbing: 0.6354587135837136
AP@0.5IOU/Walking: 0.9248658375791767
mAP@0.5IOU: 0.6651400609472282
```

## Demo
```Shell
# run test_video_ava
python test_video_ava.py --cuda -d ava_v2.2 -v yowo -size 224 --weight path/to/weight --video path/to/video
```
