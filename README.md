# Image Improvement [Dehazing]

**Credits:**

[Paper Link](https://arxiv.org/abs/1803.08396) (CVPR'18)

Link to Repository : https://github.com/hezhangsprinter/DCPDN
```
@inproceedings{dehaze_zhang_2018,		
  title={Densely Connected Pyramid Dehazing Network},
  author={Zhang, He and Patel, Vishal M},
  booktitle={CVPR},
  year={2018}
} 
```

**Changes :-**
Added a script "dehaze.py" which performs these four steps and produces results according to our requirements.
* Resizing Images to 512 x 512
* Fixing some parameters to default values and some to the values which is working better for us.
* Performs Dehazing
* Resizing Images to 300 x 300

**Usage :-**
 ```
 python dehaze.py --imgdir /path/to/images
 ```
 
 # Obstacle Detection
 
 **Credits :-**
 
 ```
 @misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
Link to Repository : https://github.com/matterport/Mask_RCNN

* Download training and validation dataset from the link below & put 'train' and 'val' folder in Mask_RCNN folder.
  - Train : https://www.dropbox.com/sh/a1slu2tancpz9s1/AADNe6auWg94Igm_Fli8raM_a?dl=0
  - Validation : https://www.dropbox.com/sh/eprk9sgist326xv/AABz3DahSbGKUcJD6sW73YN7a?dl=0

* Download pretrained obstacle weights from the link below and put it inside /Mask_RCNN/logs/obstacle_weight
  https://www.dropbox.com/s/nonyu5cgsz1zn0c/mask_rcnn_obstacle_0010.h5?dl=0

* Run Notebook inspect_obstacle_data.ipynb to verify dataset

* Run Notebook inspect_obstacle_model.ipynb to test

Configure Parameters in obstacle.py (if required)
