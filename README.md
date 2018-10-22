# Obstacle-detection-on-Rooftop-images

**Credits:**

[Paper Link](https://arxiv.org/abs/1803.08396) (CVPR'18)

Link to the reference github repository : https://github.com/hezhangsprinter/DCPDN
```
@inproceedings{dehaze_zhang_2018,		
  title={Densely Connected Pyramid Dehazing Network},
  author={Zhang, He and Patel, Vishal M},
  booktitle={CVPR},
  year={2018}
} 
```

**Changes:**
Added a script "dehaze.py" which performs these four steps and produces results according to our requirements.
*Resizing Images to 512 x 512
*Fixing some parameters to default values and some to the values which is working better for us.
*Performs Dehazing
*Resizing Images to 300 x 300

**Usage :-
 ```
 python dehaze.py --imgdir /path/to/images
 ```
 

