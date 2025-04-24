# Yolov2tiny_inference

Setting your weight path in `checkpoint = load(...)` in `inference.py`

To inference your image, simply either put your images in folder `images` or use your own folder then change the directory path in `images_dir` in function `demo()`.

RUN 
```
python inference.py
```
will give your inference result in `output_images`

RUN
```
python validate.py
```
will give your mAP result of your weight.