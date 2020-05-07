# MapExtrakt

> Convolutional Neural Networks Are Beautiful

We all take our eyes for granted, we glance at an object for an instant and our brains can identify with ease.
However distorted the information may be, we do a pretty good job at it.

Low light, obscured vision, poor eyesight... There are a myriad of situations where conditions are poor but still we manage to understand what an object it.
Context helps, but we humans were created with sight in mind.

Computers have a harder time, but modern advances with convolutional neural networks are making this task a reality and have now surpassed human level accuracy.

Computers are beautiful, neural networks are beautiful. And the maps they create to determine what makes a cat a cat are beautiful.

### MapExtrakt makes viewing feature maps a breeze.

#### Catch a glimpse of how a computer can see.

```python

# load a model 
import torchvision
model = torchvision.models.vgg16(pretrained=True)

#import FeatureExtractor
from MapExtrakt import FeatureExtractor

#load the model and image
fe = FeatureExtractor(model)
fe.set_image("cat.jpg")

#gather maps
img = fe.display_from_map(layer_no=2, out_type="pil", colourize=20, outsize=(1000,500), border=0.03, picture_in_picture=True)
img.save("example_output.jpg")
img

```
![Example Output](https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/example_output.jpg "Example Output")

### View Single Cells At a Time

```python

#gather maps
img = fe.display_from_map(layer_no=2, out_type="pil", colourize=20, outsize=(1000,500), border=0.03, picture_in_picture=False)
img.save("example_output.jpg")
img

```
![Example Output](https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/example_output2.jpg "Example Output")


### Export Cells Of Each Layer To Video

```python
#gather maps
fe.write_video(out_size=(1000,500), file_name="output.mp4", 
               write_text=True, picture_in_picture=True, draw_type="both")
```

<a href="https://www.youtube.com/watch?v=AvLTVaV5ID8&feature=youtu.be" target="_blank">
    <img src="https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/youtube.jpg" alt="MapExtrakt" border="10" />
</a>


------------------------------------------------

# Installation

## It's as easy as PyPI

```
pip install mapextrakt
```

or build from source in terminal 

```
git clone https://github.com/lewis-morris/mapextrackt
cd mapextrackt
pip install -e .
```

------------------------------------------------

# More Examples

For more - view the jupyter notebook with extra usage examples.

[Examples](./examples/examples.ipynb)

-----------------
Todo List
-----------------

- [ ] Add the ability to slice the class i.e  FeatureExtractor[1,3]
- [ ] Show parameters on the image 

-----------------
Author
-----------------

Created solely by me, but open to suggestions/ colaborators.