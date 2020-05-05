# MapExtrakt

> Convolutional Nerual Networks Are Beautiful

We all take our eyes for granted, we glance at an object for an instant and  our brains identify objects with ease.
However distorted this information may be, we do a pretty good job at it.

Low light, obscured vision, there are a myriad of situations where conditions are poor but still we manage to understand what an object it.
Context helps, but we were created with sight in mind.

Computers have a harder time, but modern advances with Convolutional Neural Networks are making this task a reality and have now surpassed human level accuracy.

Computers are beautifull, Convolutional Neural Networks are beautifull. And the maps they create to determine what makes a cat a cat are beautiful.

### MapExtrakt makes viewing feature maps a breeze.

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
![Example Output](./examples/example_output.jpg "Example Output")

### View Layers At a Time

```python

#gather maps
img = fe.display_from_map(layer_no=2, out_type="pil", colourize=20, outsize=(1000,500), border=0.03, picture_in_picture=False)
img.save("example_output.jpg")
img

```
![Example Output](./examples/example_output2.jpg "Example Output")


### Export Cells Of Each Layer To Video

```python

#gather maps
fe.write_video(out_size=(1000,500), file_name="output.mp4", colourize=20,
               border=0.03, fps=60, frames_per_cell=1, fade_frames_between_cells=6,
               write_text=True, picture_in_picture=True)


```

<a href="https://www.youtube.com/watch?v=awBDPjCNAi4&feature=youtu.be" target="_blank">
    <img src="./examples/youtube.jpg" alt="MapExtrakt" border="10" />
</a>


------------------------------------------------

# Installation

## Is as easy as pie 

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

Why not view the jupyter notebook with more examples of usage.

[Examples](./examples/examples.ipynb)