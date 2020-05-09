# MapExtrakt
![Release](https://img.shields.io/github/v/release/lewis-morris/mapextrackt "Release")
> Convolutional Neural Networks Are Beautiful

We all take our eyes for granted, we glance at an object for an instant and our brains can identify with ease.
However distorted the information may be, we do a pretty good job at it.

Low light, obscured vision, poor eyesight... There are a myriad of situations where conditions are poor but still we manage to understand what an object it.
Context helps, but we humans were created with sight in mind.

Computers have a harder time, but modern advances with convolutional neural networks are making this task a reality and have now surpassed human level accuracy.

Computers are beautiful, neural networks are beautiful. And the maps they create to determine what makes a cat a cat are beautiful.

----------------------------

## MapExtrakt makes viewing feature maps a breeze.

### Catch a glimpse of how a computer can see.


# MapExtrakt Usage

----------------------------
> First import your model

```python
import torchvision
model = torchvision.models.vgg19(pretrained=True)
```

> Import MapExtract's Feature Extractor and load in the model

```python
from MapExtrackt import FeatureExtractor
fe = FeatureExtractor(model)
```

> Set image to be analysed - input can be PIL Image, Numpy array or filepath. We are using the path

```python
fe.set_image("pug.jpg")
```
> View Layers

```python
fe.display_from_map(layer_no=1)
```

![Example Output](https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/output.jpg "Example Output")

> View Single Cells At a Time

```python
fe.display_from_map(layer_no=2, cell_no=4)
```
![Example Output](https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/output1.jpg "Example Output")

> Slice the class to get a range of cells  (Layer 2 Cells 0-9)

```python
fe[2,0:10]
```
![Example Output](https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/output2.jpg "Example Output")

> Or Export Layers To Video

```python
fe.write_video(out_size=(1200,800), file_name="output.avi", time_for_layer=60, transition_perc_layer=0.2)
```

<a href="https://www.youtube.com/watch?v=LZTGIYxczFc&feature=youtu.be" target="_blank">
    <img src="https://raw.githubusercontent.com/lewis-morris/mapextrackt/master/examples/youtube.png" alt="MapExtrackt" border="10" />
</a>

------------------------------------------------
# More Examples

For LOTS more - view the jupyter notebook.

[Examples](./examples/examples.ipynb)

------------------------------------------------

# Installation

## It's as easy as PyPI

```
pip install mapextrackt
```

or build from source in terminal 

```
git clone https://github.com/lewis-morris/mapextrackt &&\
cd mapextrackt &&\
pip install -e .
```

------------------------------------------------

Todo List
-----------------

- [x] Add the ability to slice the class i.e  FeatureExtractor[1,3]
- [ ] Show parameters on the image 
- [x] Fix video generation
- [ ] Enable individual cells to be added to video 
- [x] Add video parameters such as duration in seconds.
- [ ] Clean up code 
- [ ] Make speed improvements

-----------------
Author
-----------------

Created by me, initially to view the outputs for my own pleasure. 

If anyone has any suggestions or requests please send them over I'd be more than happy to consider.

lewis.morris@gmail.com
