import os

import cv2
import numpy as np
import scipy
from PIL import Image, ImageFont, ImageDraw, ImageStat
from torchvision import transforms as transforms


def get_rows_cols_no_size(no, width, height):
    answers = {}
    for x in range(100):
        for y in range(100):
            if x * y == no:
                answers[abs(1 - abs(((x * height) / (y * width))))] = [x, y]

    return answers[sorted(answers.keys())[0]][1], answers[sorted(answers.keys())[0]][0]


def get_rows_cols_with_size(no, width, height, act_size):
    act_h, act_w = act_size[::-1]
    ratio = act_h / act_w

    answers = {}
    for x in range(100):
        for y in range(100):
            if x * y == no:
                answers[abs(((x * height) / (y * width)) - ratio)] = [x, y]

    return answers[sorted(answers.keys())[0]][0], answers[sorted(answers.keys())[0]][1]


def get_rows_cols(no, width, height, act_size):
    if act_size is None:
        return get_rows_cols_no_size(no, width, height)
    else:
        return get_rows_cols_with_size(no, width, height, act_size)


class ResizeMe(object):

    def __init__(self, desired_size):
        pass

        self.desired_size = desired_size

    def __call__(self, img):

        img = np.array(img).astype(np.uint8)

        desired_ratio = self.desired_size[1] / self.desired_size[0]
        actual_ratio = img.shape[0] / img.shape[1]

        desired_ratio1 = self.desired_size[0] / self.desired_size[1]
        actual_ratio1 = img.shape[1] / img.shape[0]

        if desired_ratio < actual_ratio:
            img = cv2.resize(img, (int(self.desired_size[1] * actual_ratio1), self.desired_size[1]), None,
                             interpolation=cv2.INTER_AREA)
        elif desired_ratio > actual_ratio:
            img = cv2.resize(img, (self.desired_size[0], int(self.desired_size[0] * actual_ratio)), None,
                             interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (self.desired_size[0], self.desired_size[1]), None, interpolation=cv2.INTER_AREA)

        h, w, _ = img.shape

        new_img = np.zeros((self.desired_size[1], self.desired_size[0], 3))

        hh, ww, _ = new_img.shape

        yoff = int((hh - h) / 2)
        xoff = int((ww - w) / 2)

        new_img[yoff:yoff + h, xoff:xoff + w, :] = img

        return Image.fromarray(new_img.astype(np.uint8))


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def weight_images(img, img1, fades=4):
    for x in np.linspace(0, 1, fades + 1):
        yield cv2.addWeighted(np.array(img), 1 - x, np.array(img1), x, 0)


def get_bar(current, maxx, bar_length=30, bar_load="=", bar_blank="-"):
    perc = current / maxx
    bar = int(round(bar_length * perc, 0))
    blank = int(round(bar_length - (bar_length * perc), 0))
    return "[" + bar_load * bar + bar_blank * blank + "]" + f" {round(current / maxx * 100, 2)} % "


def pad_arr(img, size):

    imgchange = Image.fromarray(img)

    pad = int(imgchange.size[0] * size)
    if pad <= 0:
        pad = 1

    if len(imgchange.size) == 3:
        return np.array(transforms.functional.pad(img=imgchange, padding=pad, fill=(0, 0, 0)))
    else:
        return np.array(transforms.functional.pad(img=imgchange, padding=pad, fill=0))


def intensity_sort(image_layer):
    from scipy import stats
    ## used to sort the mappings by pixel intensity.

    #TODO Check this - sorting could be better !!
    ### I have tried many options to get them to organise nicely.
    ### This is the best i can come up with for now. That offers the best sorting to the eye

    pixel_mean = {}
    for i in range(image_layer.shape[2]):

        img = image_layer[:, :, i]
        #blured = blur(img)
        #mean = np.mean(blured)
        #median = np.median(blured)
        #mode = stats.mode(blured.reshape(blured.shape[0]*blured.shape[1]))[0].item()
        #cv2.calcHist(blured,)
        #img_mean = np.mean(cv2.calcHist([blured], [0], None, [256], [0, 256]).ravel())
        blured = blur(img)
        mean = brightness(blured)
        median = np.median(blured.ravel())
        pixel_mean[i] = (mean+median)


    pixel_mean = {k: v for k, v in sorted(pixel_mean.items(), key=lambda item: item[1], reverse=True)}

    return image_layer[:, :, list(pixel_mean.keys())]

def brightness( im_file ):
   im = Image.fromarray(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def blur(img):

    for x in range(2):
        img = cv2.blur(img, (5,5))
    return img

def colourize_image(img, colour_type=0):

    if colour_type == 0:
        base = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
        base[:, :, 0] = img
        base[:, :, 1] = img
        base[:, :, 2] = img
        return base
    else:
        return cv2.applyColorMap(img, colour_type, None)


def draw_text(img, text, subtext=None):
    ## used to draw text onto image
    img = Image.fromarray(img)

    # set font size relative to output size
    size = int(img.size[0] / 25)
    smaller_size = int(size*.8)
    # draw text using PIL
    # check for font location

    if os.name == "nt":
        font_loc = "C:\Windows\Fonts\arial.ttf"
    elif os.name == "posix":
        if os.path.isfile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            font_loc = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        elif os.path.isfile("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"):
            font_loc = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

    font = ImageFont.truetype(font_loc, size)
    smaller_font = ImageFont.truetype(font_loc, smaller_size)

    # draw with stroke
    draw = ImageDraw.Draw(img)
    init_w = int(img.size[0] * 0.01)
    init_h = int(img.size[1] * 0.01)
    text_col = (255, 255, 255)
    fill_col = (0, 0, 0)
    drawn = draw.text((init_w, init_h), text, text_col, font=font, stroke_width=2,stroke_fill=fill_col)

    if subtext is not None:

        size_of_sub = font.getsize(text)[1]
        sub_h = init_h + size_of_sub + img.size[0] * .005

        draw.text((init_w, sub_h),
                  subtext,
                  text_col,
                  font=smaller_font,
                  stroke_width=2,
                  stroke_fill=fill_col)

    return np.array(img)