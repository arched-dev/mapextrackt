import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import datetime
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import os


def get_rows_cols(no):
    for x in range(100):
        for y in range(100):
            if x * y == no and y * 0.49 < x < y * 1.51:
                return (x, y)


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


class FeatureExtractor:

    def __init__(self, model):
        """
        Accepts pytorch models for feature extraction from convolutional layers.
        Must call set_image after to load image before use.

        :param model: (pytorch model)

        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.__device)
        self.layers = self.__create_layers(count=True) - 1
        self.outputs = {}
        self.image = None

    def display_from_map(self, layer_no, cell_no=None, out_type="pil", colourize=20, outsize=None, border=None,
                         picture_in_picture=True):
        """
        returns image map of layer N and [cell n] if specified.

        :param layer_no: (int) The specific layer number to output
        :param cell_no: (int) The specific channel that you want to extract  DEFAULT None = Return full map
        :param out_type: (str) "pil" - for pillow image, "mat" for matplotlib, "np" for numpy array
        :param colourize: (int) from 1-20 applies different colour maps 0 == False or B.W Image
        :param outsize: (tuple) The size to reshape the cell in format (w,h)
        :param border: (float in range 0-1) Percentage of cell size to pad with border
        :param picture_in_picture: (bool) Draw original picture over the map
        :return:
        """

        self.__has_layers(layer_no)

        # get image map
        img = self.__return_feature_map(layer_no, single=cell_no, border=border, colourize=colourize)

        # return type

        if outsize != None:
            img = np.array(ResizeMe(outsize)(Image.fromarray(img)))

        if picture_in_picture:
            img = self.__write_picture_in_picture(img)

        if out_type.lower() == "pil":
            return Image.fromarray(img)
        elif out_type.lower() == "mat":
            fig = plt.figure()
            plt.imshow(img)
        else:
            return img

    def set_image(self, img, order_by_intensity=True):
        """
        Used to set the imput image.
        Can accept PIL image / numpy array / location of image as string

        :param img: (np.array / Pil image / STR path to file) The input file to be analysed
        :param order_by_intensity: (bool) If TRUE features from each layer are reordered by intensity.
        :return:
        """

        img = self.__convert_image_to_torch(img)
        self.__create_layers(img, intensity=order_by_intensity)

    def get_total_cells(self):
        ## returns total cells over all convolutions
        tot = 0
        for x in range(self.layers):
            for y in range(self.get_cells(x)):
                tot += 1
        return tot

    def get_cells(self, layer_no):
        ## gets total cells from given convolutional layer number

        self.__has_layers(layer_no)
        return self.outputs[layer_no].shape[1] - 1

    def write_video(self, out_size, file_name, colourize=20, border=0.03, fps=40, frames_per_cell=1,
                    fade_frames_between_cells=6, write_text=True, picture_in_picture=True):
        """
        Used to render video output from feature maps

        :param out_size: (tuple) desired output size
        :param file_name: (str) desired output file name - must be .mp4 ext
        :param colourize: (int) from 1-20 applies different colour maps 0 == False or B.W Image
        :param border: (float in range 0-1) Percentage of cell size to pad with border
        :param fps: (int) fps of video output, more = faster video
        :param frames_per_cell: (int) number of static frames for each map: more = longer screen time per cell
        :param fade_frames_between_cells: (int) number of frames to fade between cells: more = smoother transition
        :param write_text: (bool) Write layer numbers to output
        :param picture_in_picture: (bool) Draw original image over cell
        :return:
        """

        # check filename
        if not file_name.endswith(".mp4"):
            raise ValueError("Output filename must end with .mp4")

        # set fourcc type
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        # init video writer
        out = cv2.VideoWriter("./" + file_name, fourcc=fourcc, fps=fps, frameSize=out_size)

        # get counts
        tot = self.get_total_cells()
        count = 0
        start = datetime.datetime.now()

        # loop layers and cells
        for layer in range(self.layers):
            for cell in range(self.get_cells(layer)):

                # get image
                img = self.display_from_map(layer, cell, colourize=colourize, out_type="np", outsize=out_size,
                                            border=border, picture_in_picture=picture_in_picture)[:, :, ::-1]
                img1 = self.__get_next_image(layer, cell, colourize=colourize, outsize=out_size, border=border,
                                             picture_in_picture=picture_in_picture)[:, :, ::-1]

                # write text if needed

                if write_text:
                    img = self.__draw_text(img, f"Layer {layer} Cell {cell}")
                    img1 = self.__draw_text(img1, f"Layer {layer} Cell {cell}")

                # write static frames
                for static in range(frames_per_cell):
                    out.write(img)

                # get next image

                # write fade frames
                for im in weight_images(img, img1, fade_frames_between_cells):
                    out.write(im)

                # logs
                count += 1
                total_time = (datetime.datetime.now() - start).total_seconds()

                # print status
                print(
                    f"\r{count:<5}/{tot}   Total Time Taken {convert(total_time):10} Time Left {convert((total_time / count) * (tot - count)):10} {get_bar(count, tot)} ",
                    end="")

        print(f"\nVideo saved as {file_name}")

        out.release()

    def __get_next_image(self, x, y, outsize, border, colourize, picture_in_picture):
        ## used to get next cell for video rendering
        try:
            return self.display_from_map(x, y + 1, colourize=colourize, out_type="np", outsize=outsize, border=border,
                                         picture_in_picture=picture_in_picture)
        except:
            return self.display_from_map(x + 1, 0, colourize=colourize, out_type="np", outsize=outsize, border=border,
                                         picture_in_picture=picture_in_picture)

    def __draw_text(self, img, text):
        ## used to draw text onto image
        img = Image.fromarray(img)

        # set font size relative to output size
        size = int(img.size[0] / 20)

        # draw text using PIL
        # check for font location

        if os.name == "nt":
            font = ImageFont.truetype("C:\Windows\Fonts\arial.ttf", size)

        elif os.name == "posix":
            try:
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size)
            except:
                font = ImageFont.truetype("/usr/share/consolefonts/FOO.psf.gz", size)

        # draw with stroke
        draw = ImageDraw.Draw(img)
        draw.text((int(img.size[0] * 0.01), int(img.size[1] * 0.01)), text, (255, 255, 255), font=font, stroke_width=2,
                  stroke_fill=(0, 0, 0), )

        return np.array(img)

    def __has_layers(self, layer_no):
        # check layers are correct
        if layer_no < 0 or layer_no > self.layers:
            raise ValueError(f"Layer number not available. Please choose layer between range 0-{self.layers}")

    def __create_layers(self, x=None, count=False, intensity=True):
        ##create output laters

        if count:
            x = torch.rand(1, 3, 400, 400).to(self.__device)

        self.outputs = {}

        counter = -1

        for name, module in self.model.named_children():
            try:
                if type(module) == torch.nn.modules.Sequential:

                    for module1 in module.children():
                        x = module1(x)

                        if "Conv" in str(module1):
                            counter = counter + 1
                            if intensity:
                                self.outputs[counter] = self.__intensity_sort(x).squeeze(1)
                            else:
                                self.outputs[counter] = x
                else:
                    x = module(x)

                    if "Conv" in str(module):
                        counter = counter + 1
                        if intensity:
                            self.outputs[counter] = self.__intensity_sort(x).squeeze(1)
                        else:
                            self.outputs[counter] = x


            except RuntimeError as e:

                if str(e).find("size mismatch") < 0:
                    print("Error !")
                    print(e)

        if count:
            return counter

    def __convert_image_to_torch(self, img):

        if type(img) == np.ndarray or type(img) == np.array:
            self.image = Image.fromarray(img)
            img = torch.tensor(img.transpose((2, 0, 1))).unsqueeze(0).float().to(self.__device)
        elif type(img) == PIL.Image.Image:
            self.image = img
            img = torch.tensor(np.array(img).astype(np.uint8).transpose((2, 0, 1))).unsqueeze(0).float().to(
                self.__device)
        elif type(img) == str:
            self.image = Image.fromarray(cv2.imread(img))
            img = torch.tensor(cv2.imread(img).transpose((2, 0, 1))).unsqueeze(0).float().to(self.__device)
        else:
            raise ValueError("Input Unknown")
        return img

    def __colourize(self, img, colour_type=0):

        if colour_type == 0:
            base = np.zeros((img.shape[0], img.shape[1], 3))
            base[:, :, 0] = img
            base[:, :, 1] = img
            base[:, :, 2] = img
        else:
            return cv2.applyColorMap(img, colour_type, None)

    def __write_picture_in_picture(self, base_img, size=0.25):

        # get back image shape
        h, w, _ = base_img.shape

        # covert top image & get shape
        top_img = np.array(self.image)[:, :, ::-1]
        t_h, t_w, _ = top_img.shape

        # calculate new size
        new_w = w * size
        new_h = new_w * (t_h / t_w)

        # fit on new image
        try:
            base_img[int(h - new_h) + 1:, int(w - new_w):, :] = cv2.resize(top_img,
                                                                           (int(new_w),
                                                                            int(new_h)))
        except ValueError:
            base_img[int(h - new_h):, int(w - new_w):, :] = cv2.resize(top_img,
                                                                       (int(new_w),
                                                                        int(new_h)))
        return base_img

    def __return_feature_map(self, layer_no, single=None, border=None, colourize=20):

        # normalize output for np array
        out = (normalize_output(self.outputs[layer_no][0, :, :, :]) * 255).to("cpu").detach().numpy().astype(
            np.uint8).transpose(1, 2, 0)

        # get length

        length = out.shape[2]

        # return single cell if needed
        if single != None and type(single) == int:

            if single > length or single < 0:
                raise ValueError(f"Cell number not valid please select from range 0-{length}")

            img = out[:, :, single]

            # if colourize
            img = self.__colourize(img, colourize)

            # if border pad
            if border != None:
                img = self.__pad_arr(img, border)

            return img

        # get ideal shape of output
        x, y = get_rows_cols(length)

        count = 0

        # loop rows
        for idx in range(x):
            # loop columns
            for idy in range(y):
                # store image
                img = out[:, :, count]

                # if colourize
                img = self.__colourize(img, colourize)

                # if border pad
                if border != None:
                    img = self.__pad_arr(img, border)

                if idy == 0:
                    colu = img
                else:
                    # stack horizontally
                    colu = np.hstack([colu, img])

                count += 1
            # stack vertically
            if idx == 0:
                rows = colu
            else:
                rows = np.vstack([rows, colu])

        # return np.array
        return rows

    def __intensity_sort(self, tensor):

        ## used to sort the mappings by pixel intensity.

        pixel_mean = {}

        for i, x in enumerate(tensor.view(1, tensor.size(1), -1).squeeze()):
            pixel_mean[i] = torch.mean(x)

        pixel_mean = {k: v for k, v in sorted(pixel_mean.items(), key=lambda item: item[1])}

        return tensor[:, [list(pixel_mean.keys())], :, :]

    def __pad_arr(self, img, size):

        imgchange = Image.fromarray(img)

        pad = int(imgchange.size[0] * size)
        if pad <= 0:
            pad = 1

        return np.array(transforms.functional.pad(img=imgchange, padding=pad, fill=(0, 0, 0)))
