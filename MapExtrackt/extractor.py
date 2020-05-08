import torch
import numpy as np
import cv2
import datetime
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from PIL import Image

from MapExtrackt.functions import get_rows_cols, ResizeMe, convert, weight_images, get_bar, pad_arr, intensity_sort, \
    colourize_image, draw_text


class Features:
    features = {}

    def __init__(self):
        self.features = {}
        self.hooks = 0
        self.names = []

    def hook_fn(self, module, input, output):
        if len(input[0].shape) > 2:
            self.features[self.hooks] = output
            self.hooks += 1
            name = str(module).split("(")[0]
            self.add_name(name)

    def add_name(self, name):
        self.names.append(name)

    def get_layers_number(self):
        return len(self.features)

    def get_layer_type(self, layer_no):
        return self.names[layer_no]

    def get_cells(self, layer_no):
        return self.features[layer_no].shape[1]


class FeatureExtractor:

    def __init__(self, model):
        """
        Accepts pytorch models for feature extraction from convolutional layers.
        Must call set_image after to load image before use.

        :param model: (pytorch model)

        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.__device)
        self.name = str(self.model).split('(')[0]

        self.__hooks = None
        self.layers = None
        self.outputs = []
        self.layer_names = []
        self.image = None
        self.__set_default_exp_attrs()

    def __set_default_exp_attrs(self):
        self.colourize = 20
        self.outsize = (800, 600)
        self.out_type = "pil"
        self.border = 0.03
        self.picture_in_picture = True
        self.write_text = "full"

    def set_output(self, out_type=None, colourize=None, outsize=None, border=None,
                   picture_in_picture=None, write_text=None):
        if colourize is not None:
            assert colourize in [x for x in range(1, 21)], "Colourize value not in correct range of 1-20"
            self.colourize = colourize

        if outsize is not None or outsize == False:
            assert (type(outsize) == tuple and len(
                outsize) == 2) or outsize == False, "Outsize value not tuple of (Width,Height)"
            if outsize == False:
                self.outsize = None
            else:
                self.outsize = outsize

        if out_type is not None:
            assert out_type.lower() in ["pil", "np", "mat"], 'out_type value not in "pil","np","mat"]'
            self.out_type = out_type

        if border is not None:
            assert type(border) == float and 0.001 < border < 1, "Border not float in range 0-1"
            self.border = border

        if picture_in_picture is not None:
            assert type(picture_in_picture) == bool, "picture_in_picture value not bool"
            self.picture_in_picture = picture_in_picture

        if write_text is not None:
            assert type(write_text) == str and write_text in ["full", "some", "none"], 'write_text value not bool or ' \
                                                                                       'not in ["full","some","none"]'
            self.write_text = write_text

    def __str__(self):
        return f"<BASE MODEL: {self.name}>\n" \
               f"---------------------------\n" \
               f"----- Class  settings -----\n" \
               f"---------------------------\n" \
               f"Layers: {self.layers}\n" \
               f"Total Cells: {self.get_total_cells()}\n" \
               f"Image: {'Not Loaded' if self.image == None else self.image.size}\n" \
               f"Device: {self.__device}\n" \
               f"---------------------------\n" \
               f"-- Image output settings --\n" \
               f"---------------------------\n" \
               f"Output Size: {self.outsize}\n" \
               f"Out Type: {self.out_type}\n" \
               f"Border Size: {self.border * 100}%\n" \
               f"Picture in picture: {self.picture_in_picture}\n" \
               f"Colourize Style: {self.colourize}\n" \
               f"Write Text: {self.write_text}"

    def __getitem__(self, item):

        # used to slice the class
        if type(item) == int:
            if 0 > item or item > self.layers or type(self.layers) != int:
                raise ValueError(f"Layer Number not in range 0-{self.layers} or not INT\n"
                                 f"(Slicing layers not supported)")
            return self.display_from_map(layer_no=item, cell_no=None)

        if len(item) == 2:
            return self.display_from_map(layer_no=item[0], cell_no=item[1])
        else:
            raise ValueError("Too many indices")

    def display_from_map(self, layer_no, cell_no=None, out_type=None, colourize=None, outsize=None, border=None,
                         picture_in_picture=None, write_text=None):
        """
        returns image map of layer N and [cell n] if specified.

        Class settings are updated

        :param layer_no: (int) The specific layer number to output
        :param cell_no: (int) The specific channel that you want to extract  DEFAULT None = Return full map
        :param out_type: (str) "pil" - for pillow image, "mat" for matplotlib, "np" for numpy array
        :param colourize: (int) from 1-20 applies different colour maps 0 == False or B.W Image
        :param outsize: (tuple) The size to reshape the cell in format (w,h)  FALSE defaults to actual size of cell, or stacked cells that form layer.
        :param border: (float in range 0-1) Percentage of cell size to pad with border
        :param picture_in_picture: (bool) Draw original picture over the map
        :param write_text: (str) [default] "full" includes layer,cell, cell size and layer type to output
        "some" only includes layer,cell, cell size
        "none" does not write text to output
        :return: output image
        """

        self.set_output(out_type=out_type,
                        colourize=colourize,
                        outsize=outsize,
                        border=border,
                        picture_in_picture=picture_in_picture,
                        write_text=write_text)

        self.__has_layers(layer_no)

        # get image map
        img = self.__return_feature_map(layer_no, single=cell_no)

        # return type

        if self.outsize != None:
            img = np.array(ResizeMe(self.outsize)(Image.fromarray(img)))

        if self.picture_in_picture:
            img = self.__write_picture_in_picture(img)

        if self.write_text.lower() != "none":
            subtext = ""
            img = self.__write_text_(img, layer_no, cell_no)

        if self.out_type.lower() == "pil":
            return Image.fromarray(img)
        elif self.out_type.lower() == "mat":
            fig = plt.figure()
            plt.imshow(img)
        else:
            return img

    def __write_text_(self, img, layer_no, cell_no):

        if self.write_text.lower() == "full":
            subtext = self.layer_names[layer_no]
        if cell_no is None:
            img = draw_text(img, f"Layer {layer_no:<3} Cells {self.get_cells(layer_no):<4} ( "
                                 f"{self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"
                            , subtext)
        elif type(cell_no) == slice:
            if cell_no.start == None and cell_no.stop == None:
                img = draw_text(img, f"Layer {layer_no:<3} Cells {self.get_cells(layer_no):<4} ("
                                     f"{self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"

                            , subtext)
            elif cell_no.start == None:
                img = draw_text(img, f"Layer {layer_no:<3} Cell Range {'0':<4}-{cell_no.stop:<4} ( "
                                     f"{self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"
                                , subtext)

            elif cell_no.stop == None:
                img = draw_text(img, f"Layer {layer_no:<3} Cell Range {cell_no.start:<4}-{self.get_cells(layer_no):<4}"
                                     f" ( {self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"
                                , subtext)

            else:
                img = draw_text(img, f"Layer {layer_no:<3} Cell Range {cell_no.start:<4}-{cell_no.stop:<4} ( "
                                 f"{self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"
                            , subtext)
        else:
            img = draw_text(img, f"Layer {layer_no:<3} Cell # {cell_no + 1:<4} ( "
                                 f"{self.outputs[layer_no].shape[0]}x{self.outputs[layer_no].shape[1]} )"
                            , subtext)
        return img

    def set_image(self, img, order_by_intensity=True, allowed_modules=["Conv"], normalize_layer=False):
        """
        Used to set the input image.
        Can accept PIL image / numpy array / location of image as string

        :param img: (np.array / Pil image / STR path to file) The input file to be analysed
        :param order_by_intensity: (bool) If TRUE features from each layer are reordered by intensity.
        :param allowed_modules: (list or str) ["conv","relu"]  only extracts conv or relu layers, no need to add full
        name. i.e "conv" will extract Conv2d layers.
        "pool" (string not list) extracts all layers with "pool" in the module name.
        Empty list [] returns all layers.
        :param normalize_layer: (bool) if true normalizes over whole layer, if false normalization is conducted on an
        image by image basis.
        :return: None
        """

        if self.__device == "cuda":
            torch.cuda.empty_cache()

        # set hooks
        # register forward hooks
        self.__hooks = self.__set_hooks(allowed_modules)
        self.layers = self.__hooks.get_layers_number()
        self.outputs = self.__hooks.features

        # convert image
        img = self.__convert_image_to_torch(img)
        # infer
        out = self.model(img)
        # set total layers
        self.layers = self.__hooks.get_layers_number()
        # get outputs
        self.outputs = self.__hooks.features
        # set layer names
        self.layer_names = self.__hooks.names
        # order by pixel intensity

        # normalize features for viewing
        self.__normalize_features(normalize_layer)

        if order_by_intensity:
            outs = {}
            for k, v in self.outputs.items():
                outs[k] = intensity_sort(v)
            self.outputs = outs

    def get_total_cells(self):
        ## returns total cells over all convolutions
        tot = 0
        for x in range(self.layers):
            for y in range(self.get_cells(x) - 1):
                tot += 1
        return tot

    def get_cells(self, layer_no):
        ## gets total cells from given convolutional layer number

        self.__has_layers(layer_no)
        return self.outputs[layer_no].shape[2]

    def write_video(self, out_size, file_name, colourize=20, border=0.03, fps=40, frames_per_cell=1,
                    fade_frames_between_cells=2, write_text=True, picture_in_picture=True,
                    frames_per_layer=150, fade_frames_per_layer=40, draw_type="layers"):
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
        :param frames_per_layer: (int) Frames to draw per layer IF draw_layers == True
        :param fade_frames_per_layer: Frames to fade between layers IF draw_layers == True
        :param draw_type: (str) "layers" to only draw layers "cells" to only draw cells "both" to draw both
        :return: None
        """

        # check filename
        if not file_name.endswith(".mp4"):
            raise ValueError("Output filename must end with .mp4")

        # set fourcc type
        fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # init video writer
        out = cv2.VideoWriter("./" + file_name, fourcc=fourcc, fps=fps, frameSize=out_size)

        # set draw type
        if draw_type == "layers":
            draw_layers = True
            draw_cells = False
        elif draw_type == "cells":
            draw_layers = False
            draw_cells = True
        elif draw_type == "both":
            draw_layers = True
            draw_cells = True
        else:
            raise ValueError("Incorrect draw type")

        # get counts
        tot = self.get_total_cells()
        count = 0
        start = datetime.datetime.now()

        # if draw layers first
        if draw_layers:
            for layer in range(0, self.layers + 1):
                if layer < self.layers:
                    img = self.display_from_map(layer_no=layer, out_type="np", colourize=colourize, outsize=out_size,
                                                border=border,
                                                picture_in_picture=picture_in_picture)
                    img1 = self.display_from_map(layer_no=layer + 1, out_type="np", colourize=colourize,
                                                 outsize=self.out_size, border=border,
                                                 picture_in_picture=picture_in_picture)
                    img1_base = img1.copy()
                else:
                    # to deal with last image.
                    img = img1_base.copy()
                    img1 = img1_base.copy()

                if write_text:
                    # if write text displays layer cell and feature size
                    img = draw_text(img, f"Layer {layer: 3} - Cells {self.get_cells(layer): 4} - "
                                         f"{self.outputs[layer].size()[2]}x{self.outputs[layer].size()[3]}")

                    img1 = draw_text(img1, f"Layer {layer: 3} - Cells {self.get_cells(layer): 4} - "
                                           f"{self.outputs[layer].size()[2]}x{self.outputs[layer].size()[3]}")

                for times in range(frames_per_layer):
                    out.write(img[:, :, ::-1])

                for im in weight_images(img, img1, fade_frames_per_layer):
                    out.write(im[:, :, ::-1])
                # logs
                count += 1
                total_time = (datetime.datetime.now() - start).total_seconds()

                # print status
                print(
                    f"\rDrawing Layers {count:<5}/{self.layers + 1}   Total Time Taken {convert(total_time):10} Time Left"
                    f" {convert((total_time / count) * (self.layers + 1 - count)):10} {get_bar(count, self.layers + 1)} ",
                    end="")

        count = 0
        start = datetime.datetime.now()

        # loop layers and cells
        if draw_cells:

            for layer in range(self.layers):
                for cell in range(self.get_cells(layer) - 1):

                    # get image
                    img = self.display_from_map(layer, cell, colourize=self.colourize, out_type="np",
                                                outsize=self.outsize,
                                                border=self.border, picture_in_picture=self.picture_in_picture)[:, :,
                          ::-1]
                    img1 = self.__get_next_image(layer, cell, colourize=self.colourize, outsize=self.outsize,
                                                 border=self.border,
                                                 picture_in_picture=self.picture_in_picture)[:, :, ::-1]

                    # write text if needed

                    if self.write_text:
                        # if write text displays layer cell and feature size
                        img = draw_text(img, f"Layer {layer} Cell {cell}   - "
                                             f"{self.outputs[layer].size()[2]}x{self.outputs[layer].size()[3]}")
                        img1 = draw_text(img, f"Layer {layer} Cell {cell}   - "
                                              f"{self.outputs[layer].size()[2]}x{self.outputs[layer].size()[3]}")

                    # write static frames
                    for static in range(frames_per_cell):
                        out.write(img)

                    # write fade frames
                    for im in weight_images(img, img1, fade_frames_between_cells):
                        out.write(im)

                    # logs
                    count += 1
                    total_time = (datetime.datetime.now() - start).total_seconds()

                    # print status
                    print(
                        f"\rDrawing Cells {count:<5}/{tot}   Total Time Taken {convert(total_time):10} Time Left {convert((total_time / count) * (tot - count)):10} {get_bar(count, tot)} ",
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

    def __normalize_features(self, normalize_layer=True):

        for k, v in self.outputs.items():
            if not normalize_layer:
                for i, img in enumerate(v.squeeze()):
                    mx = torch.max(img.squeeze())
                    mn = torch.min(img.squeeze())
                    changed = img.squeeze() - mn
                    changed = changed / mx
                    out = (changed * 255).detach().to("cpu").numpy().astype(np.uint8)

                    if i == 0:
                        new_output = out.reshape((out.shape[0], out.shape[1], 1))
                    else:
                        new_output = np.concatenate((new_output, out.reshape((out.shape[0], out.shape[1], 1))), 2)

                self.outputs[k] = new_output
            else:
                mx = torch.max(v.squeeze())
                mn = torch.min(v.squeeze())
                changed = v.squeeze() - mn
                changed = changed / mx
                print(self.__hooks.names[k])
                out = (changed * 255).detach().to("cpu").numpy().astype(np.uint8).transpose((1, 2, 0))
                self.outputs[k] = out

    def __has_layers(self, layer_no):
        # check layers are correct
        if layer_no < 0 or layer_no > self.layers:
            raise ValueError(f"Layer number not available. Please choose layer between range 0-{self.layers}")

    def __set_hooks(self, allowed_modules):
        hooker = Features()
        # extract only allowed modules
        if type(allowed_modules) == str:
            allowed_modules = [allowed_modules]

        allowed_modules = [x.lower() for x in allowed_modules]
        count = 0
        for module in self.model.modules():
            if str(module).count("\n") == 0:
                name = str(module).split("(")[0]
                if name.lower().find("linear") >= 0:
                    break
                if len(allowed_modules) > 0:
                    for allow in allowed_modules:
                        if allow in name.lower():
                            count += 1
                            module.register_forward_hook(hooker.hook_fn)
                else:
                    count += 1
                    module.register_forward_hook(hooker.hook_fn)

        # check hooks added
        if count == 0:
            raise ValueError(f"No layers extracted with current 'allowed_module' paramater {allowed_modules}")

        return hooker

    def __convert_image_to_torch(self, img):

        if type(img) == np.ndarray or type(img) == np.array:
            self.image = Image.fromarray(img)
            img = torch.tensor(img.transpose((2, 0, 1))).unsqueeze(0).float().to(self.__device)
        elif type(img) == PIL.Image.Image:
            self.image = img
            img = torch.tensor(np.array(img).astype(np.uint8).transpose((2, 0, 1))).unsqueeze(0).float().to(
                self.__device)
        elif str(type(img)).find("PIL") >= 0:
            self.image = img
            img = torch.tensor(np.array(img).astype(np.uint8).transpose((2, 0, 1))).unsqueeze(0).float().to(
                self.__device)
        elif type(img) == str:
            self.image = Image.fromarray(cv2.imread(img))
            img = torch.tensor(cv2.imread(img).transpose((2, 0, 1))).unsqueeze(0).float().to(self.__device)
        else:
            raise ValueError("Input Unknown")
        return img

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

        for x in range(-2, 2):
            try:
                base_img[int(h - new_h) + x:, int(w - new_w):, :] = cv2.resize(top_img,
                                                                               (int(new_w),
                                                                                int(new_h)))
            except ValueError:
                pass

        return base_img

    def __return_feature_map(self, layer_no, single=None):
        # get total layer cells
        total_cells = self.get_cells(layer_no)-1
        stepper = 1
        # get ideal shape of output

        # for when sliced
        if type(single) == slice:
            #for slice like [:] display all
            if single.start == None and single.stop == None:
                length = self.outputs[layer_no].shape[2]
                count = 0
                count_to = total_cells
            #for slice like [:10]
            elif single.start == None and type(single.stop) == int:
                length = single.stop
                count = 0
                count_to = single.stop
            # for slight like [2:]
            elif type(single.start) == int and single.stop == None:
                length = total_cells - single.start
                count = single.start
                count_to = total_cells
            #for slice like [3:10]

            else:
                length = single.stop - single.start
                count = single.start
                count_to = single.stop

            #set step size

            if single.step == None:
                pass
            elif single.step < 0:
                count, count_to = count_to, count
                if count == total_cells:
                    count = total_cells
                stepper = single.step
            elif single.step >= 1:
                stepper = single.step

            if length < 0:
                length *= -1
            if hasattr(single,"step"):
                if not single.step is None and single.step != -1:
                    length = int(length/abs(single.step))


        # for single retrevial
        elif single != None and 0 <= single < total_cells:

            img = self.outputs[layer_no][:, :, single]

            # if colourize
            if self.colourize > 0:
                img = colourize_image(img, self.colourize)

            # if border pad
            if self.border is not None or self.border != 0:
                img = pad_arr(img, self.border)

            return img

        # for layer output
        elif single == None:
            length = self.outputs[layer_no].shape[2]
            count = 0
            count_to = total_cells
        # for error
        elif (single < 0 or single > total_cells) and single is not None:
            raise ValueError(f"Cell number not valid please select from range 0-{total_cells}")


        x, y = get_rows_cols(length,
                             width=self.outputs[layer_no].shape[1],
                             height=self.outputs[layer_no].shape[0],
                             act_size=self.outsize)

        # loop rows
        for idx in range(x):
            # loop columns
            for idy in range(y):
                # store image
                img = self.outputs[layer_no][:, :, count]

                # if colourize
                if self.colourize > 0:
                    img = colourize_image(img, self.colourize)

                # if border pad
                if self.border != None:
                    img = pad_arr(img, self.border)

                if idy == 0:
                    colu = img
                else:
                    # stack horizontally
                    colu = np.hstack([colu, img])

                if count + stepper > count_to and stepper > 0:
                    break
                elif count + stepper < count_to and stepper < 0:
                    break
                else:
                    count += stepper

            # stack vertically
            if idx == 0:
                rows = colu
            else:
                rows = np.vstack([rows, colu])

        # return np.array
        return rows
