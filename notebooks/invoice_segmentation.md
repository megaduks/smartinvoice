---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
import pandas as pd

img_dir = 'data'
data = []
invoices = os.listdir(img_dir)

for invoice in invoices:
    if invoice.split(".")[1]!="png":
        continue
    data.append(invoice)

df = pd.DataFrame(data, columns=['img_file'])
```

```python
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytesseract
import cv2 as cv

%matplotlib widget


def unique_sorted_values_plus_select(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, "Select")
    return unique


class OCRDisplay():
    def process_text(self):
        xlim = self.ax.get_xlim()
        xlim = (int(round(xlim[0])), int(round(xlim[1])))
        ylim = self.ax.get_ylim()
        ylim = (int(round(ylim[0])), int(round(ylim[1])))
        im_crop = self.im[ylim[1]:ylim[0]-1, xlim[0]:xlim[1]-1]
        img_width = int(im_crop.shape[1] * self.scale.value)
        img_height = int(im_crop.shape[0] * self.scale.value)
        output_dimension = (img_width, img_height)

        self.parsed_text.value = "Processing..."
        im_crop = cv.resize(im_crop, output_dimension,
                            interpolation=cv.INTER_CUBIC)

        result = pytesseract.image_to_string(
            im_crop, config='--psm {}'.format(self.psm.value), lang='pol')
        self.parsed_text.value = result

    def on_xlims_change(self, event_ax):
        if self.event_counter == 0:
            self.event_counter = 1
            return
        else:
            self.event_counter = 0

        if not self.changed:
            self.changed = True
            return
        self.process_text()

    def on_ocr_params_change(self, change):
        if self.dropdown_invoice.value == "Select":
            return

        self.process_text()

    def display(self):
        plt.close()
        with self.out:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.ax.callbacks.connect('ylim_changed', self.on_xlims_change)
        display(self.gui)

    def on_image_change(self, change):
        self.parsed_text.value = ""

        if self.dropdown_invoice.value == "Select":
            return

        self.im = cv.imread('data/{}'.format(self.dropdown_invoice.value), 0)
        
        if self.invert.value == 'negative':
            self.im = cv.bitwise_not(cv.imread('data/{}'.format(self.dropdown_invoice.value), 0))

        self.changed = False
        self.ax.imshow(self.im, cmap='gray', vmin=0, vmax=255)
        self.changed = False
        self.ax.autoscale()

    def __init__(self, dataframe):
        plt.close()
        self.df = dataframe
        self.event_counter = 0
        self.changed = False
        
        self.dropdown_invoice = widgets.Dropdown(
            options=["Select"], description="Invoice:")
        
        self.dropdown_invoice.options = unique_sorted_values_plus_select(self.df.img_file)
        
        self.parsed_text = widgets.Textarea(
            value='',
            placeholder='The output will appear here',
            disabled=True,
            layout={'width': '99%', 'height': '400px'}
        )
        self.scale = widgets.IntSlider(
            value=3, min=1, max=5, description="Image scale")
        self.psm = widgets.IntSlider(
            value=6, min=0, max=13, description="OCR mode")
        self.invert = widgets.ToggleButtons(
            options = ['original','negative']
        )
        self.out = widgets.Output()
        
        with self.out:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        
        self.hud = widgets.VBox(
            [self.dropdown_invoice, self.scale, self.psm, self.invert, self.parsed_text])
        self.gui = widgets.HBox([self.out, self.hud])

        self.dropdown_invoice.observe(self.on_image_change, names='value')
        self.psm.observe(self.on_ocr_params_change, names='value')
        self.scale.observe(self.on_ocr_params_change, names='value')
        self.invert.observe(self.on_image_change, names='value')


# display(gui)
ocr = OCRDisplay(df)
```

```python
ocr.display()
```

```python

```
