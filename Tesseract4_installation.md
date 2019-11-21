
Tesseract Version 4.0 - Installation
======
A quick guide on installing Google's Open Source OCR engine on debian based Linux distributions. 
The newest version 4.0 offers better accuracy than the predecessor with the inclusion of deep-learning capabilities with LSTM networks 
***

Tesseract is available directly from many Linux distributions which means we can easily install it by running the command: 

```
sudo apt install tesseract-ocr
```
With most of the scanned documents being in Polish it would be recommended to install a language and script pack as well, which can be done by running another command:
```
sudo apt install tesseract-ocr-pol
```
In this case we are installing the Polish language pack, but since we might run into other languages as well, we can either download them separately using the appropriate code (f.e. deu - German) or all of them with the command:
```
sudo apt install tesseract-ocr-all
``` 
***
More info about installing tesseract on different distros can be found in the tesseract-ocr  [official wiki.](https://github.com/tesseract-ocr/tesseract/wiki)
 
 ## Running Tesseract
To test if we have installed tesseract properly we can use the simplest command line invocation:

```
tesseract imagename outputbase 
```
Example: 
```
tesseract noisy_example.png noisy_output
```
![example_01](https://user-images.githubusercontent.com/34404522/69370594-88efbb80-0c9e-11ea-8d40-5261d3cf92f0.png)
And our output (noisy_output.txt) reads: 
>Noisyimage 
>to test
Tesseract OCR



Note:  
Any input must be readable by Leptonica, which includes formats like: BMP, PNM, PNG, JFIF, JPEG, and TIFF formats. 
Outputs formats available are: TXT, PDF, HOCR, TSV and PDF with text layer only. The standard output is a .txt file. 

And if we wished to test it out with polish language the command would look like:
```
tesseract imagename outputbase -l pol
```
Which with our previous example produced almost the same results: 
>Noisy image
to test
Tesseract OCR

More on arguments for tesseract command line usage can be found in the [official documentation.](https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage)
W
