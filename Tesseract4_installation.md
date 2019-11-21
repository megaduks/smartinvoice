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
More info can be found in the tesseract-ocr documentation which can be found at the [official git repository.](https://github.com/tesseract-ocr/tesseract)


