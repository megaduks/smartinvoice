# Model for the detection of rotated images

The model is built using the `ludwig` machine learning library. 
The code and data are not stored in the repository. This README contains
information required to reproduce all the steps of data preparation and model training.

1. Upload all un-rotated images to a single directory and set the env variable  
    ```bash
    bash$ export IMG_DIR=<directory with images>
    bash$ cd $IMG_DIR
    ```  
1. Create directories for rotated images:
    ```bash
    bash$ mkdir 0 90 180 270
    ```
3. Using `ImageMagic` perform the rotation of image files and copy rotated images to correct direttories
    ```bash
    for img_file in *
      do
        cp $img_file 0
        convert $img_file -rotate 90 90/$img_file
        convert $img_file -rotate 180 180/$img_file
        convert $img_file -rotate 270 270/$img_file
      done
    ```
4. Create a training CSV with names of all files
    ```bash
    bash$ find . -name *.png > input_data.csv
    ```
5. Edit `input_data.csv` file:
     - add a header with column names: `image_path,label`
     - remove trailing `'./'` from each line
     - copy the first "word" in each line to the end of the line as a label (quite easy with `vim`)
6. Run training command
    ```bash
   bash$ ludwig train --data_csv input_data.csv --model_definition_file mdf_image_rotation.yaml 
   ```
 
 