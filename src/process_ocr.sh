#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -i input_dir -m model_dir"
   echo -e "\t-i Input directory with raw OCR output"
   echo -e "\t-m Directory containing NER recognition model"
   exit 1 # Exit script after printing help
}

# directories
default_input_dir="experimental/opencv-text-recognition"
default_model_dir="models/invoice_final_ner_model"
processed_ocr_dir="data/processed/ocr"
processed_json_dir="data/processed/json"

matchers="NIP,BANK_ACCOUNT_NO,REGON,INVOICE_NUMBER,GROSS_VALUE,DATE"

while getopts "i:m:" opt
do
   case "$opt" in
      i ) input_dir="$OPTARG" ;;
      m ) model_dir="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$input_dir" ]
then
  input_dir="$default_input_dir"
fi

if [ -z "$model_dir" ]
then
   model_dir="$default_model_dir"
fi

export PYTHONPATH=.:$PYTHONPATH

# Begin script in case all parameters are correct
python3 utils/clean_ocr.py -i "$input_dir" -o "$processed_ocr_dir" -m "$matchers"

python3 classifiers.py -i "$processed_ocr_dir" -o "$processed_json_dir" -m "$model_dir"
