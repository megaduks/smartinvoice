# Training models with Prodigy

Suppose we want to train a statistical model for predicting the position of the invoice number in the text.

In order to train NER model with Prodigy the following steps must be followed.

1. Export OCR output to the format acceptable for Prodigy (JSON, JSONL, txt)

    ```bash
    python utils/ocr2json.py -i data/ocr_raw_3 -f json -o data/output.json
    ```

2. Create a database of annotations and start annotating data

    ```bash
    prodigy ner.manual invoice_number_db pl_model data/output.json --label INVOICE_NUMBER
    ```

    where `invoice_number_db` is the database where annotations will be stored, `pl_model` is the 
    spaCy language model, `data/output.json` is the source data for annotations, and `INVOICE_NUMBER`
    is the NER label to be trained.

3. Optionally, we can look at the annotations

    ```bash
    prodigy db.out invoice_number_db > ./annotations.json
    ``` 
   
4. After annotating enough data we can start training the model

    ```bash
    prodigy train ner invoice_number_db pl_model --output models/invoice_number_model 
    ```
   
   Tunable parameters of training include:
   - `--n-iter, -n`: number of iterations of training
   - `--batch-size, -b`: size of a single training batch
   - `--dropout, -d`: dropout rate
   - `--binary, -b`: boolean flag for binary NER tagging (either accept or reject label)