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
   
# Training NER models with Prodigy

## The second approach to invoice entity recognition

1. Export OCR output files to a single file that Prodigy can read using the text loader

```bash
paste --delimiter=\\n --serial data/processed/ocr/*.txt > invoices.txt
```

2. Manually mark entity positions in a number of invoices

```bash
prodigy ner.manual invoice_data pl_core_news_md invoices.txt --loader txt 
  --label "INVOICE_NO, INVOICE_TYPE, PURCHASE_DATE, ISSUE_DATE, NIP_SELLER, NAME_SELLER, ADDRESS_SELLER, NIP_BUYER, 
           NAME_BUYER, ADDRESS_BUYER, NAME_PRODUCT, MEASURE_PRODUCT, QTY_PRODUCT, UNIT_PRICE_PRODUCT, 
           NET_AMOUNT_PRODUCT, TAX_RATE_PRODUCT, TAX_AMOUNT_PRODUCT, GROSS_AMOUNT_PRODUCT, NET_AMOUNT_TOTAL, 
           GROSS_AMOUNT_TOTAL, TAX_AMOUNT_TOTAL, PAYMENT_FORM, PAYMENT_DUE, BANK_ACCOUNT_NO" 
```

3. After manually annotating at least 50 invoices train the initial model

```bash
prodigy train ner invoice_data pl_core_news_md --output /tmp/invoice_model --eval-split 0.2 --n-iter 100
```

4. Verify if there is still capacity to learn more by giving more examples

```bash
prodigy train-curve ner invoice_data pl_core_news_md --eval-split 0.2 --n-samples 10
```

5. Add more examples using active learning by the initial model

```bash
prodigy ner.correct invoice_data_correct /tmp/invoice_model invoices.txt --loader txt 
    --label "INVOICE_NO, INVOICE_TYPE, PURCHASE_DATE, ISSUE_DATE, NIP_SELLER, NAME_SELLER, ADDRESS_SELLER, 
             NIP_BUYER, NAME_BUYER, ADDRESS_BUYER, NAME_PRODUCT, MEASURE_PRODUCT, QTY_PRODUCT, UNIT_PRICE_PRODUCT, 
             NET_AMOUNT_PRODUCT, TAX_RATE_PRODUCT, TAX_AMOUNT_PRODUCT, GROSS_AMOUNT_PRODUCT, NET_AMOUNT_TOTAL, 
             GROSS_AMOUNT_TOTAL, TAX_AMOUNT_TOTAL, PAYMENT_FORM, PAYMENT_DUE, BANK_ACCOUNT_NO" 
    --exclude invoice_data --unsegmented
```

6. Combine both datasets to train the final model

```bash
prodigy train ner invoice_data,invoice_data_correct pl_core_news_md --output /tmp/invoice_model_corrected 
    --eval-split 0.2 --n-iter 300 --dropout 0.05 --batch-size 1
```

