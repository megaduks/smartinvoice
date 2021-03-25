# Training models with Prodigy

Suppose we want to train a statistical model for predicting the position of the invoice number in the text.

In order to train NER model with Prodigy the following steps must be followed.

1. Clean all spurious characters from OCR output

    ```bash
   python utils/clean_ocr.py -i data/ocr_raw_4_blocks -o data/ocr_raw_4_blocks/clean 
        -m NIP,BANK_ACCOUNT_NO,REGON,INVOICE_NUMBER,GROSS_VALUE,DATE,MONEY
   
   ```
1. Export OCR output to the format acceptable for Prodigy (JSON, JSONL, txt)

    ```bash
    python utils/ocr2json.py -i data/ocr_raw_r_blocks/clean -f json -o data/output.json
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

1. Clean all spurious characters from OCR output

    ```bash
   python utils/clean_ocr.py -i data/ocr_raw_4_blocks -o data/processed/ocr 
        -m NIP,BANK_ACCOUNT_NO,REGON,INVOICE_NUMBER,GROSS_VALUE,DATE,MONEY
   
   ```

1. Export OCR output files to a single file that Prodigy can read using the text loader

```bash
paste --delimiter=\\n --serial data/processed/ocr/*.txt > invoices.txt
```

2. Manually mark entity positions in a number of invoices

```bash
prodigy ner.manual invoice_data pl_core_news_lg data/processed/ocr/invoices.txt --loader txt 
  --label "INVOICE_NO, INVOICE_TYPE, PURCHASE_DATE, ISSUE_DATE, NIP_SELLER, REGON_SELLER, NIP_BUYER, 
           REGON_BUYER, NAME_PRODUCT, MEASURE_PRODUCT, QTY_PRODUCT, UNIT_PRICE_PRODUCT, 
           NET_AMOUNT_PRODUCT, TAX_RATE_PRODUCT, TAX_AMOUNT_PRODUCT, GROSS_AMOUNT_PRODUCT, NET_AMOUNT_TOTAL, 
           GROSS_AMOUNT_TOTAL, TAX_AMOUNT_TOTAL, PAYMENT_FORM, PAYMENT_DUE, BANK_ACCOUNT_NO" 
```

3. After manually annotating at least 50 invoices train the initial model

```bash
prodigy train ner invoice_data pl_core_news_lg --output /tmp/invoice_model --eval-split 0.2 --n-iter 100
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

### The third approach (separate annotation of entities and joint training)

1. Clean all spurious characters from OCR output

    ```bash
   python utils/clean_ocr.py -i data/ocr_raw_4_blocks -o data/processed/ocr 
        -m NIP,BANK_ACCOUNT_NO,REGON,INVOICE_NUMBER,GROSS_VALUE,DATE,MONEY   
   ```

1. Export OCR output files to a single file that Prodigy can read using the text loader

    ```bash
    paste --delimiter=\\n --serial data/processed/ocr/*.txt > invoices.txt
    ```

1. Create a Prodigy dataset to store annotations of invoice numbers

    ```bash
    prodigy dataset invoice_no_dataset "Dataset with INVOICE_NO annotations"
    ```

1. Manually annotate examples of invoice numbers

    ```bash
    prodigy ner.manual invoice_no_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label INVOICE_NO
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner invoice_no_dataset pl_core_news_lg 
        --output models/ner_model_invoice_no --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach invoice_no_dataset models/ner_model_invoice_no data/processed/ocr/invoices.txt 
        --loader txt --label INVOICE_NO --unsegmented
   ```
   
1. Transform annotations into a JSON file

    ```bash
   prodigy terms.to-patterns invoice_no_dataset models/invoice_no.jsonl --label INVOICE_NO
        --spacy-model pl_core_news_lg  
   ```
   
1. Create a Prodigy dataset to store annotations of invoice types

    ```bash
    prodigy dataset invoice_type_dataset "Dataset with INVOICE_TYPE annotations"
    ```

1. Manually annotate examples of invoice numbers

    ```bash
    prodigy ner.manual invoice_type_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label INVOICE_TYPE
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner invoice_type_dataset pl_core_news_lg 
        --output models/ner_model_invoice_type --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach invoice_type_dataset models/ner_model_invoice_type data/processed/ocr/invoices.txt 
        --loader txt --label INVOICE_TYPE --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of purchase dates

    ```bash
    prodigy dataset purchase_date_dataset "Dataset with PURCHASE_DATE annotations"
    ```

1. Manually annotate examples of purchase dates

    ```bash
    prodigy ner.manual purchase_date_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label PURCHASE_DATE
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner purchase_date_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach purchase_date_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label PURCHASE_DATE --unsegmented
   ```
   
1. Create a Prodigy dataset to store annotations of issue dates

    ```bash
    prodigy dataset issue_date_dataset "Dataset with ISSUE_DATE annotations"
    ```

1. Manually annotate examples of issue dates

    ```bash
    prodigy ner.manual issue_date_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label ISSUE_DATE
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner issue_date_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach issue_date_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label ISSUE_DATE --unsegmented
   ```
   
1. Create a Prodigy dataset to store annotations of sellers' NIPs

    ```bash
    prodigy dataset nip_seller_dataset "Dataset with NIP_SELLER annotations"
    ```

1. Manually annotate examples of sellers' NIPs

    ```bash
    prodigy ner.manual nip_seller_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label NIP_SELLER
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner nip_seller_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach nip_seller_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label NIP_SELLER --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of buyers' NIPs

    ```bash
    prodigy dataset nip_buyer_dataset "Dataset with NIP_BUYER annotations"
    ```

1. Manually annotate examples of buyers' NIPs

    ```bash
    prodigy ner.manual nip_buyer_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label NIP_BUYER
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner nip_buyer_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach nip_buyer_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label NIP_BUYER --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of bank account numbers

    ```bash
    prodigy dataset bank_account_no_dataset "Dataset with BANK_ACCOUNT_NO annotations"
    ```

1. Manually annotate examples of bank account numbers

    ```bash
    prodigy ner.manual bank_account_no_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label BANK_ACCOUNT_NO
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner bank_account_no_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach bank_account_no_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label BANK_ACCOUNT_NO --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of payment forms

    ```bash
    prodigy dataset payment_form_dataset "Dataset with PAYMENT_FORM annotations"
    ```

1. Manually annotate examples of payment forms

    ```bash
    prodigy ner.manual payment_form_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label PAYMENT_FORM
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner payment_form_dataset pl_core_news_lg 
        --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach payment_form_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label PAYMENT_FORM --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of payment due dates

    ```bash
    prodigy dataset payment_due_dataset "Dataset with PAYMENT_DUE annotations"
    ```

1. Manually annotate examples of payment due dates

    ```bash
    prodigy ner.manual payment_due_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label PAYMENT_DUE
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner payment_due_dataset pl_core_news_lg 
            --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach payment_due_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label PAYMENT_DUE --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of tax rates

    ```bash
    prodigy dataset tax_rate_dataset "Dataset with TAX_RATE_PRODUCT annotations"
    ```

1. Manually annotate examples of tax rates

    ```bash
    prodigy ner.manual tax_rate_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label TAX_RATE_PRODUCT
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner tax_rate_dataset pl_core_news_lg 
            --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach tax_rate_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label TAX_RATE_PRODUCT --unsegmented
   ```

1. Create a Prodigy dataset to store annotations of product measures and quantities

    ```bash
    prodigy dataset measure_quantity_dataset "Dataset with QTY_PRODUCT and MEASURE_PRODUCT annotations"
    ```

1. Manually annotate examples of product measures and quantities

    ```bash
    prodigy ner.manual measure_quantity_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label QTY_PRODUCT,MEASURE_PRODUCT
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner measure_quantity_dataset pl_core_news_lg 
            --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach measure_quantity_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label QTY_PRODUCT,MEASURE_PRODUCT --unsegmented
   ```
   
1. Create a Prodigy dataset to store annotations of product names

    ```bash
    prodigy dataset product_name_dataset "Dataset with PRODUCT_NAME annotations"
    ```

1. Manually annotate examples of product names

    ```bash
    prodigy ner.manual product_name_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt --label PRODUCT_NAME
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner product_name_dataset pl_core_news_lg 
            --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach product_name_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt --label PRODUCT_NAME --unsegmented
   ```
   
1. Create a Prodigy dataset to store annotations of tax, gross and net amounts

    ```bash
    prodigy dataset amounts_dataset "Dataset with amounts annotations"
    ```

1. Manually annotate examples of tax, gross, and net amounts

    ```bash
    prodigy ner.manual amounts_dataset pl_core_news_lg 
        data/processed/ocr/invoices.txt --loader txt 
        --label UNIT_PRICE_PRODUCT,
                NET_AMOUNT_PRODUCT,GROSS_AMOUNT_PRODUCT,TAX_AMOUNT_PRODUCT,
                NET_AMOUNT_TOTAL,GROSS_AMOUNT_TOTAL,TAX_AMOUNT_TOTAL
    ```
   
1. Train the initial NER detection model

    ```bash
    prodigy train ner amounts_dataset pl_core_news_lg 
            --output models/ner_model --n-iter 1000 --dropout 0.25
    ```

1. Add more annotations using model-in-the-loop approach

    ```bash
    prodigy ner.teach amounts_dataset models/ner_model data/processed/ocr/invoices.txt 
        --loader txt 
        --label UNIT_PRICE_PRODUCT,
                NET_AMOUNT_PRODUCT,GROSS_AMOUNT_PRODUCT,TAX_AMOUNT_PRODUCT,
                NET_AMOUNT_TOTAL,GROSS_AMOUNT_TOTAL,TAX_AMOUNT_TOTAL
        --unsegmented
   ```
----
   
1. Merge datasets into a single dataset

    ```bash
   prodigy db-merge invoice_no_dataset,invoice_type_dataset,purchase_date_dataset,
        issue_date_dataset,nip_seller_dataset,nip_buyer_dataset,bank_account_no_dataset,
        payment_form_dataset,payment_due_dataset,tax_rate_dataset,measure_quantity_dataset,
        product_name_dataset,amounts_dataset invoice_ner_dataset
   ```
   
1. Train the initial joint model for invoice NER detection

    ```bash
    prodigy train ner invoice_ner_dataset pl_core_news_lg
        --output models/invoice_initial_ner_model
        --n-iter 1000 --dropout 0.05
    ```
   
1. Manually correct the initial model by running it on a sample of inputs

    ```bash
    prodigy ner.correct invoice_ner_dataset_corrected models/invoice_initial_ner_model
        data/processed/ocr/invoices.txt --loader txt
        --label UNIT_PRICE_PRODUCT,
                NET_AMOUNT_PRODUCT,GROSS_AMOUNT_PRODUCT,TAX_AMOUNT_PRODUCT,
                NET_AMOUNT_TOTAL,GROSS_AMOUNT_TOTAL,TAX_AMOUNT_TOTAL,
                PRODUCT_NAME,QTY_PRODUCT,MEASURE_PRODUCT,TAX_RATE_PRODUCT,
                PAYMENT_DUE,PAYMENT_FORM,BANK_ACCOUNT_NO,NIP_SELLER,NIP_BUYER,
                INVOICE_NO,INVOICE_TYPE,PURCHASE_DATE,ISSUE_DATE
        --exclude invoice_ner_dataset --unsegmented
    ```
   
1. Train the final NER detection model by combining two datasets

    ```bash
    prodigy train ner invoice_ner_dataset,invoice_ner_dataset_corrected pl_core_news_lg 
        --output models/invoice_final_ner_model --n-iter 2000 --dropout 0.05 --eval-split 0.2
    ```
