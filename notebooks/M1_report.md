---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2

from IPython.core.display import display, HTML
from IPython.display import clear_output

import pandas as pd
import numpy as np

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.options.display.width = 2000
pd.options.display.max_colwidth = 2000

display(HTML("<style>.container { width:75% !important; }</style>"))
```

# M1: raport badawczy

data: 13.02.2019

Zespół badawczy:
  - dr hab. inż. Mikołaj Morzy, prof. PP
  - inż Oliver Pieńkos


W ramach pierwszych trzech miesięcy pracy zespół badawczy zrealizował następujące zadania:

- przygotowanie schematu adnotacji faktur wchodzących w skład "złotego standardu"
- ręczna adnotacja 100 różnych faktur (nabywcy, sprzedawcy, faktury vat, faktury proforma, ...)
- przygotowanie początkowego zestawu filtrów do identyfikacji podstawowych encji występujących na fakturze (data wystawienia, data sprzedaży, numery NIP, numery kont bankowych, numery REGON)
- oprogramowanie systemu OCR (ang. *optical character recognition*) do automatycznego sczytania tekstu występującego na fakturze
- przetworzenie 100 faktur stanowiących "złoty standard" i zaaplikowanie filtrów do oznaczenia istotnych fragmentów tekstu stanowiącego wynik procesu sczytania faktury na podstawie zdjęcia wykonanego telefonem komórkowym
- wytreniowanie 4 modeli uczenia maszynowego (ang. *machine learning*) do automatycznej identyfikacji wybranych encji występujących na fakturze:

  - typ faktury
  - numer faktury
  - NIP sprzedawcy
  - kwota brutto


### Opis procesu adnotacji i czytania zdjęć faktur


Poniżej przedstawiamy przykład zastosowania opracowanych modeli do jednej faktury, a także pokazujemy podsumowanie wyniku uczenia dla faktur stanowiących "złoty standard".


Wyjściowy obraz faktury jest wynikiem fotografii wykonanej przy użyciu telefonu komórkowego. Należy tutaj podkreślić, że jest to obraz który powstał tylko w wersji cyfrowej fotografii, bez dostępu do treści oryginalnego dokumentu (innymi słowy, faktura istnieje tylko w wersji papierowej, nie mamy dostępu np. do dokumentu `*.pdf` który został wydrukowany. Stanowi to istotną różnicę w stosunku do systemów, które operują właśnie na dokumentach, które powstały w wyniku eksportu do formatu `*.pdf`, i w przypadku których ekstrakcja spójnego i poprawnego tekstu jest trywialna.

![nieprzetworzona faktura](m1/00c0c0fcfcfcfcfc.png)


Pierwszym krokiem było pzygotowanie modułu OCR i wykonanie początkowego sczytania tekstu. Niestety, próby przeprowadzone na trzech wybranych systemach (Google Tesseract, OCR East, Firebase OCR) nie przyniosły zadowalających rezultatów. Tekst na fakturze był czytany fragmentarycznie, co powodowało, że wyjściowy strumień tekstu stanowił chaotycznyh strumień (np. fragmenty numerów kont bankowych były losowo przemieszane).

Poniżej przedstawiamy wynik zaznaczenia fragmentów na fakturze przed przeprowadzeniem dodatkowych prac badawczych.

![przed grupowaniem](m1/beforeGrouping.png)


Jak widać, wiele fragmentów faktury jest rozpoznawanych przez dużą liczbę wzajemnie nakładających się prostokątów, co powoduje niemożność analizy tekstu stanowiącego wynik działania systemu OCR


Analiza tego stanu rzeczy przekonała nas, że konieczne jest przepisanie fragmentu systemu do rozpoznawania tekstu ze zdjęć w taki sposób, aby ściśle przylegające do siebie fragmenty łączyły się w jeden spójny obszar, dzięki czemu unikniemy dzielenia semantycznie jednorodnych fragmentów tekstu. Po przeprowadzeniu prac sposób identyfikacji tekstu na fakturach uległ znaczącej poprawie, poniżej przedstawiamy tę samą fakturę po przetworzeniu za pomocą poprawionego systemu OCR


![po poprawkach OCR](m1/after.png)


Jak widać, każda linia tekstu jest rozpoznawana w sposób ciągły. Poniżej tekstowa reprezentacja tej faktury w formie stanowiącej bezpośredni surowy wynik działania systemu OCR.

```python
!cat m1/00c0c0fcfcfcfcfc.txt
```

Taki surowy strumień tekstu jest następnie przez nas poddawany filtrowaniu przez zbiór filtrów, które próbują zidentyfikować w tekście istotne elementy, takie jak:

- numer konta bankowego
- numer NIP
- numer REGON
- kod pocztowy
- fragmenty tekstu wyglądające jak daty
- fragmenty tekstu wyglądające jak kwoty pieniężne
- fragmenty tekstu sugerujące wartość stawki podatku

Poniżej przedstawiamy tę samą fakturę, po dodatkowym wzbogaceniu tekstu z OCR o znaczniki wskazujące na możliwe wystąpienia powyższych encji w tekście. Istotne jest, że nie polegamy na danych adnotowanych przez filtry, ponieważ filtry mają charakter regułowy (przykładowo, kod pocztowy jest rozpoznawany jako sekwencja `dd-ddd` gdzie `d` jest dowolną cyfrą, NIP jest rozpoznawany jako jeden ze wzorców `ddd-dd-dd-ddd`, `ddd-ddd-dd-dd`, lub `dddddddddd`, z ew. przedrostkiem reprezentującym kraj). Znaczniki wstawione w poniższym tekście stanowią jedynie wzbogacenie, wzmocnienie sygnału dla modułu uczenia maszynowego, za pomocą których model statystyczny szybciej uczy się rozpoznawać fragmenty tekstu reprezentujące te i inne encje.

```python
!cat m1/00c0c0fcfcfcfcfc-annotated.txt
```

### Opis procesu uczenia maszynowego


W dalszej części raportu prezentujemy fragmenty zbioru uczącego (ang. *training set*) składającego się ze 100 anotowanych i ręcznie opisanych faktur oraz prezentujemy uzyskane wyniki.

```python
import spacy
import pandas as pd

df = pd.read_csv('../data/annotations/invoice_data.csv')

df['id'] = df.filename.apply(lambda x: x.split('.')[0])
df['ocr'] = df.id.apply(lambda x: ''.join(open(f'../data/raw_ocr/{x}.txt','r').readlines()))
```

Dla każdej faktury dokonujemy adnotacji następujących pól:

- typ faktury
- numer faktury
- numer konta sprzedawcy
- kwota brutto
- stawka VAT
- NIP sprzedawcy
- nazwa sprzedawcy

```python
columns= ['typ_faktury','numer_faktury','numer_konta','stawka_VAT','sprzedawca_nip','kwota_brutto','sprzedawca','ocr']

df[columns][5:10]
```

W kolejnym kroku tworzymy definicję modelu uczenia maszynowego, które na podstawie tekstu z systemu OCR przewiduje wartość atrybutu `typ_faktury`

```python
df[['ocr','numer_konta']].to_csv('train.csv', header=['doc_txt','class'], index=False)
```

```python
%%writefile model_definition.yaml

input_features:
    -
        name: doc_txt
        type: text
        encoder: cnnrnn
        cell_type: lstm
        level: char

output_features:
    -
        name: class
        type: category
            
training:
    learning_rate: 0.001
    early_stop: 10
    batch_size: 5
    learning_rate_warmup_epochs: 3
    validation_measure: accuracy

```

```python
!ludwig train --data_csv train.csv --model_definition_file model_definition.yaml > m1/typ_faktury_train.txt
```

```python
!cat m1/typ_faktury_train.txt
```

Powtórzyliśmy proces uczenia maszynowego w celu wytreniowania pozostałych modeli, w celu oszczędzenia miejsca raportujemy wyniki ostatniej epoki uczenia dla rozpoznawania kolejnych encji.


#### NIP sprzedawcy

```python
!cat m1/sprzedawca_nip_train.txt
```

#### Kwota brutto

```python
!cat m1/kwota_brutto_train.txt
```

#### Numer faktury

```python
!cat m1/numer_faktury_train.txt
```

#### Numer konta sprzedawcy

```python
!cat m1/numer_konta_train.txt
```

## Podsumowanie


Zadania badawcze zdefiniowane w ramach kamienia milowego M1 zostały zrealizowane. Dokonano ręcznej adnotacji 100 faktur i uruchomiono pierwszy zestaw modeli uczenia maszynowego wytrenowanych na tym zbiorze. Dla wybranych encji uzyskano następującą dokładność:

- typ faktury: 77%
- NIP sprzedawcy: 80%
- kwota brutto: 76%
- numer faktury: 79%
- numer konta: 81%

Oczywiście należy podkreślić, że są to wyniki wstępne. Dane wejściowe zostaną poddane jeszcze dokładniejszej adnotacji, dodamy także więcej filtrów identyfikujących ważne elementy faktury, modele ML staną się też bardziej stabilne wraz ze wzrostem liczby przetworzonych faktur.

W najbliższym czasie planujemy realizację następujących kroków:

- dodanie filtrów rozpoznających:
  - kwoty wpisane ręcznie
  - nazwy ulic i miast
  - liczby sztuk
  - numery NIP spoza Polski
  - faktury zagraniczne
- poprawienie systemu OCR w taki sposób, aby łączyć prostokąty opakowujące (ang. *bounding box*) także w pionie, w celu identyfikacji danych wpisanych do faktury w formacie tabelarycznym
- rozszerzenie informacji ręcznie adnotowanych w "złotym standardzie" o wszystkie cechy faktury zdefiniowane w schemacie adnotacji
- zasilenie zbioru uczącego fakturami spoza "złotego standardu" (czyli fakturami nieposiadającymi ręcznej adnotacji) i przetestowanie precyzji modeli na takich fakturach
- rozpoczęcie eksperymentów z paragonami

```python

```
