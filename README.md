# Bank-Cheque-OCR
Used computer vision with OCR to extract feilds from bank cheques, thereby automating the process of cheque processing in Banks.

### Run the code

```python
cd scripts
python main.py --input_image ./../cheques/Cheque_6.jpg
```

### Libraries used
- Opencv, PIL ,numpy, os, shutil, google.cloud.vision, keras, regex, imutils, skimage

### OCR 
- Used Google Cloud Vision API to convert image to text.

### Input
- Path to the scanned cheque image

### Output
- Can be customized to extract desired fields from the cheque image

