# 导入工具包
#from imutils import contours
import numpy as np
import argparse
#import cv2
#import myutils
import re
import fitz
from utils import *


pdf_path="D:\\Workspace\\GitHub\\ms-ai\\kaggle\\data\\test\\PDF\\10.1002_2017jc013030.pdf"
doc = fitz.open(pdf_path)
text = ""
for page in doc:
	page_text = page.get_text()
	text += page_text + "\n"

doc.close()
# text = re.sub(r'\s+', ' ', content)
text = text.strip()
text = re.sub(r'\n+', '\n', text)
text = re.sub(r'\\_', '_', text)


text = remove_references_section(text)
min_char_len = 100
max_char_len = 500
overlap_char_len = 50
text_chunks = smart_chunker(text, min_char_len, max_char_len, overlap_char_len)
print(text_chunks)
