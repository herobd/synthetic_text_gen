# Synthetic Text Generation

Brian Davis (2022)

Just run: `python setup.py install`

This is a python module designed to render text to an image using random fonts. The primary use is to generate words using SyntheticWord.

It requires a dataset of fonts (which it will sample from randomly using getFont()). I used all the free-for-commercial-use fonts from 1001fonts.com and my scaping and preperation scripts are described below. This gives about 10,000 fonts.

If 1001fonts.com changes and the scraping doesn't work, an easy place to get free fonts is https://github.com/google/fonts which has ~1000 fonts. You'll just need to create  the `fonts.list` file for them (or another set), which is just a text file with the path to each font on their own line. You can then use the rest of the preperation scripts as below.

## Use
Here's the example in `mytest.py`
```
from synthetic_text_gen import SyntheticWord
import cv2

st = SyntheticWord('../fontsScraped/')
font,font_name,fontN,fontN_name = st.getFont()
img,word_new = st.getRenderedText(font,"Test")
cv2.imshow('t',img)
cv2.waitKey()
```

`word_new` will be the same text provided to `getRenderedText`, but may be converted to uppercase if the font only has uppercase.
SyntheticWord is intended to only work with words (not long strings).

# Scrape 1001fonts.com

(These scripts expect Python2, sorry.)

`scrape.py` is a tool which scrapes fonts from 1001fonts, particularly we're grabing the free for commercial use fonts.

Usage: `python scrape.py [TYPE (serif,handwritten,free-for-commercial-use,etc)] [NUMPAGES (number of result pages, check website)]`

`clean.py` is a script which goes through each font and renders a few characters to test what character sets a font has (numbers, upper/lower case, a few punctuation marks) and saves the results (this resulting file is what is used by SyntheticWord.

Steps to set up fonts as used in Dessurt (https://arxiv.org/abs/2203.16618 / https://github.com/herobd/dessurt):
```
#Dependency
pip install wget


mkdir all_unzipped #hard-coded directory name for scrape.py
python scrape.py free-for-commercial-use 506 #scrape and download
rm *.zip
mkdir PATH/TO/FINAL_DIR/fonts
#get only font files
find all_unzipped -name \*.ttf -exec mv -t PATH/TO/FINAL_DIR/fonts {} +
find all_unzipped -name \*.TTF -exec mv -t PATH/TO/FINAL_DIR/fonts {} +
find all_unzipped -name \*.otf -exec mv -t PATH/TO/FINAL_DIR/fonts {} +
find all_unzipped -name \*.woff -exec mv -t PATH/TO/FINAL_DIR/fonts {} +
rm -r all_unzipped
ls fonts/* > PATH/TO/FINAL_DIR/fonts.list

#find out which fonts can render numbers and have lower case
# by manually rendering out text with every font.
python clean.py PATH/TO/FINAL_DIR/ 

#This next part is only needed if doing distillation (and it requires pytesseract)
#It gets a list of easy-to-read fonts (as determined using tesseract)
python find_good_fonts.py PATH/TO/FINAL_DIR/
python cnvert_scored_fonts_to_csv.py PATH/TO/FINAL_DIR/ scored_fonts*.csv
```
