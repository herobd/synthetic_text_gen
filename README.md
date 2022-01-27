# Synthetic Text Generation

Brian Davis 2022

This is designed to render text to an image using random fonts. The primary use is to generate words using SyntheticWord.

It requires a dataset of fonts (which it will sample from randomly using getFont()). I used all the free for-commercial-use fonts from 1001fonts.com and my scaping tool and preperation scripts are below. This gives about 10,000 fonts.

If 1001fonts.com changes and the scraping doesn't work, and easy place to get free fonts is https://github.com/google/fonts which has ~1000 fonts. You'll just need to create  the `fonts.list` file for them (or another set), which is just a text file with the path to each font on their own line. You can then use the rest of the preperation scrips as below.


# Scrape 1001fonts.com

`scrape.py` is a tool which scrapes fonts from 1001fonts, particularly we're grabing the free for commercial use fonts.

Usage:python scrape.py [TYPE (serif,handwritten,etc)] [NUMPAGES (number of result pages, chech website)]

Steps to set up fonts as expected by synthetic_text_gen:

Python2
```
#Dependency
pip install wget


mkdir all_unzipped
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

#This next part is only if doing distillation (and it requires pytesseract)
#It gets a list of easy-to-read fonts (as determined using tesseract)
python find_good_fonts.py PATH/TO/FINAL_DIR/
python cnvert_scored_fonts_to_csv.py PATH/TO/FINAL_DIR/ scored_fonts*.csv
```
