Brian Davis 2019

This generates random text line images, intended for CycleGAN adaption to an unlabeled target dataset.


# Scrape 1001fonts.com

`scrape.py` is a tool which scrapes fonts from 1001fonts, particularly we're grabing the free for commercial use fonts.

Usage:python scrape.py [TYPE] [NUMPAGES]

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

python clean.py PATH/TO/FINAL_DIR/ #find out which can render numbers and have lower case
python find_good_fonts.py PATH/TO/FINAL_DIR/
python cnvert_scored_fonts_to_csv.py PATH/TO/FINAL_DIR/ scored_fonts*.csv
```
