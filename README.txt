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
mkdir fonts
#get only font files
find all_unzipped -name \*.ttf -exec mv -t fonts {} +
find all_unzipped -name \*.TTF -exec mv -t fonts {} +
find all_unzipped -name \*.otf -exec mv -t fonts {} +
find all_unzipped -name \*.woff -exec mv -t fonts {} +
rm -r all_unzipped
ls fonts/* > fonts.list

python clean.py fonts #find out which can render numbers and have lower case
```
