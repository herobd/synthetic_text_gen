from PIL import ImageFont, ImageDraw, Image
from . import img_f as cv2
#import img_f as cv2
import numpy as np
import skimage
import string
import random
import os,sys
import math,re,csv
import timeit

#import pyvips

#https://stackoverflow.com/a/47269413/1018830
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

class SyntheticWord:

    def __init__(self,font_dir,clean=True,clear=False):
        self.font_dir = font_dir
        if clean or clear:
            if clear:
                csv_file = 'clear_fonts.csv'
            else:
                csv_file = 'clean_fonts.csv'
            with open(os.path.join(font_dir,csv_file)) as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                self.fonts = [row for row in reader]
            self.fonts=self.fonts[1:] #discard header row
        else:
            with open(os.path.join(font_dir,'fonts.list')) as f:
                self.fonts = f.read().splitlines()

    def getTestFontImages(self,start):
        ret=[]
        texts = ['abcdefg','hijklmn','opqrst','uvwxyz','12345','67890','ABCDEFG','HIJKLMN','OPQRST','UVWXYZ']
        these_fonts = self.fonts[start:start+1000]
        for index, (filename,hasLower,hasNums) in enumerate(these_fonts):
            if hasNums and hasLower:
                print('rendering {}/{}'.format(index+start,len(these_fonts)+start),end='\r')
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100)
                minY,maxY = self.getRenderedText(font,'Tlygj|') 
                font = (font,minY,maxY)
                bad=False
                images=[]
                for s in texts:
                    image = self.getRenderedText(font,s)
                    if image is None:
                        bad=True
                        break
                    images.append((s,image))
                if bad:
                    continue
                ret.append((index+start,filename,images))
        return ret

    def getFont(self):
        while True:
            index = np.random.choice(len(self.fonts))
            filename, hasLower, hasNums = self.fonts[index]
            try:
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100) 

                minY,maxY = self.getRenderedText(font,'Tlygj|')
                font = (font,minY,maxY)
                break
            except OSError:
                pass
        if hasNums:
            fontNums = font
            filenameNums = filename
        else:
            while True:
                indexNums = np.random.choice(len(self.fonts))
                filenameNums, hasLower, hasNums = self.fonts[indexNums]
                if hasNums:
                    try:
                        fontNums = ImageFont.truetype(os.path.join(self.font_dir,filenameNums), 100) 
                        minY,maxY = self.getRenderedText(font,'Tlygj|1')
                        font = (font,minY,maxY)
                        break
                    except OSError:
                        pass
        return (font,filename,fontNums,filenameNums)

    def getRenderedText(self,font,text,ink=0.99):
        if type(font) is tuple:
            font,minY,maxY = font
        else:
            minY=None
            maxY=None

        for retry in range(7):

            #create big canvas as it's hard to predict how large font will render
            size=(250+190*max(2,len(text))+200*retry,920+200*retry)
            image = Image.new(mode='L', size=size)

            draw = ImageDraw.Draw(image)
            try:
                draw.text((400, 250), text, font=font,fill=1)
            except OSError:
                print('ERROR: failed generating text "{}"'.format(text))
                continue
            np_image = np.array(image)

            horzP = np.max(np_image,axis=0)
            minX=first_nonzero(horzP,0)
            maxX=last_nonzero(horzP,0)
            if minY is None:
                vertP = np.max(np_image,axis=1)
                minY=first_nonzero(vertP,0)
                maxY=last_nonzero(vertP,0)
                return minY,maxY

            #print('minY: {}'.format(minY))


            if (minX<maxX and minY<maxY):
                #print('original {}'.format(np_image.shape))
                #return np_image,new_text,minX,maxX,minY,maxY,font, f_index,ink
                return np_image[minY:maxY+1,minX:maxX+1]
            else:
                #print('uhoh, blank image, what do I do?')
                return None
        return None


if __name__ == "__main__":
    font_dir = sys.argv[1]
    text = sys.argv[2]
    sw = SyntheticWord(font_dir)
    font,name,fontN,nameN = sw.getFont()
    for text in [text,text+'y',text+'t']:
        if re.match('\d',text):
            im = sw.getRenderedText(fontN,text)
        else:
            im = sw.getRenderedText(font,text)

        cv2.imshow('x',im)
        cv2.show()
