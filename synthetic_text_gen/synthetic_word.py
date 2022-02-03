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
                if clear:
                    self.fonts = [(path,True,True,True) for path,case,num in reader]
                    self.bracket_fonts = self.fonts
                else:
                    self.fonts = [(path,case!='False',num!='False',bracket!='False') for path,case,num,bracket in reader]
                    self.bracket_fonts = [font for font in self.fonts if font[3]]
            self.fonts=self.fonts[1:] #discard header row

        else:
            with open(os.path.join(font_dir,'fonts.list')) as f:
                self.fonts = f.read().splitlines()

    def getTestFontImages(self,start):
        ret=[]
        texts = ['abcdefg','hijklmn','opqrst','uvwxyz','12345','67890','ABCDEFG','HIJKLMN','OPQRST','UVWXYZ']
        these_fonts = self.fonts[start:start+1000]
        for index, (filename,hasLower,hasNums,hasBracket) in enumerate(these_fonts):
            if hasNums and hasLower:
                print('rendering {}/{}'.format(index+start,len(these_fonts)+start),end='\r')
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100)
                minY,maxY = self.getRenderedText(font,'Tlygj|') 
                font = (font,minY,maxY,True)
                bad=False
                images=[]
                for s in texts:
                    image,text = self.getRenderedText(font,s)
                    if image is None:
                        bad=True
                        break
                    images.append((s,image))
                if bad:
                    continue
                ret.append((index+start,filename,images))
        return ret

    def getFont(self,target=None):
        while True:
            if target is None:
                index = np.random.choice(len(self.fonts))
                filename, hasLower, hasNums, hasBrackets = self.fonts[index]
            else:
                for filename,hasLower,hasNums,hasBrackets in self.fonts:
                    if filename==target:
                        break
            try:
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100) 

                minY,maxY = self.getRenderedText(font,'Tlygj|)]')
                fontR = (font,minY,maxY,hasLower,hasBrackets)
                break
            except OSError:
                pass
        if hasNums:
            fontNumsR = fontR
            filenameNums = filename
        else:
            while True:
                indexNums = np.random.choice(len(self.fonts))
                filenameNums, hasLower, hasNums, hasBrackets = self.fonts[indexNums]
                if hasNums:
                    try:
                        fontNums = ImageFont.truetype(os.path.join(self.font_dir,filenameNums), 100) 
                        minY,maxY = self.getRenderedText(fontNums,'Tlygj|1)]')
                        fontNumsR = (fontNums,minY,maxY,hasLower,hasBrackets)
                        break
                    except OSError:
                        pass
        return (fontR,filename,fontNumsR,filenameNums)
    def getBracketFont(self):
        while True:
            index = np.random.choice(len(self.bracket_fonts))
            filename, hasLower, hasNums, hasBrackets = self.fonts[index]
            try:
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100) 

                minY,maxY = self.getRenderedText(font,'Tlygj|)]')
                fontR = (font,minY,maxY,hasLower,hasBrackets)
                break
            except OSError:
                pass
        return fontR

    def getRenderedText(self,fontP,text,ink=0.99):
        if isinstance(fontP, tuple):
            font,minY,maxY,hasLower,hasBrackets = fontP
            if not hasLower:
                text=text.upper()
            if not hasBrackets :
                if '(' in text:
                    text= text.replace('(','')
                if ')' in text:
                    text= text.replace(')','')
                if '[' in text:
                    text= text.replace('[','')
                if ']' in text:
                    text= text.replace(']','')

        else:
            font = fontP
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
                return np_image[minY:maxY+1,minX:maxX+1],text
            else:
                #print('uhoh, blank image, what do I do?')
                return None,None
        return None,None
    
    def getBrackets(self,fontP=None,paren=True):
        if fontP is not None:
            font,minY,maxY,hasLower,hasBrackets = fontP
        else:
            hasBrackets = False
        if not hasBrackets:
            fontP = self.getBracketFont()
        open_img,_ = self.getRenderedText(fontP,'(' if paren else '[') 
        close_img,_ = self.getRenderedText(fontP,')' if paren else ']') 
        if open_img is None or close_img is None:
            return self.getBrackets(paren=paren)
        return open_img,close_img



if __name__ == "__main__":
    font_dir = sys.argv[1]
    text = sys.argv[2]
    sw = SyntheticWord(font_dir)
    font,name,fontN,nameN = sw.getFont('text_fonts/Kabrio by Zetafonts/Kabrio-Alternate-Regular-trial.ttf')
    for text in [text]:#,text+'y',text+'t']:
        if re.match('\d',text):
            im,text = sw.getRenderedText(fontN,text)
        else:
            im,text = sw.getRenderedText(font,text)
        print(text)
        cv2.imshow('x',im)
        cv2.show()
