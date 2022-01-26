import urllib2
import re
import sys
import wget
import zipfile
import os

'''This script is for scraping fonts of a specific stype from 1001fonts.com
    I don't know if there's a way to only get free-for-commercial-use only,
    currently, it gets everything.
    Usage: type-keyword num-pages count-per-page(10) 
'''

url ='http://www.1001fonts.com/{}-fonts.html?page={}&items={}' #tag,page,num on page
url_all ='http://www.1001fonts.com/?page={}&items={}' #tag,page,num on page

def getFonts(typ,page,num,commercial=False):
    downloadURLs=[]
    try:
        if typ=='all':
            wordurl=url_all.format(page,num)
        else:
            wordurl=url.format(typ,page,num)
        print(wordurl)
        #page = urllib2.urlopen(wordurl)
        #pp = page.read()
        #lines = pp.split('\n')
        filename = wget.download(wordurl)
        with open(filename) as f:
            pp=f.read()
        if commercial:
            ms = re.findall('font-toolbar-wrapper.+?commercial use.+?href=([^"]*/[^/"]*\.zip)',pp)
            if len(ms)==0:
                ms = re.findall('font-toolbar-wrapper.+?commercial use.+?href="([^"]*/[^/"]*\.zip)"',pp)
            #import pdb;pdb.set_trace()
        else:
            ms = re.findall('href=([^"]*/[^/"]*\.zip)',pp) #newline dependent
        for m in ms:
            #zipUrls.append(m[1])
            #zips.append(m[2])

            downloadURL = 'http://www.1001fonts.com'+m
            downloadURLs.append(downloadURL)

        os.remove(filename)

        #return zipUrls,zips
    except urllib2.HTTPError:
        print 'failed for : '+urlS
    for downloadURL in downloadURLs:
        try:
            filename = wget.download(downloadURL)
            print(filename)
            with zipfile.ZipFile(filename,"r") as zip_ref:
                zip_ref.extractall("all_unzipped")
        except Exception as e:
            print(e)
            print('Couldnt download/unzip '+downloadURL)
    #return [],[]

typ = sys.argv[1]
count = int(sys.argv[2])
per = 10#sys.argv[3] if len(sys.argv)>3 else 10
start = int(sys.argv[3]) if len(sys.argv)>3 else 0

for i in range(start,count):
    print 'NEW PAGE ({}/{})'.format(i+1,count)
    getFonts(typ,i+1,per,True)
