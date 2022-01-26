from synthetic_text_gen import SyntheticWord
import cv2

st = SyntheticWord('../fontScraper/')
font,font_name,fontN,fontN_name = st.getFont()
img,word_new = st.getRenderedText(font,"Test")
cv2.imshow('t',img)
cv2.waitKey()

#print('top {}, bot {}, left {}, right {}'.format(st.maxOverH0,st.maxOverH,st.maxOverW0,st.maxOverW))
