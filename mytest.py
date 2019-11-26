from synthetic_text_gen import SyntheticText
import cv2

st = SyntheticText('../data/fonts/handwritten_fonts','../data/OANC_text',line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=2.5,text_len=40,use_warp=0.7,warp_std=[1,4], warp_intr=[12,100])

fonts = st.getFonts()

st.changeFontProb(5,.1)
print('{}, {}, {}, {}, {}'.format(st.fontProbs[1],st.fontProbs[2],st.fontProbs[3],st.fontProbs[4],st.fontProbs[5]))
st.changeFontProb(4,1)
print('{}, {}, {}, {}, {}'.format(st.fontProbs[1],st.fontProbs[2],st.fontProbs[3],st.fontProbs[4],st.fontProbs[5]))
st.changeFontProb(3,10)
print('{}, {}, {}, {}, {}'.format(st.fontProbs[1],st.fontProbs[2],st.fontProbs[3],st.fontProbs[4],st.fontProbs[5]))
#st.changeFontProb(2,-1)
#print('{}, {}, {}, {}, {}'.format(st.fontProbs[1],st.fontProbs[2],st.fontProbs[3],st.fontProbs[4],st.fontProbs[5]))
#for i in range(30):
    #image,text,f_index= st.getSample()
#for font in ['Kids Book/Kids Book Italic.ttf']:
for f_index in [594,784,64]: #range(30):
    i=f_index    
    text='best'
    font = fonts[f_index]
    print(font)
    image =st.getFixedSample(text,f_index)
    minV = image.min()
    maxV = image.max()
    print('{} font[{}]:{},  text:{}'.format(i,f_index,fonts[f_index],text))
    fn = 'test/{}.png'.format(font[:5])
    #fn = 'test/{}.png'.format(i)
    cv2.imwrite(fn,255*image)

#print('top {}, bot {}, left {}, right {}'.format(st.maxOverH0,st.maxOverH,st.maxOverW0,st.maxOverW))
