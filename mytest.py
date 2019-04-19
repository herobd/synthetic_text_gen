from synthetic_text_gen import SyntheticText
import cv2

st = SyntheticText('../data/fonts/handwritten_fonts','../data/OANC_text',line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=2.5,text_len=40,use_warp=0.7,warp_std=[1,4], warp_intr=[12,100])

for i in range(30):
    image,text= st.getSample()
    minV = image.min()
    maxV = image.max()
    print('{} min:{}, max:{}, height:{},  text:{}'.format(i,minV,maxV,image.shape[0],text))
    fn = 'test/{}.png'.format(i)
    cv2.imwrite(fn,255*image)

#print('top {}, bot {}, left {}, right {}'.format(st.maxOverH0,st.maxOverH,st.maxOverW0,st.maxOverW))
