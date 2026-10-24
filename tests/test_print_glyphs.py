import cv2
import numpy as np
from rinkspace.vision_print_glyphs import extract_glyphs,validate_holes,clean_outline


def test_white_letters_and_holes_exclude_colored_background_texture():
    image=np.full((140,280,3),(110,60,130),np.uint8)
    for y in (30,65,100):cv2.line(image,(5,y),(270,y+15),(230,110,100),3)
    cv2.putText(image,'AO',(25,105),cv2.FONT_HERSHEY_SIMPLEX,2.6,(245,245,245),5,cv2.LINE_AA)
    result=extract_glyphs(image,[10,10,270,130],['A','O'])
    assert set(result['paths'])=={'A','O','A_inner','O_inner'}
    validate_holes(result['paths'],['A','O'])
    assert not result['extraction']['background_pattern_used']


def test_straight_bar_and_smooth_round_outline_keep_geometry():
    rectangle=np.array([[0.,0.],[12.,0.],[12.,65.],[0.,65.],[0.,0.]])
    dense=np.concatenate([np.linspace(a,b,200,endpoint=False) for a,b in zip(rectangle[:-1],rectangle[1:])])
    p=clean_outline(dense)
    distance=np.min(np.c_[abs(p[:,0]),abs(p[:,0]-12),abs(p[:,1]),abs(p[:,1]-65)],axis=1)
    assert distance.max()<1e-6
    theta=np.linspace(0,2*np.pi,600)
    ellipse=np.c_[22*np.cos(theta),30*np.sin(theta)]
    q=clean_outline(ellipse)
    assert np.max(abs((q[:,0]/22)**2+(q[:,1]/30)**2-1))<.035
    assert np.array_equal(q[0],q[-1])
