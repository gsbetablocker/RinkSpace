"""Actual visible contour strokes, separate from model predictions."""
import cv2
import numpy as np


def draw_landmarks(image,row,*,all_candidates=False):
    output=image.copy();height,width=image.shape[:2];counts=dict(detected=0,associated=0,used=0,conflict=0)
    for item in row.get('landmark_observations',[]):
        points=np.asarray(item.get('points_source_px',[]),float).reshape(-1,2)
        if not len(points):continue
        counts['detected']+=1
        counts['associated']+=bool(item.get('selected_model_id'))
        counts['used']+=bool(item.get('used_by_solver'))
        counts['conflict']+=item.get('association_status')=='conflict'
        if not all_candidates and not item.get('used_by_solver'):continue
        size=item.get('source_size',row.get('source_size',[1920,1080]))
        scale=np.array([width/size[0],height/size[1]])
        explicit=item.get('visible_segments_source_px')
        segments=[np.asarray(x,float) for x in explicit] if explicit else [points[a:b] for a,b in item.get('visible_segment_ranges',[[0,len(points)]])]
        color=(30,230,90) if item.get('used_by_solver') else (30,175,245)
        if item.get('association_status')=='conflict':color=(20,110,255)
        for segment in segments:
            if len(segment)>1:cv2.polylines(output,[np.rint(segment*scale).astype(np.int32)],False,color,2,cv2.LINE_AA)
        if item.get('selected_model_id'):
            p=np.rint(points[len(points)//2]*scale).astype(int)
            cv2.putText(output,item['selected_model_id'],tuple(p),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return output,counts
