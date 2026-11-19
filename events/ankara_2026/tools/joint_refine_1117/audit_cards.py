"""Original-image identity cards for the remaining large source residuals."""
import cv2
import numpy as np
from shared import OUT,arrays,read,write
from review import source_frames

CASES=[('nayeon_s05',1201664,66380),('nayeon_s10',2124288,53230),
       ('okada_s04',924160,4827),('nayeon_s04',879360,112165),
       ('nayeon_s06',1544960,205863),('nayeon_s03',612096,45169),
       ('chen_s13',1587200,313093),('jin_s14',1558016,3420)]


def run():
    setup=read(OUT/'prepared/setup.json');folder=OUT/'remaining_source_audit';folder.mkdir(exist_ok=True)
    manifest=[]
    for name,pt,origin in CASES:
        s=setup['shots'][name];a=arrays(OUT/'prepared'/(name+'.npz'))
        j=np.flatnonzero((a['pts']==pt)&(a['origin_row']==origin))[0];path=a['path'][j]
        use=(a['pts']==pt)&(a['path']==path)&(a['fold']<5);uv=a['uv'][use]
        _,raw=next(source_frames(s['source'],[pt]));marked=raw.copy()
        for xy in uv:cv2.drawMarker(marked,tuple(np.rint(xy).astype(int)),(0,0,255),cv2.MARKER_CROSS,9,1)
        cv2.circle(marked,tuple(np.rint(a['uv'][j]).astype(int)),9,(0,220,255),1,cv2.LINE_AA)
        centre=a['uv'][j];lo=np.maximum(np.floor(centre-[220,170]).astype(int),[0,0])
        hi=np.minimum(np.ceil(centre+[220,170]).astype(int),[1920,1080])
        tiles=[cv2.resize(raw,(960,540)),raw[lo[1]:hi[1],lo[0]:hi[0]],marked[lo[1]:hi[1],lo[0]:hi[0]]]
        tiles=[cv2.resize(t,None,fx=540/t.shape[0],fy=540/t.shape[0]) for t in tiles]
        width=sum(t.shape[1] for t in tiles);header=np.full((60,width,3),235,np.uint8)
        cv2.putText(header,f'{name} | {pt/12800:.3f}s | row {origin} | {path}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(header,'Original context | original crop | current source UV (red); selected row circled yellow; NO MODEL',(12,49),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA)
        image=np.vstack((header,np.hstack(tiles)));target=folder/(name+'_'+str(pt)+'.jpg')
        cv2.imwrite(str(target),image,[cv2.IMWRITE_JPEG_QUALITY,97])
        manifest.append(dict(shot=name,pts=pt,origin_row=origin,path=str(path),uv=a['uv'][j].tolist(),
            source=str(s['source']),image=str(target),group_origin_rows=a['origin_row'][use].tolist(),
            source_crop=[*lo.tolist(),*hi.tolist()],measurement='paint width midpoint' if a['family'][j] in ('A1','A2','A3','A4','B1','B2','C','C0') else 'named material contour'))
    write(folder/'manifest.json',manifest)


if __name__=='__main__':run()
