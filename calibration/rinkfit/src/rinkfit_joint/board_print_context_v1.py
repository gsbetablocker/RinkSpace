"""Authenticated printed board context from existing native source templates.

The inherited programme SIFT reader locates an entity. Its transformed body
polygon only carries source ownership; material means are reread separately.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .io_utils import read_json,file_sha256
from rinkspace.vision_programme_board_review import BoardProgrammeSourceReader,_root_sift

class NativePrintedBoardContext:
    def __init__(self,feature_paths,*,source_sha256,shot_id):
        self.shot_id=shot_id;self.source_sha256=source_sha256;self.refs=[]
        self.reader=BoardProgrammeSourceReader([],source_sha256=source_sha256,time_base='1/12800',device='cpu')
        self.refs.append(self.ref(Path(__file__)))
        import rinkspace.vision_programme_board_review as old
        self.refs.append(self.ref(Path(old.__file__)))
        for feature_path in feature_paths:
            q=read_json(feature_path);s=q['source'];p=Path(s['image_path']);m=Path(s['mask_path'])
            if s['source_video_sha256']!=source_sha256 or s['time_base']!='1/12800' or str(s['shot_id'])!=shot_id.removeprefix('s0'):
                raise ValueError('Printed source template belongs to another source/shot')
            if file_sha256(p)!=s['png_sha256'] or file_sha256(m)!=s['mask_sha256']:raise ValueError('Original print/mask changed')
            image=cv2.imread(str(p));mask=cv2.imread(str(m),0)>0
            if hashlib.sha256(image.tobytes()).hexdigest()!=s['decoded_BGR_sha256']:raise ValueError('Original print BGR changed')
            self.refs.extend(self.ref(x)for x in(Path(feature_path),p,m))
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            byid={f['id']:f for f in q['features']}
            numbers=sorted({int(k.split('.')[0][2:])for k in byid if k.startswith('Ad')and k.endswith('.top_edge')})
            for number in numbers:
                top=byid.get('Ad'+str(number)+'.top_edge');bottom=byid.get('Ad'+str(number)+'.bottom_edge')
                if not top or not bottom:continue
                uv=np.concatenate([np.asarray(z['uv'],float)for f in(top,bottom)for z in f['segments']]);lo=uv.min(0);hi=uv.max(0)
                poly=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],np.float32)
                region=np.zeros(mask.shape,np.uint8);cv2.fillConvexPoly(region,np.rint(poly).astype(np.int32),255);region[mask]=0
                keys,descs=self.reader.sift.detectAndCompute(gray,region)
                if descs is None or len(keys)<8:continue
                xy=np.asarray([k.pt for k in keys],np.float32)
                cap=np.concatenate([np.asarray(z.get('source_native_readout_uv',z['uv']),float)for f in q['features']if f['id'].startswith('board_top.')for z in f['segments']])
                cap=cap[(cap[:,0]>=lo[0])&(cap[:,0]<=hi[0])]
                anchor=dict(source_cap_navigation_uv=cap.tolist(),anchor_id='source_print_context/'+str(s['native_pts'])+'/Ad'+str(number),physical_material_id='Ad'+str(number)+'.body_context',advertisement_number=number,shot_id=shot_id,curve=dict(kind='advertisement_boundary'),edge='body_context',source_entity_context_uv=poly.tolist(),source_context_feature=self.ref(Path(feature_path)),source_native_pts=s['native_pts'],source_BGR_sha256=s['decoded_BGR_sha256'])
                self.reader._install_template(anchor,image,poly,xy,descs)
        if not self.reader.templates:raise ValueError('No authenticated existing printed board template')
    @staticmethod
    def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
    def read(self,rgb,mask,receipt,*,navigation_roi):
        if(receipt['source_video_sha256']!=self.source_sha256 or receipt['shot_id']!=self.shot_id or receipt['time_base']!='1/12800' or hashlib.sha256(rgb.tobytes()).hexdigest()!=receipt['decoded_native_RGB_sha256']):raise ValueError('Current print context source identity differs')
        x0,y0,x1,y1=map(int,navigation_roi);roi=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
        bands=[dict(blue_upper_uv=roi[:2],yellow_lower_uv=roi[[3,2]])]
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        found=self.reader._template_matches(gray,mask,bands,self.shot_id)
        # The existing CPU reader leaves cross-instance uniqueness diagnostic
        # unset. Recompute the SAME target descriptors and compare all named
        # source banks explicitly, before granting printed-context ownership.
        region=np.zeros(mask.shape,np.uint8);cv2.fillPoly(region,[roi.astype(np.int32)],255);region[mask]=0
        ys,xs=np.nonzero(region)
        if not len(xs):return []
        lo=np.array([xs.min(),ys.min()]);hi=np.array([xs.max()+1,ys.max()+1]);keys,descs=self.reader.sift.detectAndCompute(gray[lo[1]:hi[1],lo[0]:hi[0]],region[lo[1]:hi[1],lo[0]:hi[0]])
        if descs is None:return []
        xy=np.asarray([k.pt for k in keys],np.float32)+lo;descs=_root_sift(descs);dist={}
        for n in sorted({t['anchor']['advertisement_number']for t in self.reader.templates}):
            bank=np.concatenate([t['desc']for t in self.reader.templates if t['anchor']['advertisement_number']==n]);dist[n]=np.array([z.distance for z in self.reader.bf.match(descs,bank)])
        output=[]
        for name,r in found.items():
            a=r['template']['anchor'];n=a['advertisement_number'];ids=np.array(sorted(r['matched_target_indices']),int);other=[d[ids]for k,d in dist.items()if k!=n]
            unique=dist[n][ids]<.76*np.min(other,axis=0)if other else np.zeros(len(ids),bool);support=xy[ids[unique]]
            if len(support)<8 or np.ptp(support,axis=0).min()<8 or np.ptp(support,axis=0).max()<40:continue
            mapped_polygon=np.asarray(r['source_entity_context_uv'],np.float32)
            carrier=cv2.getPerspectiveTransform(np.asarray(a['source_entity_context_uv'],np.float32),mapped_polygon)
            source_cap=np.asarray(a['source_cap_navigation_uv'],np.float32).reshape(-1,2)
            current_cap=cv2.perspectiveTransform(source_cap[None],carrier)[0]if len(source_cap)else source_cap
            output.append(dict(current_source_cap_search_navigation_uv=current_cap.tolist(),schema='rinkfit.current_named_print_body_context.v1',native_pts=receipt['native_pts'],shot_id=self.shot_id,actual_native_RGB_sha256=receipt['decoded_native_RGB_sha256'],physical_board_instance='Ad'+str(n),current_source_entity_identity_confirmed=True,current_source_print_body_polygon_uv=np.asarray(r['source_entity_context_uv']).tolist(),current_matched_static_print_pixels_uv=np.asarray(r['support_xy']).tolist(),current_discriminating_context_pixels_uv=support.tolist(),identity_matches=r['identity_matches'],source_anchor_id=a['anchor_id'],source_seed_native_pts=a['source_native_pts'],source_context_references=self.refs,source_template_operator='existing BoardProgrammeSourceReader._template_matches plus actual CPU cross-instance RootSIFT comparison',model_or_residual_used_for_identity=False,polygon_is_source_to_source_ownership_context_only=True,transformed_coordinates_are_not_boundary_measurements=True,source_foot_boundary_qualified_by_this_context=False))
        return output
