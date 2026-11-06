"""Read finite directional steel-edge likelihoods from original source pixels.

Profiles stay inside the original annotation band. They add image information
within that band without turning its centre into a measured material point.
No body/model output is used to choose a pixel peak or camera/time alignment.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np
from .fullbody_round import write_json

def attach(document,source_sequence,output):
    import cv2
    from rinkspace.viewer3d.source import SourceVideo
    doc=copy.deepcopy(document);sequence=json.loads(Path(source_sequence).read_text(encoding='utf-8'));reader=SourceVideo(sequence['source_video']['path'],(1,12800));cache={};count=0;frames=set()
    for row in doc['full_source_skate_structures']['rows']:
        if row.get('source_structure_feature')!='runner_lower_edge'or row['structure_feature']!='runner_lower_edge':continue
        pts=int(row['pts']);points=np.asarray(row['polyline_uv'],float);radius=float(row['pixel_uncertainty_radius_px'])
        if pts not in cache:
            rgb=reader.read(pts);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)/255.;smooth=cv2.GaussianBlur(gray,(0,0),.7);gx=cv2.Sobel(smooth,cv2.CV_64F,1,0,ksize=3)/8;gy=cv2.Sobel(smooth,cv2.CV_64F,0,1,ksize=3)/8;cache[pts]=(gx,gy,hashlib.sha256(rgb.tobytes()).hexdigest())
        gx,gy,frame_hash=cache[pts];tangent=np.gradient(points,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1),1e-12)[:,None];normals=np.c_[-tangent[:,1],tangent[:,0]];normals[normals[:,1]<0]*=-1;offset=np.linspace(-radius,radius,int(np.ceil(8*radius))+1);profiles=[]
        for point,normal,tan in zip(points,normals,tangent):
            samples=point+offset[:,None]*normal;values=[]
            for along in [-.6,0.,.6]:
                query=samples+along*tan;xx=query[:,0].astype(np.float32)[None];yy=query[:,1].astype(np.float32)[None];a=cv2.remap(gx.astype(np.float32),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0];b=cv2.remap(gy.astype(np.float32),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0];values.append(a*normal[0]+b*normal[1])
            signed=np.mean(values,axis=0);strength=np.maximum(signed,0);maximum=float(strength.max());noise=max(float(np.median(abs(signed))),1/255/8);reliability=float(np.clip((maximum-noise)/(maximum+noise),0,1));peaks=np.flatnonzero((strength[1:-1]>=strength[:-2])&(strength[1:-1]>=strength[2:]))+1
            profiles.append(dict(normal=normal.tolist(),offset_px=offset.tolist(),positive_native_image_gradient=strength.tolist(),noise_floor=max(noise,.05*maximum),maximum_gradient=maximum,reliability=reliability,peak_offsets_px=offset[peaks].tolist(),all_peaks_preserved=True,contact_material='unknown'))
        row['source_pixel_edge_profiles']=profiles;row['source_pixel_edge_budget_share']=.5;row['source_pixel_profile_provenance']=dict(pts=pts,source_rgb_sha256=frame_hash,source_only=True,model_used=False,smoothing_sigma_px=.7,normal_direction='toward lower native image side of the source lower-steel polyline',original_radius_px=radius,source_radius_unchanged=True)
        count+=len(profiles);frames.add(pts)
    reader.close();doc['source_pixel_edge_revision']=dict(source_frames=len(frames),profiles=count,independent_source_exposures_added=0,source_band_unchanged=True,photometric_frame_budget_shared=True,shadow_material_identity_not_resolved=True)
    write_json(Path(output),doc);return doc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--document',type=Path,required=True);ap.add_argument('--source-sequence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    doc=attach(json.loads(a.document.read_text(encoding='utf-8')),a.source_sequence,a.output);print(json.dumps(doc['source_pixel_edge_revision']),flush=True)

if __name__=='__main__':main()
