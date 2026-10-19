from pathlib import Path
p=Path('calibration/program_tracking/native_ice_material_repair.py');s=p.read_text()
needle=' def read(self,fid,index,native):'
s=s.replace(needle,''' def conic(self,curve):
  center,axes,angle=cv2.fitEllipse(np.asarray(curve,np.float32));angle=np.deg2rad(angle)
  rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  return np.asarray(center),rotation@np.diag(1/(np.asarray(axes)*.5)**2)@rotation.T
 def transformed_conic_distance(self,conic,points,z):
  center,matrix=conic;tx,ty,xx,xy,yx,yy=z
  inverse=np.linalg.inv(np.array([[1+xx,xy],[yx,1+yy]]))
  delta=(np.asarray(points)-[960+tx,540+ty])@inverse.T+[960,540]-center
  gradient=2*(delta@matrix)@inverse
  return (np.einsum('ij,ij->i',delta@matrix,delta)-1)/np.maximum(1e-9,np.linalg.norm(gradient,axis=1))
''' +needle)
s=s.replace("base_curves={role:self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0] for role in group['radii']}","base_curves={role:self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0] for role in group['radii']}\n  conics={role:self.conic(curve) for role,curve in base_curves.items()}")
s=s.replace("values.extend(self.curve_distance(self.source_transform(base_curves[role],z),observed))","values.extend(self.transformed_conic_distance(conics[role],observed,z))")
s=s.replace("errors=abs(self.curve_distance(curve,np.array([r['uv'] for r in selected])))","errors=abs(self.transformed_conic_distance(conics[role],np.array([r['uv'] for r in selected]),fitted))")
p.write_text(s)
