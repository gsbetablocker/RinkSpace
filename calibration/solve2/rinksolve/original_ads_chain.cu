// Current ad-only data chain. The shared board_at and project_at kernels remain unchanged.
// FP64 throughout; no fast math, row substitution, source masking or finite-station clamp.
extern "C" __global__ void ad_unwrapped(const double* x,const double* points,const long long* curve_offsets,
 const int* curve_sizes,const int* curve_ids,const long long* ad_offsets,const double* anchors,
 const long long* station_offsets,const long long* row_offsets,const double* seeds,int n,double* q,double* dq,unsigned char* status) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ci=curve_ids[i],sz=curve_sizes[ci];
 double s=station_offsets[i]>=0?x[station_offsets[i]]:seeds[i];status[i]=0;
 if(!(isfinite(s)&&s>=0&&s<=1)){status[i]=1;return;}
 double position=s*(sz-1);int ix=sz==1?0:min((int)floor(position),sz-2);double w=sz==1?0:position-ix;
 const double* a=points+2*(curve_offsets[ci]+ix);const double* b=points+2*(curve_offsets[ci]+(sz==1?ix:ix+1));
 const double* v=x+ad_offsets[i];double c=cos(v[3]),ss=sin(v[3]),scale=exp(v[2]);
 double p0=a[0]*(1-w)+b[0]*w-anchors[2*ci],p1=a[1]*(1-w)+b[1]*w-anchors[2*ci+1];
 q[2*i]=anchors[2*ci]+scale*(p0*c-p1*ss)+v[0];q[2*i+1]=anchors[2*ci+1]+scale*(p0*ss+p1*c)+v[1];
 double d0=(b[0]-a[0])*(sz-1),d1=(b[1]-a[1])*(sz-1);
 dq[2*i]=scale*(d0*c-d1*ss);dq[2*i+1]=scale*(d0*ss+d1*c);
}
extern "C" __global__ void ad_board_camera_rows(const double* board_out,const double* x,const long long* row_offsets,
 const double* frame_times,double tau_over_h,double tau,int n,double* xyz,double* times) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int k=0;k<3;k++)xyz[3*i+k]=board_out[6*i+k];times[i]=frame_times[i]+tau_over_h*x[row_offsets[i]]-.5*tau;
}
extern "C" __global__ void ad_zero(double* out,long long n){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=0;}
extern "C" __global__ void ad_projection_chain(const double* full,const double* b,const double* q,const double* dq,const double* x,
 const int* curve_ids,const double* anchors,const long long* ad_offsets,const long long* ad_jbase,const long long* local_jbase,
 const int* local_width,const int* station_column,const int* row_column,double tau_over_h,int n,double* compact,double* jac) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const double* f=full+69*i;double* out=compact+12*i;
 for(int k=0;k<10;k++)out[k]=f[k];
 out[10]=f[10]*f[20]+f[11]*f[21]+f[12]*f[22];out[11]=f[13]*f[20]+f[14]*f[21]+f[15]*f[22];
 int ci=curve_ids[i];const double* v=x+ad_offsets[i];double off0=q[2*i]-anchors[2*ci]-v[0],off1=q[2*i+1]-anchors[2*ci+1]-v[1];
 double qj0[4]={1.,0.,off0,-off1},qj1[4]={0.,1.,off1,off0};
 for(int component=0;component<2;component++){
  const double* j=f+4+component*3;double tangent=j[0]*b[6*i+3]+j[1]*b[6*i+4]+j[2]*b[6*i+5];
  for(int k=0;k<4;k++)jac[ad_jbase[i]+4*component+k]=tangent*qj0[k]+j[2]*qj1[k];
  long long base=local_jbase[i]+component*(long long)local_width[i];
  if(station_column[i]>=0)jac[base+station_column[i]]=tangent*dq[2*i]+j[2]*dq[2*i+1];
  jac[base+row_column[i]]=out[10+component]*tau_over_h;
 }
}
extern "C" __global__ void ad_compact_uv(const double* uv,int n,double* compact){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){compact[12*i]=uv[2*i];compact[12*i+1]=uv[2*i+1];}}

extern "C" __global__ void ad_constraints(const double* forward,int stride,const double* target,const double* x,
 const long long* row_offsets,const int* budget_index,const double* budget_values,const long long* budget_offsets,
 const long long* equality_offsets,const long long* ad_jbase,const long long* local_jbase,const int* local_width,const int* row_column,
 const double* raw_jac,int n,int nb,long long budget_vector_offset,long long equality_jac_offset,int full,double* constraints,double* jac) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const double* uv=forward+stride*i;
 constraints[nb+i]=x[row_offsets[i]]-uv[1];int bi=budget_index[i];double cu=0.,cv=0.;
 if(bi>=0){double u=uv[0]-target[2*i],v=uv[1]-target[2*i+1],E=budget_values[bi];
  constraints[bi]=(u*u+v*v)/(E*E)-x[budget_vector_offset];cu=2*u/(E*E);cv=2*v/(E*E);}
 if(!full)return;int w=local_width[i];long long eq=equality_jac_offset+equality_offsets[i];long long bo=bi>=0?budget_offsets[bi]:0;
 for(int k=0;k<4;k++){double u=raw_jac[ad_jbase[i]+k],v=raw_jac[ad_jbase[i]+4+k];jac[eq+k]=-v;if(bi>=0)jac[bo+k]=cu*u+cv*v;}
 for(int k=0;k<w;k++){double u=raw_jac[local_jbase[i]+k],v=raw_jac[local_jbase[i]+w+k];jac[eq+4+k]=-v+(k==row_column[i]?1.:0.);if(bi>=0)jac[bo+4+k]=cu*u+cv*v;}
 if(bi>=0)jac[bo+4+w]=-1.;
}
