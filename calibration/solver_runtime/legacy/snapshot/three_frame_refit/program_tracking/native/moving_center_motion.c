/* Strict-double row-fused moving-centre Brown/profile/Schur kernel.
 * Compile without fast-math. Every source row and both residual components
 * survive; all limits, curvature terms and source plane derivatives follow
 * z190_moving_center_motion.py. OpenMP only partitions independent rows. */
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
static double max2(double a,double b){return a>b?a:b;}
static double min2(double a,double b){return a<b?a:b;}
static double dot3(const double*a,const double*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void mv(const double*A,const double*x,double*y){for(int j=0;j<3;j++)y[j]=dot3(A+3*j,x);}
static int inverse(const double*uv,double f,const double*pp,double k,double*n,double*nf){
 double d[2]={(uv[0]-pp[0])/f,(uv[1]-pp[1])/f};n[0]=d[0];n[1]=d[1];
 for(int it=0;it<12;it++){
  double s=1+k*(n[0]*n[0]+n[1]*n[1]),a=s+2*k*n[0]*n[0],b=2*k*n[0]*n[1],c=s+2*k*n[1]*n[1],det=a*c-b*b;
  if(det<=1e-12)return 1;
  double rx=n[0]*s-d[0],ry=n[1]*s-d[1],sx=(c*rx-b*ry)/det,sy=(a*ry-b*rx)/det;
  n[0]-=sx;n[1]-=sy;if(max2(fabs(sx),fabs(sy))<2e-13)break;
 }
 double s=1+k*(n[0]*n[0]+n[1]*n[1]);if(max2(fabs(n[0]*s-d[0]),fabs(n[1]*s-d[1]))>1e-9)return 2;
 double a=s+2*k*n[0]*n[0],b=2*k*n[0]*n[1],c=s+2*k*n[1]*n[1],det=a*c-b*b;
 nf[0]=-(c*d[0]-b*d[1])/det;nf[1]=-(a*d[1]-b*d[0])/det;return 0;
}
/* Same global max-step stop as the NumPy source Brown inversion, with all
 * arithmetic parenthesized to preserve its ufunc operation ordering. */
API int inverse_brown_strict(int64_t N,const double*uv,const double*f,const double*pp,double k,
                            int threads,double*n,double*nf,int*errors){
 double*partial=(double*)calloc(threads,sizeof(double));if(!partial)return 1;
 int64_t i;int done=0;
 #pragma omp parallel num_threads(threads) shared(done,partial)
 {
  int tid=omp_get_thread_num();
  #pragma omp for schedule(static)
  for(i=0;i<N;i++){n[2*i]=(uv[2*i]-pp[0])/f[i];n[2*i+1]=(uv[2*i+1]-pp[1])/f[i];}
  for(int iteration=0;iteration<12;iteration++){
   double largest=0;
   #pragma omp for schedule(static)
   for(i=0;i<N;i++){
    double x=n[2*i],y=n[2*i+1],s=1+k*(x*x+y*y),a=s+2*k*(x*x),b=2*k*x*y,c=s+2*k*(y*y),det=a*c-b*b;
    if(det<=1e-12){errors[i]=1;continue;}
    double dx=(uv[2*i]-pp[0])/f[i],dy=(uv[2*i+1]-pp[1])/f[i],rx=x*s-dx,ry=y*s-dy;
    double sx=(c*rx-b*ry)/det,sy=(a*ry-b*rx)/det;n[2*i]-=sx;n[2*i+1]-=sy;largest=max2(largest,max2(fabs(sx),fabs(sy)));
   }
   partial[tid]=largest;
   #pragma omp barrier
   #pragma omp single
   {double largest_all=0;for(int worker=0;worker<threads;worker++)largest_all=max2(largest_all,partial[worker]);done=largest_all<2e-13;}
   if(done)break;
  }
  #pragma omp for schedule(static)
  for(i=0;i<N;i++){
   double x=n[2*i],y=n[2*i+1],s=1+k*(x*x+y*y),dx=(uv[2*i]-pp[0])/f[i],dy=(uv[2*i+1]-pp[1])/f[i];
   if(max2(fabs(x*s-dx),fabs(y*s-dy))>1e-9){errors[i]=2;continue;}
   double a=s+2*k*(x*x),b=2*k*x*y,c=s+2*k*(y*y),det=a*c-b*b;
   nf[2*i]=-(c*dx-b*dy)/det;nf[2*i+1]=-(a*dy-b*dx)/det;
  }
 }
 free(partial);return 0;
}
/* Line-chart z=1 projection, exact tangent/curvature in lambda. */
static void line(double nx,double ny,const double*u,double f,const double*pp,double k,double*pixel,double*D,double*jr,double*jrr){
 double s=1+k*(nx*nx+ny*ny);pixel[0]=f*nx*s+pp[0];pixel[1]=f*ny*s+pp[1];
 D[0]=s+2*k*(nx*nx);D[1]=D[2]=2*k*nx*ny;D[3]=s+2*k*(ny*ny);
 jr[0]=(f*D[0])*u[0]+(f*D[1])*u[1];jr[1]=(f*D[2])*u[0]+(f*D[3])*u[1];
 double nu=nx*u[0]+ny*u[1],uu=u[0]*u[0]+u[1]*u[1];
 jrr[0]=f*(2*k*(2*nu*u[0]+uu*nx));jrr[1]=f*(2*k*(2*nu*u[1]+uu*ny));
}
static int project(const double*Y,double f,const double*pp,double k,double*pixel,double*FY){
 if(Y[2]<=0||!isfinite(Y[0])||!isfinite(Y[1])||!isfinite(Y[2]))return 3;
 double n[2]={Y[0]/Y[2],Y[1]/Y[2]},u[2]={0,0},D[4],jr[2],jrr[2];line(n[0],n[1],u,f,pp,k,pixel,D,jr,jrr);
 for(int c=0;c<2;c++){FY[3*c]=f*D[2*c]/Y[2];FY[3*c+1]=f*D[2*c+1]/Y[2];FY[3*c+2]=-f*(D[2*c]*n[0]+D[2*c+1]*n[1])/Y[2];}return 0;
}
static int motion_impl(int64_t N,const double*a,const double*b,const double*fa,const double*fb,
 const double*Ra,const double*Rb,const double*dRa,const double*dRb,const double*Ca,const double*Cb,
 const double*dCa_s,const double*dCb_s,const double*dCa_v,const double*dCb_v,
 const double*inverse_a,const double*inverse_af,const double*inverse_b,
 const double*source_direction,const double*target_direction,const double*target_translation,const double*pp,double k,
 const double*planes,const double*sig,const double*weight,double depth_floor,int jac,int schur,int threads,
 double*res,double*Ja,double*Jb,double*Js,double*trace,int*errors,double*nuisance){
 int64_t i;
 #pragma omp parallel for schedule(static) num_threads(threads)
 for(i=0;i<N;i++){
  const double*aa=a+2*i,*bb=b+2*i,*ra=Ra+9*i,*rb=Rb+9*i,*ca=Ca+3*i,*cb=Cb+3*i,*pl=planes+4*i;
  const double*na=inverse_a+2*i,*nb=inverse_b+2*i,*naf=inverse_af+2*i;
  double W[2]={weight[2*i]/(sig[2*i]*sig[2*i]),weight[2*i+1]/(sig[2*i+1]*sig[2*i+1])};int error=0;
  double local[3]={na[0],na[1],1},delta[3];
  const double*da=source_direction+3*i,*q=target_direction+3*i,*t=target_translation+3*i;
  for(int j=0;j<3;j++)delta[j]=ca[j]-cb[j];
  int known=dot3(pl,pl)>0;double tangent[3]={t[0]-q[0],t[1]-q[1],t[2]-q[2]};
  double low=0,upper=1;if(tangent[2]>0)low=max2(0,(1e-12-q[2])/tangent[2]);if(tangent[2]<0)upper=min2(1,(q[2]-1e-12)/(-tangent[2]));
  if(upper<=low||(q[2]<=0&&t[2]<=0)){errors[i]=4;continue;}
  double v[2]={q[0]-nb[0]*q[2],q[1]-nb[1]*q[2]},w[2]={t[0]-nb[0]*t[2],t[1]-nb[1]*t[2]};
  double rho=max2(-(v[0]*w[0]+v[1]*w[1])/max2(w[0]*w[0]+w[1]*w[1],1e-30),0),tau=min2(max2(rho/(1+rho),low),upper);
  double pd=-pl[3]-dot3(pl,ca);
  if(known){rho=dot3(pl,da)/pd;if(rho<=0||q[2]+rho*t[2]<=0||!isfinite(rho)){errors[i]=5;continue;}tau=rho/(1+rho);}
  if(q[2]<=0){errors[i]=6;continue;}
  double nq[2]={q[0]/q[2],q[1]/q[2]},h[2]={t[0]-nq[0]*t[2],t[1]-nq[1]*t[2]};
  double L=sqrt(h[0]*h[0]+h[1]*h[1]),u[2]={h[0]/max2(L,1e-30),h[1]/max2(L,1e-30)};
  int identifiable=L>1e-14,positive_tz=t[2]>0,floor_bound=0;
  double lam=max2(0,(nb[0]-nq[0])*u[0]+(nb[1]-nq[1])*u[1]),lam_upper=positive_tz?L/t[2]:INFINITY,rho_max=0;
  if(depth_floor>0){rho_max=1/depth_floor;double den=q[2]+rho_max*t[2],floor_lam=den>0?rho_max*L/den:INFINITY;floor_bound=!known&&floor_lam<=lam_upper;lam_upper=min2(lam_upper,floor_lam);}
  lam=min2(lam,lam_upper);if(!identifiable)lam=0;
  double pixel[2],D[4],jr[2],jrr[2],r[2],g=0,GN=0,H=0;int row_it=0,search_trials=0,search_failed=0,iterations=0;
  for(iterations=0;iterations<16;iterations++){
   line(nq[0]+lam*u[0],nq[1]+lam*u[1],u,fb[i],pp,k,pixel,D,jr,jrr);
   r[0]=pixel[0]-bb[0];r[1]=pixel[1]-bb[1];g=W[0]*r[0]*jr[0]+W[1]*r[1]*jr[1];GN=W[0]*(jr[0]*jr[0])+W[1]*(jr[1]*jr[1]);H=GN+(W[0]*r[0]*jrr[0]+W[1]*r[1]*jrr[1]);
   int at_low=lam<=1e-14,at_upper=lam>=lam_upper-1e-14,stationary=fabs(g)<=1e-8*max2(sqrt(GN),1e-12);
   if(known||!identifiable||GN<=1e-22||(at_low&&g>=0)||(at_upper&&g<=0)||stationary)break;
   row_it++;double step=-g/max2(H>0?H:GN,1e-30),candidate=min2(max2(lam+step,0),lam_upper),basecost=W[0]*(r[0]*r[0])+W[1]*(r[1]*r[1]);int changed=0;
   for(int tr=0;tr<12;tr++){
    search_trials++;double cp[2],dd[4],jj[2],rr[2];line(nq[0]+candidate*u[0],nq[1]+candidate*u[1],u,fb[i],pp,k,cp,dd,jj,rr);
    double x=cp[0]-bb[0],y=cp[1]-bb[1],cost=W[0]*(x*x)+W[1]*(y*y);
    if(cost<=basecost+DBL_EPSILON*8*max2(1,basecost)){lam=candidate;changed=1;break;}candidate=(candidate+lam)*.5;
   }
   if(!changed){search_failed=1;break;}
  }
  if(!known){double den=L+lam*(q[2]-t[2]);tau=den>0?lam*q[2]/den:0;}tau=min2(max2(tau,0),1);
  double n[2]={nq[0]+lam*u[0],nq[1]+lam*u[1]};line(n[0],n[1],u,fb[i],pp,k,pixel,D,jr,jrr);
  int floor_endpoint=floor_bound&&lam>=lam_upper-1e-14;
  double fy[6],Y[3];
  if(floor_endpoint){for(int j=0;j<3;j++)Y[j]=q[j]+rho_max*t[j];error=project(Y,fb[i],pp,k,pixel,fy);}
  if(known){for(int j=0;j<3;j++)Y[j]=q[j]+tau*tangent[j];error=project(Y,fb[i],pp,k,pixel,fy);}
  if(error){errors[i]=error;continue;}
  r[0]=pixel[0]-bb[0];r[1]=pixel[1]-bb[1];res[2*i]=r[0];res[2*i+1]=r[1];
  g=W[0]*r[0]*jr[0]+W[1]*r[1]*jr[1];GN=W[0]*(jr[0]*jr[0])+W[1]*(jr[1]*jr[1]);H=GN+(W[0]*r[0]*jrr[0]+W[1]*r[1]*jrr[1]);
  int at_low=lam<=1e-14,at_upper=lam>=lam_upper-1e-14,boundary=!known&&((at_low&&g>=0)||(at_upper&&g<=0));
  int stationary=fabs(g)<=1e-8*max2(sqrt(GN),1e-12),unidentified=!known&&(!identifiable||GN<=1e-22);
  int converged=known||boundary||unidentified||(!known&&stationary&&H>1e-22),interior=!known&&!boundary&&!unidentified&&stationary&&H>1e-22;
  int implicit_valid=!known&&!boundary&&!unidentified&&H>1e-22,schur_active=schur&&interior&&GN>1e-22;
  /* numeric trace is retained for equivalence audits without full tensors. */
  double*tr=trace+16*i;tr[0]=lam;tr[1]=tau;tr[2]=g;tr[3]=GN;tr[4]=H;tr[5]=row_it;tr[6]=search_trials;tr[7]=search_failed;tr[8]=converged;tr[9]=interior;tr[10]=boundary;tr[11]=unidentified;tr[12]=known;tr[13]=0;tr[14]=0;tr[15]=iterations+1;
  /* Optional local eliminated-depth moments, before the original Schur
   * projection. This workspace changes neither pixels nor returned J. */
  if(nuisance){double*nn=nuisance+14*i;for(int p=0;p<14;p++)nn[p]=0.;nn[0]=jr[0];nn[1]=jr[1];}
  if(!jac)continue;
  double dDa[12]={0},dqa[12]={0},dqb[12]={0},dta[12]={0},dtb[12]={0},dts[12]={0};
  for(int j=0;j<3;j++){
   for(int p=0;p<3;p++)for(int x=0;x<3;x++)dDa[4*j+p]+=local[x]*dRa[27*i+9*p+3*x+j];
   dDa[4*j+3]=naf[0]*ra[j]+naf[1]*ra[3+j];
  }
  for(int j=0;j<3;j++)for(int p=0;p<4;p++)for(int x=0;x<3;x++){
   dqa[4*j+p]+=rb[3*j+x]*dDa[4*x+p];
   dta[4*j+p]+=rb[3*j+x]*dCa_v[12*i+4*x+p];dtb[4*j+p]-=rb[3*j+x]*dCb_v[12*i+4*x+p];
   dts[4*j+p]+=rb[3*j+x]*(dCa_s[12*i+4*x+p]-dCb_s[12*i+4*x+p]);
   if(p<3){dqb[4*j+p]+=dRb[27*i+9*p+3*j+x]*da[x];dtb[4*j+p]+=dRb[27*i+9*p+3*j+x]*delta[x];}
  }
  double zero[12]={0};const double*dqs[3]={dqa,dqb,zero},*dts3[3]={dta,dtb,dts};double*outs[3]={Ja+8*i,Jb+8*i,Js+8*i};
  for(int which=0;which<3;which++){
   const double*dq=dqs[which],*dt=dts3[which];double*dst=outs[which];
   for(int p=0;p<4;p++){
    double dnq[2]={(dq[p]-nq[0]*dq[8+p])/q[2],(dq[4+p]-nq[1]*dq[8+p])/q[2]};
    double dh[2]={dt[p]-dnq[0]*t[2]-nq[0]*dt[8+p],dt[4+p]-dnq[1]*t[2]-nq[1]*dt[8+p]};
    double dl=u[0]*dh[0]+u[1]*dh[1],du[2]={(dh[0]-u[0]*dl)/max2(L,1e-30),(dh[1]-u[1]*dl)/max2(L,1e-30)};
    double dn[2]={dnq[0]+lam*du[0],dnq[1]+lam*du[1]},raw[2],mixed[2],block[2],drho=0;
    double nd=n[0]*dn[0]+n[1]*dn[1],nu=n[0]*u[0]+n[1]*u[1],ud=u[0]*dn[0]+u[1]*dn[1];
    for(int c=0;c<2;c++){raw[c]=fb[i]*(D[2*c]*dn[0]+D[2*c+1]*dn[1]);mixed[c]=fb[i]*(D[2*c]*du[0]+D[2*c+1]*du[1]+2*k*(nd*u[c]+nu*dn[c]+ud*n[c]));}
    if(which==1&&p==3){raw[0]+=pixel[0]-pp[0];raw[1]+=pixel[1]-pp[1];mixed[0]+=jr[0];mixed[1]+=jr[1];}
    if(implicit_valid)drho=-((W[0]*jr[0]*raw[0]+W[1]*jr[1]*raw[1])+(W[0]*r[0]*mixed[0]+W[1]*r[1]*mixed[1]))/H;
    if(known&&which!=1){const double*centre=which==0?dCa_v+12*i:dCa_s+12*i;const double*direction=which==0?dDa:zero;double dd=0,dc=0;for(int x=0;x<3;x++){dd+=pl[x]*direction[4*x+p];dc+=pl[x]*centre[4*x+p];}drho=((dd+rho*dc)/pd)*(1-tau)*(1-tau);}
    block[0]=raw[0]+jr[0]*drho;block[1]=raw[1]+jr[1]*drho;
    if(boundary&&at_upper&&positive_tz&&!floor_bound){double dbound=(dl*t[2]-L*dt[8+p])/max2(t[2]*t[2],1e-30);block[0]=raw[0]+jr[0]*dbound;block[1]=raw[1]+jr[1]*dbound;}
    if(boundary&&at_upper&&floor_bound){for(int x=0;x<3;x++)Y[x]=q[x]+rho_max*t[x];project(Y,fb[i],pp,k,pixel,fy);for(int c=0;c<2;c++){block[c]=0;for(int x=0;x<3;x++)block[c]+=fy[3*c+x]*(dq[4*x+p]+rho_max*dt[4*x+p]);if(which==1&&p==3)block[c]+=pixel[c]-pp[c];}}
    if(unidentified){for(int c=0;c<2;c++){block[c]=fb[i]*(D[2*c]*dnq[0]+D[2*c+1]*dnq[1]);if(which==1&&p==3)block[c]+=pixel[c]-pp[c];}}
    if(known){for(int x=0;x<3;x++)Y[x]=q[x]+tau*tangent[x];project(Y,fb[i],pp,k,pixel,fy);for(int c=0;c<2;c++){double fraw=0,ftan=0;for(int x=0;x<3;x++){fraw+=fy[3*c+x]*((1-tau)*dq[4*x+p]+tau*dt[4*x+p]);ftan+=fy[3*c+x]*tangent[x];}if(which==1&&p==3)fraw+=pixel[c]-pp[c];raw[c]=fraw;block[c]=fraw+ftan*drho;}}
    double exact0=block[0],exact1=block[1];if(schur_active){double factor=(W[0]*jr[0]*raw[0]+W[1]*jr[1]*raw[1])/max2(GN,1e-30);block[0]=raw[0]-jr[0]*factor;block[1]=raw[1]-jr[1]*factor;if(nuisance)nuisance[14*i+2+4*which+p]=factor;}
    dst[p]=block[0];dst[4+p]=block[1];tr[13]=max2(tr[13],fabs(W[0]*r[0]*(block[0]-exact0)+W[1]*r[1]*(block[1]-exact1)));
    if(schur_active)tr[14]=max2(tr[14],fabs(W[0]*jr[0]*block[0]+W[1]*jr[1]*block[1]));
   }
  }
 }
 return 0;
}

/* Keep the historical ABI and arithmetic for every ordinary caller. */
#define MOTION_ARGUMENTS int64_t N,const double*a,const double*b,const double*fa,const double*fb, \
 const double*Ra,const double*Rb,const double*dRa,const double*dRb,const double*Ca,const double*Cb, \
 const double*dCa_s,const double*dCb_s,const double*dCa_v,const double*dCb_v, \
 const double*inverse_a,const double*inverse_af,const double*inverse_b, \
 const double*source_direction,const double*target_direction,const double*target_translation,const double*pp,double k, \
 const double*planes,const double*sig,const double*weight,double depth_floor,int jac,int schur,int threads, \
 double*res,double*Ja,double*Jb,double*Js,double*trace,int*errors
#define MOTION_VALUES N,a,b,fa,fb,Ra,Rb,dRa,dRb,Ca,Cb,dCa_s,dCb_s,dCa_v,dCb_v, \
 inverse_a,inverse_af,inverse_b,source_direction,target_direction,target_translation,pp,k, \
 planes,sig,weight,depth_floor,jac,schur,threads,res,Ja,Jb,Js,trace,errors
API int motion_strict(MOTION_ARGUMENTS){return motion_impl(MOTION_VALUES,NULL);}
API int motion_strict_nuisance(MOTION_ARGUMENTS,double*nuisance){return motion_impl(MOTION_VALUES,nuisance);}
