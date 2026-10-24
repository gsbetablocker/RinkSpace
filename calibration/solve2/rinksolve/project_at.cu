// RinkSolve 2: same physical state and equations as ProjectionOperator.
// All arithmetic is double. No fast-math, clipping or observed-row substitution.
// NVRTC supplies device math builtins without a host C runtime include.
__device__ void mv(const double* A,const double* x,double* y){for(int i=0;i<3;i++){y[i]=0;for(int j=0;j<3;j++)y[i]+=A[3*i+j]*x[j];}}
__device__ void mtv(const double* A,const double* x,double* y){for(int i=0;i<3;i++){y[i]=0;for(int j=0;j<3;j++)y[i]+=A[3*j+i]*x[j];}}
__device__ void mm(const double* A,const double* B,double* O){for(int i=0;i<3;i++)for(int j=0;j<3;j++){O[3*i+j]=0;for(int k=0;k<3;k++)O[3*i+j]+=A[3*i+k]*B[3*k+j];}}
__device__ void cross(const double* a,const double* b,double* o){o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
__device__ void skew(const double* a,double* K){K[0]=0;K[1]=-a[2];K[2]=a[1];K[3]=a[2];K[4]=0;K[5]=-a[0];K[6]=-a[1];K[7]=a[0];K[8]=0;}
__device__ void rot(const double* a,double angle,double* R){double K[9];skew(a,K);double c=cos(angle),s=sin(angle);for(int i=0;i<3;i++)for(int j=0;j<3;j++)R[3*i+j]=(i==j?c:0)+(1-c)*a[i]*a[j]+s*K[3*i+j];}
extern "C" __global__ void project_at(const double* xyz,const double* times,const double* controls,const double* cam,int n,int nc,int full,double* out,unsigned char* status){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n)return;
 int stride=full?69:2;double* o=out+index*stride;for(int j=0;j<stride;j++)o[j]=nan("");status[index]=1;
 double t=times[index],dt=cam[37],first=cam[38];
 if(!isfinite(t)||t<first+dt||t>first+(nc-2)*dt)return;
 const double* X=xyz+index*3;for(int j=0;j<3;j++)if(!isfinite(X[j]))return;
 double st=(t-first)/dt;int cell=(int)floor(st);double u=st-cell;
 // Closed cardinal-spline domain uses the corresponding endpoint span.
 if(t==first+dt){cell=1;u=0;}
 if(t==first+(nc-2)*dt){cell=nc-3;u=1;}
 double w[4]={(1-u)*(1-u)*(1-u)/6,(3*u*u*u-6*u*u+4)/6,(-3*u*u*u+3*u*u+3*u+1)/6,u*u*u/6};
 double dw[4]={-(1-u)*(1-u)/2,(9*u*u-12*u)/6,(-9*u*u+6*u+3)/6,u*u/2};
 double values[3]={0,0,0},rates[3]={0,0,0};for(int k=0;k<4;k++){int cidx=cell-1+k;if(cidx<0||cidx>=nc){if(w[k]!=0)return;continue;}for(int j=0;j<3;j++){values[j]+=w[k]*controls[cidx*3+j];rates[j]+=dw[k]*controls[cidx*3+j]/dt;}}
 double A[9],B[9],tmp[9],R[9];rot(cam+9,values[0],A);rot(cam+12,values[1],B);mm(cam,A,tmp);mm(B,tmp,R);
 double z=values[2]-log(cam[24]);double d[3]={cam[18],cam[19],cam[20]+cam[22]*z+cam[23]*z*z};
 double v[3],a0[3],a[3];mtv(R,cam+12,v);cross(cam+9,v,a0);double norm=sqrt(a0[0]*a0[0]+a0[1]*a0[1]+a0[2]*a0[2]);if(!(norm>0))return;for(int j=0;j<3;j++)a[j]=a0[j]/norm;
 double Xp[3],q[3],Ra[3];for(int j=0;j<3;j++)Xp[j]=X[j]-cam[15+j];mv(R,Xp,q);mv(R,a,Ra);for(int j=0;j<3;j++)q[j]-=d[j]+cam[21]*Ra[j];
 double x=q[0]/q[2],y=q[1]/q[2],r2=x*x+y*y;
 double k1=cam[31]+cam[32]*z,k2=cam[33]+cam[34]*z,p1=cam[35],p2=cam[36];double radial=1+k1*r2+k2*r2*r2,g=k1+2*k2*r2;
 double xd=x*radial+2*p1*x*y+p2*(r2+2*x*x),yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y;
 double fx=exp(values[2]+cam[26]/2),fy=exp(values[2]-cam[26]/2);
 o[0]=fx*xd+cam[27]+cam[29]*z;o[1]=fy*yd+cam[28]+cam[30]*z;if(full){o[2]=q[2];o[3]=sqrt(r2);}
 if(!(q[2]>0)||!isfinite(o[0])||!isfinite(o[1])||r2>cam[25]*cam[25]){status[index]=2;return;}
 status[index]=0;if(!full)return;
 double J00=radial+2*x*x*g+2*p1*y+6*p2*x,J01=2*x*y*g+2*p1*x+2*p2*y,J11=radial+2*y*y*g+6*p1*y+2*p2*x;
 double Kn[9],Kb[9],dRa[9],dRb[9];skew(cam+9,Kn);skew(cam+12,Kb);mm(R,Kn,dRa);mm(Kb,R,dRb);
 for(int j=0;j<6;j++){
   double dq[3]={0,0,0};
   if(j<3){for(int h=0;h<3;h++)dq[h]=R[3*h+j];}
   else if(j==5)dq[2]=-(cam[22]+2*cam[23]*z);
   else{
     double* dR=(j==3?dRa:dRb);double dv[3],dcross[3],da[3],dRX[3],dRaxis[3],Rda[3];
     mtv(dR,cam+12,dv);cross(cam+9,dv,dcross);double dot=0;for(int h=0;h<3;h++)dot+=a[h]*dcross[h];for(int h=0;h<3;h++)da[h]=(dcross[h]-a[h]*dot)/norm;
     mv(dR,Xp,dRX);mv(dR,a,dRaxis);mv(R,da,Rda);for(int h=0;h<3;h++)dq[h]=dRX[h]-cam[21]*(dRaxis[h]+Rda[h]);
   }
   double dx=(dq[0]-x*dq[2])/q[2],dy=(dq[1]-y*dq[2])/q[2];double du=fx*(J00*dx+J01*dy),dv=fy*(J01*dx+J11*dy);
   if(j==5){double dk=cam[32]*r2+cam[34]*r2*r2;du+=fx*(xd+x*dk)+cam[29];dv+=fy*(yd+y*dk)+cam[30];}
   int base=j<3?4:10;int col=j<3?j:j-3;o[base+col]=du;o[base+3+col]=dv;
 }
 for(int j=0;j<4;j++)o[16+j]=w[j];for(int j=0;j<3;j++)o[20+j]=rates[j];for(int j=0;j<4;j++)o[65+j]=cell-1+j;
 // Shared rig derivatives at fixed physical time. ABI prefix [0,23) unchanged.
 double rayJ[6]={fx*J00/q[2],fx*J01/q[2],-fx*(J00*x+J01*y)/q[2],fy*J01/q[2],fy*J11/q[2],-fy*(J01*x+J11*y)/q[2]};
 for(int j=0;j<6;j++){o[23+j]=-o[4+j];o[29+j]=-rayJ[j];}
 for(int j=0;j<24;j++)o[35+j]=0;
 o[35]=.5*fx*xd;o[35+12]=-.5*fy*yd;
 o[35+1]=1;o[35+12+2]=1;o[35+3]=z;o[35+12+4]=z;
 for(int h=0;h<2;h++){
   double f=h?fy:fx,coord=h?y:x;int base=35+h*12;
   o[base+5]=f*coord*r2;o[base+6]=f*coord*r2*z;o[base+7]=f*coord*r2*r2;o[base+8]=f*coord*r2*r2*z;
   o[base+9]=f*(h?r2+2*y*y:2*x*y);o[base+10]=f*(h?2*x*y:r2+2*x*x);
 }
 double qd[3]={q[0]+d[0],q[1]+d[1],q[2]+d[2]},crossQ[9],chain[9];skew(qd,crossQ);mm(crossQ,cam+42,chain);
 for(int h=0;h<2;h++)for(int j=0;j<3;j++){double value=0;for(int k=0;k<3;k++)value-=rayJ[h*3+k]*chain[k*3+j];o[59+h*3+j]=value;}
}
