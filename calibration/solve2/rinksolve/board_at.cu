// New continuous-board batch. Same 128-node integral as ContinuousBoard.
// Coordinates are Euclidean L=1; station wraps on one continuous closed course.
__device__ double speed_at(double theta,const double* b) {
 double c=fmax(cos(theta),0.0),s=fmax(sin(theta),0.0),p=b[3];
 double norm=pow(c,p)+pow(s,p),rho=pow(norm,-1.0/p);
 double d=(pow(c,p-1.0)*s-pow(s,p-1.0)*c)/norm;
 double dx=b[1]*rho*(-s+c*d),dy=b[2]*rho*(c+s*d);
 return hypot(dx,dy);
}
__device__ double integral_at(double theta,const double* b,const double* nodes,const double* weights) {
 double total=0.0;
 for(int k=0;k<128;k++)total+=speed_at((nodes[k]+1.0)*theta*0.5,b)*weights[k];
 return total*theta*0.5;
}
__device__ double inverse_at(double target,const double* b,const double* nodes,const double* weights,int* failed) {
 const double halfpi=1.57079632679489661923132169163975144;
 double length=b[5];
 if(target < -1e-12 || target > length+1e-12){*failed=2;return 0.0;}
 // This is solely the same finite-arclength endpoint roundoff handling as CPU.
 target=fmin(fmax(target,0.0),length);
 double angle=target/length*halfpi,lower=0.0,upper=halfpi;
 for(int it=0;it<28;it++) {
  double residual=integral_at(angle,b,nodes,weights)-target;
  if(fabs(residual)<=4e-15)break;
  if(residual<0.0)lower=angle;
  if(residual>0.0)upper=angle;
  double candidate=angle-residual/speed_at(angle,b);
  if(!(candidate>lower && candidate<upper))candidate=(lower+upper)*0.5;
  angle=candidate;
 }
 if(fabs(integral_at(angle,b,nodes,weights)-target)>1e-11)*failed=3;
 return angle;
}
extern "C" __global__ void board_at(const double* q,const double* boards,const double* nodes,const double* weights,
 int count,int variants,double* output,unsigned char* status) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count*variants)return;
 const double* b=boards+(i/count)*23;
 double* out=output+i*6;for(int k=0;k<6;k++)out[k]=0.0;status[i]=0;
 double station=q[i*2],height=q[i*2+1],width=b[0],rx=b[1],ry=b[2],p=b[3];
 if(!(isfinite(station)&&isfinite(height)&&width>0&&rx>0&&rx<0.5&&ry>0&&ry<width*0.5&&p>1.0&&b[4]>0)) {status[i]=1;return;}
 const double* breaks=b+14;double perimeter=breaks[8];
 double s=fmod(station,perimeter);if(s<0)s+=perimeter;if(s>=perimeter)s=0;
 int part=0;while(part<7 && s>breaks[part+1])part++;
 double along=s-breaks[part],x=0.0,y=0.0,tx=0.0,ty=0.0;
 if(part==0){x=-0.5+rx+along;y=-width*0.5;tx=1.0;}
 else if(part==2){x=0.5;y=-width*0.5+ry+along;ty=1.0;}
 else if(part==4){x=0.5-rx-along;y=width*0.5;tx=-1.0;}
 else if(part==6){x=-0.5;y=width*0.5-ry-along;ty=-1.0;}
 else {
  bool reverse=(part==1||part==5);double target=reverse?b[5]-along:along;
  int fail=0;double theta=inverse_at(target,b,nodes,weights,&fail);if(fail){status[i]=fail;return;}
  double c=fmax(cos(theta),0.0),sn=fmax(sin(theta),0.0),norm=pow(c,p)+pow(sn,p),rho=pow(norm,-1.0/p);
  double sx=(part==1||part==3)?1.0:-1.0,sy=(part==3||part==5)?1.0:-1.0;
  x=sx*(0.5-rx+rx*c*rho);y=sy*(width*0.5-ry+ry*sn*rho);
  double ld=(pow(c,p-1.0)*sn-pow(sn,p-1.0)*c)/norm;
  double dx=rx*rho*(-sn+c*ld),dy=ry*rho*(c+sn*ld),v=hypot(dx,dy),sign=reverse?-1.0:1.0;
  tx=dx*sx*sign/v;ty=dy*sy*sign/v;
 }
 out[0]=x;out[1]=y;out[2]=height;out[3]=tx;out[4]=ty;out[5]=0.0;
 if(!(isfinite(x)&&isfinite(y)&&isfinite(tx)&&isfinite(ty)))status[i]=4;
}
