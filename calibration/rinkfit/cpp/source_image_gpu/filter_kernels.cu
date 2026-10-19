__device__ int reflect(int i,int n){while(i<0||i>=n){if(i<0)i=-i-1;else i=2*n-i-1;}return i;}
extern "C" __global__ void gauss(const float* a,float* b,int h,int w,int count,int axis,double k0,double k1,double k2,double k3){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;int y=(p/w)%h,x=p%w;double t=double(a[p])*k0;
 for(int j=3;j>=1;--j){int lo=axis==0?p+(reflect(y-j,h)-y)*w:p+reflect(x-j,w)-x;int hi=axis==0?p+(reflect(y+j,h)-y)*w:p+reflect(x+j,w)-x;double k=j==1?k1:(j==2?k2:k3);t+=(double(a[lo])+double(a[hi]))*k;}
 b[p]=float(t);
}
extern "C" __global__ void convert(const float* a,double* b,int count){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)b[p]=double(a[p]);}
extern "C" __global__ void spline(double* c,int h,int w,int batch,int axis,double z,double gain){
 int line=blockIdx.x*blockDim.x+threadIdx.x,lines=batch*(axis==0?w:h);if(line>=lines)return;int image=line/(axis==0?w:h),q=line%(axis==0?w:h),n=axis==0?h:w,stride=axis==0?w:1;double* a=c+image*h*w+(axis==0?q:q*w);
 for(int i=0;i<n;++i)a[i*stride]*=gain;
 double zi=z,zn1=pow(z,n-1);a[0]=a[0]+zn1*a[(n-1)*stride];
 for(int i=1;i<n-1;++i){a[0]+=zi*(a[i*stride]+zn1*a[(n-1-i)*stride]);zi*=z;}
 a[0]/=1-zn1*zn1;
 for(int i=1;i<n;++i)a[i*stride]+=z*a[(i-1)*stride];
 a[(n-1)*stride]=(z*a[(n-2)*stride]+a[(n-1)*stride])*z/(z*z-1);
 for(int i=n-2;i>=0;--i)a[i*stride]=z*(a[(i+1)*stride]-a[i*stride]);
}
