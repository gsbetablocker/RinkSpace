// Real triangle-section consumer.  C ABI only: no Torch C++ dependency.
// All arrays are device pointers owned by the caller; launches use its stream.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
__device__ double cross2(double ax,double ay,double bx,double by) {return ax*by-ay*bx;}
__device__ double norm2(double x,double y) {return sqrt(x*x+y*y);}
__device__ bool intersects(const double* a,const double* b,const double* c,const double* d,double tol) {
    double abx=b[0]-a[0],aby=b[1]-a[1],cdx=d[0]-c[0],cdy=d[1]-c[1];
    double at=tol*fmax(norm2(abx,aby),norm2(cdx,cdy));
    double o1=cross2(abx,aby,c[0]-a[0],c[1]-a[1]);
    double o2=cross2(abx,aby,d[0]-a[0],d[1]-a[1]);
    double o3=cross2(cdx,cdy,a[0]-c[0],a[1]-c[1]);
    double o4=cross2(cdx,cdy,b[0]-c[0],b[1]-c[1]);
    bool hit=((o1<=at&&o2>=-at)||(o2<=at&&o1>=-at))&&((o3<=at&&o4>=-at)||(o4<=at&&o3>=-at));
    for(int k=0;k<2;k++) hit=hit&&(fmax(fmin(a[k],b[k]),fmin(c[k],d[k]))<=fmin(fmax(a[k],b[k]),fmax(c[k],d[k]))+tol);
    return hit;
}
__device__ bool simple_loop(const double* xy,const int64_t* order,int count,double tol) {
    if(count<3)return false;
    double area=0.;bool good=true;
    for(int i=0;i<count;i++) {
        int ia=order?int(order[i]):i,ib=order?int(order[(i+1)%count]):(i+1)%count;
        const double *a=xy+2*ia,*b=xy+2*ib;
        area+=cross2(a[0],a[1],b[0],b[1]);
        if(norm2(b[0]-a[0],b[1]-a[1])<=tol)good=false;
        for(int j=i+1;j<count;j++) {
            if(j==i+1||(i==0&&j==count-1))continue;
            int ic=order?int(order[j]):j,id=order?int(order[(j+1)%count]):(j+1)%count;
            if(intersects(a,b,xy+2*ic,xy+2*id,tol))good=false;
        }
    }
    return good&&fabs(area/2)>tol*tol;
}
__device__ double signed_gap(double px,double py,const double* polygon,int np) {
    double distance=INFINITY;int inside=0;
    for(int i=0;i<np;i++) {
        const double *a=polygon+2*i,*b=polygon+2*((i+1)%np);
        double vx=b[0]-a[0],vy=b[1]-a[1],dx=px-a[0],dy=py-a[1];
        double denom=vx*vx+vy*vy;
        double fraction=denom>0?(dx*vx+dy*vy)/denom:0.;
        fraction=fmin(fmax(fraction,0.),1.);
        distance=fmin(distance,norm2(dx-fraction*vx,dy-fraction*vy));
        double safe=vy==0.?1.:vy;
        if((a[1]>py)!=(b[1]>py))inside+=px<vx*(py-a[1])/safe+a[0];
    }
    return inside%2?distance:-distance;
}

// Reason codes match REASONS in body_foot_cuda_bridge.py.  Workspace width is
// the complete edge set, not an assumed maximum section length.
__global__ void collar(const double* vertices,int nv,const int64_t* edges,int ne,
    const int64_t* triangles,int nf,const int64_t* edge_faces,const int64_t* edge_counts,
    const double* polygon,int np,const unsigned char* z_changed,double height,double tol,double domain,
    double* xy,unsigned char* crossed,unsigned char* visited,int64_t* order,
    unsigned char* triangle_number,int64_t* triangle_sum,
    double* residual,double* breach,double* minimum,int64_t* sample_count,int32_t* reason) {
    int row=blockIdx.x,lane=threadIdx.x;
    const double* v=vertices+(int64_t)row*nv*3;
    double* points=xy+(int64_t)row*ne*2;
    unsigned char* cut=crossed+(int64_t)row*ne;
    unsigned char* seen=visited+(int64_t)row*ne;
    int64_t* loop=order+(int64_t)row*ne;
    unsigned char* number=triangle_number+(int64_t)row*nf;
    int64_t* sums=triangle_sum+(int64_t)row*nf;
    const double* poly=polygon+(int64_t)row*np*2;
    for(int e=lane;e<ne;e+=blockDim.x) {
        const double *a=v+3*edges[2*e],*b=v+3*edges[2*e+1];
        double za=a[2]-height,zb=b[2]-height;
        cut[e]=za*zb<0.;seen[e]=0;
        double fraction=fabs(za-zb)>tol?za/(za-zb):0.;
        points[2*e]=a[0]+fraction*(b[0]-a[0]);points[2*e+1]=a[1]+fraction*(b[1]-a[1]);
    }
    __syncthreads();
    for(int t=lane;t<nf;t+=blockDim.x) {
        int c=0;int64_t s=0;
        for(int j=0;j<3;j++){int e=int(triangles[3*t+j]);c+=cut[e];if(cut[e])s+=e;}
        number[t]=c;sums[t]=s;
    }
    __syncthreads();
    __shared__ int why;
    if(lane==0) {
        int first=-1,ncut=0,ntwo=0;bool on=false,bad_degree=false,bad_triangle=false,nonfinite=false;
        for(int e=0;e<ne;e++) {
            if(cut[e]){if(first<0)first=e;ncut++;if(edge_counts[e]!=2)bad_degree=true;}
            const double *a=v+3*edges[2*e],*b=v+3*edges[2*e+1];
            if(fabs(a[2]-height)<=tol||fabs(b[2]-height)<=tol)on=true;
            for(int k=0;k<3;k++)if(!isfinite(a[k])||!isfinite(b[k]))nonfinite=true;
        }
        for(int t=0;t<nf;t++){if(number[t]==2)ntwo++;else if(number[t]!=0)bad_triangle=true;}
        sample_count[row]=ncut+ntwo;
        why=first<0?1:on?2:bad_degree?3:bad_triangle?4:0;
        if(nonfinite)why=10;
        if(!why) {
            int current=first,previous=-1,length=0;
            for(;;) {
                loop[length++]=current;seen[current]=1;
                int f0=int(edge_faces[2*current]),f1=int(edge_faces[2*current+1]);
                int a=int(sums[f0])-current,b=int(sums[f1])-current;
                int following=a!=previous?a:b;
                if(following<0||following>=ne){why=5;break;}
                bool returned=following==first;
                if(seen[following]&&!returned){why=5;break;}
                if(returned){if(length!=ncut)why=6;break;}
                if(length>=ne){why=5;break;}
                previous=current;current=following;
            }
            if(!why&&!simple_loop(points,loop,length,tol))why=7;
        }
        if(!why&&!simple_loop(poly,nullptr,np,tol))why=8;
        if(!why&&z_changed[row])why=9;
    }
    __syncthreads();
    double value=INFINITY;
    if(!why) {
        for(int e=lane;e<ne;e+=blockDim.x)if(cut[e])value=fmin(value,signed_gap(points[2*e],points[2*e+1],poly,np));
        for(int t=lane;t<nf;t+=blockDim.x)if(number[t]==2) {
            double px=0.,py=0.;
            for(int j=0;j<3;j++){int e=int(triangles[3*t+j]);if(cut[e]){px+=points[2*e];py+=points[2*e+1];}}
            value=fmin(value,signed_gap(px/2.,py/2.,poly,np));
        }
    }
    __shared__ double reduction[128];reduction[lane]=value;__syncthreads();
    for(int stride=64;stride;stride/=2){if(lane<stride)reduction[lane]=fmin(reduction[lane],reduction[lane+stride]);__syncthreads();}
    if(lane==0){reason[row]=why;minimum[row]=why?NAN:reduction[0];breach[row]=why?0.:fmax(-reduction[0],0.);residual[row]=why?domain:breach[row];}
}
}

#ifdef _WIN32
#define BODY_FOOT_EXPORT __declspec(dllexport)
#else
#define BODY_FOOT_EXPORT __attribute__((visibility("default")))
#endif
extern "C" BODY_FOOT_EXPORT int body_foot_collar(
    const double* vertices,int n,int nv,const int64_t* edges,int ne,
    const int64_t* triangles,int nf,const int64_t* edge_faces,const int64_t* edge_counts,
    const double* polygon,int np,const unsigned char* z_changed,double height,double tol,double domain,
    double* xy,unsigned char* crossed,unsigned char* visited,int64_t* order,
    unsigned char* triangle_number,int64_t* triangle_sum,
    double* residual,double* breach,double* minimum,int64_t* sample_count,int32_t* reason,void* stream) {
    if(n)collar<<<n,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(vertices,nv,edges,ne,triangles,nf,edge_faces,edge_counts,
        polygon,np,z_changed,height,tol,domain,xy,crossed,visited,order,triangle_number,triangle_sum,residual,breach,minimum,sample_count,reason);
    return int(cudaGetLastError());
}
