#pragma once
#include <cstddef>
struct GPUMeta {double R0[9],n[3],b[3],t1[3],t2[3],fref,Rmax,dt,first,refw[4];int nc,cb,dep,ref[4];};
struct GPUPath {int kind,geom,ad,ri,bs,begin,count;double a0,a1,h;};
struct GPURow {int path,kind;double t,s,v,alpha,uv[2],normal[2],E,query[12];};
struct GPUOutput {double world[3],uv[2],depth,r2,fold,raw[2],q,rs,query[2];int arithmetic,status;};
struct GPURoot {double uv[2],row,depth,equality,denominator;int roots,valid,allvalid,monotonic;};
struct GPUTiming {float h2d_ms,kernel_ms,d2h_ms,root_ms;double allocation_wall_ms,total_wall_ms;};
extern "C" int run_gpu(const GPUMeta*,const int*,const double*,int,int,const GPUPath*,int,const double*,int,const GPURow*,int,GPUOutput*,GPURoot*,GPUTiming*,char*,int,int);
