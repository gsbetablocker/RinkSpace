#pragma once
#include <cstddef>
struct GPUMeta {double R0[9],n[3],b[3],t1[3],t2[3],fref,Rmax,dt,first,refw[4];int nc,cb,dep,ref[4];};
struct GPUPath {int kind,geom,ad,ri,bs,begin,count;double a0,a1,h;};
struct UVRow {double t,s,v,alpha,budget;};
struct UVLayout {int global,geom,ad,path,block_begin,block_count,control_begin,control_count;};
struct UVJob {int row,column;};
struct UVPartial {double u,v;int arithmetic,status;};
struct UVTiming {float h2d_ms=0,kernel_ms=0,d2h_ms=0;double setup_ms=0,total_ms=0;int batches=0;};
extern "C" int rf_uv_create(const GPUMeta*,const int*,const int*,int,int,const GPUPath*,int,const double*,int,const int*,int,const int*,const int*,int,const UVLayout*,int,const UVJob*,int,void**,UVTiming*,char*,int);
extern "C" int rf_uv_evaluate(void*,const double*,int,const UVRow*,int,UVPartial*,int,UVTiming*,char*,int);
extern "C" void rf_uv_destroy(void*);
