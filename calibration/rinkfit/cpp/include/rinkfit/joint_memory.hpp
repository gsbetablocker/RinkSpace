#pragma once
#include "rinkfit/event_io.hpp"
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif
namespace rinkfit {
inline eventio::Json joint_memory_sample(const std::string&stage){
 eventio::Json j={{"stage",stage},{"shared_working_sets_summed",false}};
#ifdef _WIN32
 PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);if(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m))){j["pid"]=GetCurrentProcessId();j["private_commit_bytes"]=m.PrivateUsage;j["working_set_bytes"]=m.WorkingSetSize;j["peak_working_set_bytes"]=m.PeakWorkingSetSize;j["peak_commit_bytes"]=m.PeakPagefileUsage;j["page_fault_count"]=m.PageFaultCount;}
 PERFORMANCE_INFORMATION p{};p.cb=sizeof(p);if(GetPerformanceInfo(&p,sizeof(p))){j["system_physical_available_bytes"]=p.PhysicalAvailable*p.PageSize;j["system_commit_bytes"]=p.CommitTotal*p.PageSize;j["system_commit_limit_bytes"]=p.CommitLimit*p.PageSize;}
 IO_COUNTERS io{};if(GetProcessIoCounters(GetCurrentProcess(),&io)){j["process_file_read_bytes"]=io.ReadTransferCount;j["process_file_write_bytes"]=io.WriteTransferCount;}
#else
 j["process_memory_status"]="unavailable_on_this_platform";
#endif
 return j;
}
inline void joint_memory_marker(const std::filesystem::path&out,const std::string&stage,eventio::Json extra=eventio::Json::object()){
 static const auto start=std::chrono::steady_clock::now();auto j=joint_memory_sample(stage);j["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();j["details"]=std::move(extra);std::filesystem::create_directories(out);std::ofstream f(out/"memory_stages.jsonl",std::ios::app);f<<j.dump()<<std::endl;
}
}
