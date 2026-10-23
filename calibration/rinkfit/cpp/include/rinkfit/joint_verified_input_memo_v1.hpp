#pragma once
#include "rinkfit/event_io.hpp"
#include <atomic>
#include <thread>
#include <iostream>
namespace rinkfit {
#ifdef _WIN32
inline std::string joint_input_file_version_v1(const std::filesystem::path&path){
 const HANDLE h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);if(h==INVALID_HANDLE_VALUE)return {};
 BY_HANDLE_FILE_INFORMATION info{};FILE_BASIC_INFO basic{};const bool ok=GetFileInformationByHandle(h,&info)&&GetFileInformationByHandleEx(h,FileBasicInfo,&basic,sizeof(basic));CloseHandle(h);if(!ok)return {};
 const auto index=(std::uint64_t(info.nFileIndexHigh)<<32)|info.nFileIndexLow,size=(std::uint64_t(info.nFileSizeHigh)<<32)|info.nFileSizeLow;
 return std::to_string(info.dwVolumeSerialNumber)+":"+std::to_string(index)+":"+std::to_string(size)+":"+std::to_string(basic.CreationTime.QuadPart)+":"+std::to_string(basic.LastWriteTime.QuadPart)+":"+std::to_string(basic.ChangeTime.QuadPart);
}
inline eventio::Json reuse_joint_verified_input_memo_v1(const std::filesystem::path&file,int threads){
 const auto start=std::chrono::steady_clock::now();if(!std::filesystem::exists(file))return {{"reused_files",0}};
 const auto doc=eventio::read_json(file);if(doc.value("schema",std::string{})!="rinkfit.verified_input_digest_cache.v1")throw std::runtime_error("unsupported input digest cache");const auto&entries=doc.at("entries");std::atomic<std::size_t>next{0},reused{0};
 auto work=[&](){for(;;){const auto i=next.fetch_add(1);if(i>=entries.size())break;try{const auto&r=entries.at(i);const auto p=std::filesystem::u8path(r.at("file").get<std::string>());const auto before=joint_input_file_version_v1(p);if(before.empty()||before!=r.at("fingerprint").get<std::string>())continue;const auto size=std::filesystem::file_size(p);const auto modified=std::filesystem::last_write_time(p);if(joint_input_file_version_v1(p)!=before)continue;const auto digest=r.at("sha256").get<std::string>();if(digest.size()!=64||digest.find_first_not_of("0123456789abcdef")!=std::string::npos)continue;const auto key=std::filesystem::absolute(p).lexically_normal().generic_u8string();auto&m=eventio::verification_memo();{std::lock_guard<std::mutex>lock(m.mutex);m.entries.insert_or_assign(key,eventio::VerificationMemo::Entry{size,modified,digest});}++reused;}catch(const std::filesystem::filesystem_error&){/* Ordinary source verification will report missing or changed inputs. */}}};
 std::vector<std::thread>pool;for(int i=1;i<std::max(1,threads);++i)pool.emplace_back(work);work();for(auto&t:pool)t.join();return {{"reused_files",reused.load()},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"same_volume_file_id_size_creation_write_and_change_times",true},{"changed_files_still_hashed",true}};
}
#else
inline eventio::Json reuse_joint_verified_input_memo_v1(const std::filesystem::path&,int){return {{"reused_files",0},{"platform_support",false}};}
#endif
}
