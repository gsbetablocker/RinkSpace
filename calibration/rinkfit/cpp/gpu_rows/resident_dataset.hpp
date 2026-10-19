#pragma once
#include "rinkfit/event_packet.hpp"
namespace resident { const rinkfit::EventPacket& dataset(const std::filesystem::path&,const std::filesystem::path&); }
int resident_qbar_main(int,char**);
int resident_values_main(int,char**);

int resident_whole_uv_main(int,char**);
int resident_profile_implicit_main(int,char**);
