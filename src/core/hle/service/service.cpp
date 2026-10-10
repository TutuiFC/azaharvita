// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <fmt/format.h>
#ifdef __PSVITA__
#include <array>
#include <atomic>
#include <cstring>
#include <vector>
#include "common/vita_diag.h"
#include "core/vita_loop_profile.h"
#endif
#include "common/assert.h"
#include "common/hacks/hack_manager.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/hle/ipc.h"
#include "core/hle/kernel/client_port.h"
#include "core/hle/kernel/handle_table.h"
#include "core/hle/kernel/process.h"
#include "core/hle/kernel/server_port.h"
#include "core/hle/kernel/server_session.h"
#include "core/hle/service/ac/ac.h"
#include "core/hle/service/act/act.h"
#include "core/hle/service/am/am.h"
#include "core/hle/service/apt/apt.h"
#include "core/hle/service/boss/boss.h"
#include "core/hle/service/cam/cam.h"
#include "core/hle/service/cam/y2r_u.h"
#include "core/hle/service/cecd/cecd.h"
#include "core/hle/service/cfg/cfg.h"
#include "core/hle/service/csnd/csnd_snd.h"
#include "core/hle/service/dlp/dlp.h"
#include "core/hle/service/dsp/dsp_dsp.h"
#include "core/hle/service/err/err_f.h"
#include "core/hle/service/frd/frd.h"
#include "core/hle/service/fs/archive.h"
#include "core/hle/service/fs/fs_user.h"
#include "core/hle/service/gsp/gsp.h"
#include "core/hle/service/gsp/gsp_lcd.h"
#include "core/hle/service/hid/hid.h"
#include "core/hle/service/http/http_c.h"
#include "core/hle/service/ir/ir.h"
#include "core/hle/service/ldr_ro/ldr_ro.h"
#include "core/hle/service/mcu/mcu.h"
#include "core/hle/service/mic/mic_u.h"
#include "core/hle/service/mvd/mvd.h"
#include "core/hle/service/ndm/ndm_u.h"
#include "core/hle/service/news/news.h"
#include "core/hle/service/nfc/nfc.h"
#include "core/hle/service/nim/nim.h"
#include "core/hle/service/nwm/nwm.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/hle/service/pm/pm.h"
#include "core/hle/service/ps/ps_ps.h"
#include "core/hle/service/ptm/ptm.h"
#include "core/hle/service/pxi/pxi.h"
#include "core/hle/service/qtm/qtm.h"
#include "core/hle/service/service.h"
#include "core/hle/service/sm/sm.h"
#include "core/hle/service/sm/srv.h"
#include "core/hle/service/soc/soc_u.h"
#include "core/hle/service/ssl/ssl_c.h"
#include "core/loader/loader.h"

namespace Service {

const std::array<ServiceModuleInfo, 41> service_module_map{
    {{"FS", 0x00040130'00001102, FS::InstallInterfaces, false},
     {"PM", 0x00040130'00001202, PM::InstallInterfaces, false},
     {"LDR", 0x00040130'00003702, LDR::InstallInterfaces, false},
     {"PXI", 0x00040130'00001402, PXI::InstallInterfaces, false},

     {"ERR", 0x00040030'00008A02, ERR::InstallInterfaces, false},
     {"AC", 0x00040130'00002402, AC::InstallInterfaces, false},
     {"ACT", 0x00040130'00003802, ACT::InstallInterfaces, true},
     {"AM", 0x00040130'00001502, AM::InstallInterfaces, false},
     {"BOSS", 0x00040130'00003402, BOSS::InstallInterfaces, true},
     {"CAM", 0x00040130'00001602,
      [](Core::System& system) {
          CAM::InstallInterfaces(system);
          Y2R::InstallInterfaces(system);
      },
      false},
     {"CECD", 0x00040130'00002602, CECD::InstallInterfaces, true},
     {"CFG", 0x00040130'00001702, CFG::InstallInterfaces, false},
     {"DLP", 0x00040130'00002802, DLP::InstallInterfaces, false},
     {"DSP", 0x00040130'00001A02, DSP::InstallInterfaces, false},
     {"FRD", 0x00040130'00003202, FRD::InstallInterfaces, true},
     {"GSP", 0x00040130'00001C02, GSP::InstallInterfaces, false},
     {"HID", 0x00040130'00001D02, HID::InstallInterfaces, false},
     {"IR", 0x00040130'00003302, IR::InstallInterfaces, false},
     {"MIC", 0x00040130'00002002, MIC::InstallInterfaces, false},
     {"MVD", 0x00040130'20004102, MVD::InstallInterfaces, false},
     {"NDM", 0x00040130'00002B02, NDM::InstallInterfaces, false},
     {"NEWS", 0x00040130'00003502, NEWS::InstallInterfaces, false},
     {"NFC", 0x00040130'00004002, NFC::InstallInterfaces, false},
     {"NIM", 0x00040130'00002C02, NIM::InstallInterfaces, true},
     {"NS", 0x00040130'00008002, APT::InstallInterfaces, false},
     {"NWM", 0x00040130'00002D02, NWM::InstallInterfaces, false},
     {"PTM", 0x00040130'00002202, PTM::InstallInterfaces, false},
     {"QTM", 0x00040130'00004202, QTM::InstallInterfaces, false},
     {"CSND", 0x00040130'00002702, CSND::InstallInterfaces, false},
     {"HTTP", 0x00040130'00002902, HTTP::InstallInterfaces, false},
     {"SOC", 0x00040130'00002E02, SOC::InstallInterfaces, false},
     {"SSL", 0x00040130'00002F02, SSL::InstallInterfaces, false},
     {"PS", 0x00040130'00003102, PS::InstallInterfaces, false},
     {"PLGLDR", 0x00040130'00006902, PLGLDR::InstallInterfaces, false},
     {"MCU", 0x00040130'00001F02, MCU::InstallInterfaces, false},
     // no HLE implementation
     {"CDC", 0x00040130'00001802, nullptr, false},
     {"GPIO", 0x00040130'00001B02, nullptr, false},
     {"I2C", 0x00040130'00001E02, nullptr, false},
     {"MP", 0x00040130'00002A02, nullptr, false},
     {"PDN", 0x00040130'00002102, nullptr, false},
     {"SPI", 0x00040130'00002302, nullptr, false}}};

/**
 * Creates a function string for logging, complete with the name (or header code, depending
 * on what's passed in) the port name, and all the cmd_buff arguments.
 */
[[maybe_unused]] static std::string MakeFunctionString(std::string_view name,
                                                       std::string_view port_name,
                                                       const u32* cmd_buff) {
    // Number of params == bits 0-5 + bits 6-11
    int num_params = (cmd_buff[0] & 0x3F) + ((cmd_buff[0] >> 6) & 0x3F);

    std::string function_string = fmt::format("function '{}': port={}", name, port_name);
    for (int i = 1; i <= num_params; ++i) {
        function_string += fmt::format(", cmd_buff[{}]={:#X}", i, cmd_buff[i]);
    }
    return function_string;
}

ServiceFrameworkBase::ServiceFrameworkBase(const char* service_name, u32 max_sessions,
                                           InvokerFn* handler_invoker)
    : service_name(service_name), max_sessions(max_sessions), handler_invoker(handler_invoker) {}

ServiceFrameworkBase::~ServiceFrameworkBase() = default;

void ServiceFrameworkBase::InstallAsService(SM::ServiceManager& service_manager) {
    std::shared_ptr<Kernel::ServerPort> port;
    R_ASSERT(service_manager.RegisterService(std::addressof(port), service_name, max_sessions));
    port->SetHleHandler(shared_from_this());
}

void ServiceFrameworkBase::InstallAsNamedPort(Kernel::KernelSystem& kernel) {
    auto [server_port, client_port] = kernel.CreatePortPair(max_sessions, service_name);
    server_port->SetHleHandler(shared_from_this());
    kernel.AddNamedPort(service_name, std::move(client_port));
}

void ServiceFrameworkBase::RegisterHandlersBase(const FunctionInfoBase* functions, std::size_t n) {
    handlers.reserve(handlers.size() + n);
    for (std::size_t i = 0; i < n; ++i) {
        // Usually this array is sorted by id already, so hint to insert at the end
        handlers.emplace_hint(handlers.cend(), functions[i].command_id, functions[i]);
    }
}

void ServiceFrameworkBase::ReportUnimplementedFunction(u32* cmd_buf, const FunctionInfoBase* info) {
    IPC::Header header{cmd_buf[0]};
    int num_params = header.normal_params_size + header.translate_params_size;
    std::string function_name = info == nullptr ? fmt::format("{:#08x}", cmd_buf[0]) : info->name;

    std::string result =
        fmt::format("function '{}': port='{}' cmd_buf={{[0]={:#x} (0x{:04X}, {}, {})",
                    function_name, service_name, header.raw, header.command_id.Value(),
                    header.normal_params_size.Value(), header.translate_params_size.Value());
    for (int i = 1; i <= num_params; ++i) {
        result += fmt::format(", [{}]={:#x}", i, cmd_buf[i]);
    }

    result.push_back('}');

    LOG_ERROR(Service, "unknown / unimplemented {}", result);
    // TODO(bunnei): Hack - ignore error
    header.normal_params_size.Assign(1);
    header.translate_params_size.Assign(0);
    cmd_buf[0] = header.raw;
    cmd_buf[1] = 0;
}

#ifdef __PSVITA__
namespace {
/**
 * QUE SERVICIOS SE LLEVAN EL TIEMPO (0.3.2.3), por comando: ver TakeSvcProfile.
 * La clave es el FunctionInfoBase del comando; el nombre del servicio se copia
 * al crear la entrada (los servicios se destruyen al cerrar la partida).
 */
struct IpcCost {
    std::atomic<const void*> key{nullptr};
    const char* function = nullptr;
    char service[12] = {};
    std::atomic<u32> calls{0};
    std::atomic<u32> timed_us{0};
};
std::array<IpcCost, 128> g_ipc_costs{};
u32 g_ipc_tick = 0;
unsigned long long g_ipc_since = 0;

IpcCost* IpcCostOf(const void* key, const char* function, const std::string& service) {
    const std::size_t start = (reinterpret_cast<std::uintptr_t>(key) >> 3) % g_ipc_costs.size();
    for (std::size_t i = 0; i < g_ipc_costs.size(); i++) {
        IpcCost& cost = g_ipc_costs[(start + i) % g_ipc_costs.size()];
        const void* seen = cost.key.load(std::memory_order_acquire);
        if (seen == key) {
            return &cost;
        }
        if (seen == nullptr) {
            cost.function = function;
            std::strncpy(cost.service, service.c_str(), sizeof(cost.service) - 1);
            cost.key.store(key, std::memory_order_release);
            return &cost;
        }
    }
    return nullptr;
}
} // Anonymous namespace
#endif

void ServiceFrameworkBase::HandleSyncRequest(Kernel::HLERequestContext& context) {
    auto itr = handlers.find(context.CommandHeader().command_id.Value());
    const FunctionInfoBase* info = itr == handlers.end() ? nullptr : &itr->second;
    if (info == nullptr || info->handler_callback == nullptr) {
        context.ReportUnimplemented();
        return ReportUnimplementedFunction(context.CommandBuffer(), info);
    }

    LOG_TRACE(Service, "{}",
              MakeFunctionString(info->name, GetServiceName(), context.CommandBuffer()));
#ifdef __PSVITA__
    IpcCost* const cost = IpcCostOf(info, info->name, service_name);
    const bool timed = (++g_ipc_tick & 7u) == 0;
    const unsigned long long begin = timed ? Common::VitaMicros() : 0;
#endif
    handler_invoker(this, info->handler_callback, context);
#ifdef __PSVITA__
    if (cost != nullptr) {
        cost->calls.fetch_add(1, std::memory_order_relaxed);
        if (timed) {
            cost->timed_us.fetch_add(static_cast<u32>(Common::VitaMicros() - begin),
                                     std::memory_order_relaxed);
        }
    }
#endif
}

std::string ServiceFrameworkBase::GetFunctionName(IPC::Header header) const {
    auto itr = handlers.find(header.command_id.Value());
    if (itr == handlers.end()) {
        return "";
    }

    return itr->second.name;
}

static bool AttemptLLE(const ServiceModuleInfo& service_module, u64 loading_titleid) {
    const bool enable_recommended_lle_modules = Common::Hacks::hack_manager.OverrideBooleanSetting(
        Common::Hacks::HackType::ONLINE_LLE_REQUIRED, loading_titleid,
        Settings::values.enable_required_online_lle_modules.GetValue());

    if (!Settings::values.lle_modules.at(service_module.name) &&
        (!enable_recommended_lle_modules || !service_module.is_online_recommended))
        return false;
    std::unique_ptr<Loader::AppLoader> loader =
        Loader::GetLoader(AM::GetTitleContentPath(FS::MediaType::NAND, service_module.title_id));
    if (!loader) {
        LOG_ERROR(Service,
                  "Service module \"{}\" could not be loaded; Defaulting to HLE implementation.",
                  service_module.name);
        return false;
    }
    std::shared_ptr<Kernel::Process> process;
    Loader::ResultStatus load_result = loader->Load(process);
    if (load_result != Loader::ResultStatus::Success) {
        LOG_ERROR(Service,
                  "Service module \"{}\" could not be loaded (ResultStatus={}); Defaulting to HLE "
                  "implementation.",
                  service_module.name, static_cast<int>(load_result));
        return false;
    }
    LOG_DEBUG(Service, "Service module \"{}\" has been successfully loaded.", service_module.name);
    return true;
}

/// Initialize ServiceManager
void Init(Core::System& core, u64 loading_titleid, std::vector<u64>& lle_modules, bool allow_lle) {
    SM::ServiceManager::InstallInterfaces(core);
    core.Kernel().SetAppMainThreadExtendedSleep(false);
    bool lle_module_present = false;

    for (const auto& service_module : service_module_map) {
        if (core.GetSaveStateStatus() == Core::System::SaveStateStatus::LOADING &&
            std::find(lle_modules.begin(), lle_modules.end(), service_module.title_id) !=
                lle_modules.end()) {
            // The system module has already been loaded before, do not attempt to load again as the
            // process, threads, etc are already serialized in the kernel structures.
            lle_module_present |= true;
            continue;
        }

        const bool has_lle = allow_lle &&
                             core.GetSaveStateStatus() != Core::System::SaveStateStatus::LOADING &&
                             AttemptLLE(service_module, loading_titleid);
        if (has_lle) {
            lle_modules.push_back(service_module.title_id);
        }
        if (!has_lle && service_module.init_function != nullptr) {
            service_module.init_function(core);
        }
        lle_module_present |= has_lle;
    }
    if (lle_module_present) {
        // If there is at least one LLE module, tell the kernel to
        // add a extended sleep to the app main thread (if option enabled).
        core.Kernel().SetAppMainThreadExtendedSleep(true);
    }
    LOG_DEBUG(Service, "initialized OK");
}

} // namespace Service

#ifdef __PSVITA__
std::string Core::TakeIpcProfile() {
    const unsigned long long now = Common::VitaMicros();
    const unsigned long long wall =
        Service::g_ipc_since != 0 ? now - Service::g_ipc_since : 0;
    Service::g_ipc_since = now;
    struct Row {
        const Service::IpcCost* cost;
        u32 calls;
        u32 timed_us;
    };
    std::vector<Row> rows;
    for (const auto& cost : Service::g_ipc_costs) {
        if (cost.key.load(std::memory_order_acquire) == nullptr) {
            continue;
        }
        auto& mutable_cost = const_cast<Service::IpcCost&>(cost);
        const u32 calls = mutable_cost.calls.exchange(0, std::memory_order_relaxed);
        const u32 timed = mutable_cost.timed_us.exchange(0, std::memory_order_relaxed);
        if (calls != 0) {
            rows.push_back({&cost, calls, timed});
        }
    }
    if (wall == 0 || rows.empty()) {
        return "-";
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.timed_us > b.timed_us; });
    std::string text = "ms por s:";
    for (std::size_t i = 0; i < rows.size() && i < 8; i++) {
        text += fmt::format(" {}::{} {:.1f} ({}/s)", rows[i].cost->service,
                            rows[i].cost->function != nullptr ? rows[i].cost->function : "?",
                            static_cast<double>(rows[i].timed_us) * 8.0 * 1000.0 /
                                static_cast<double>(wall),
                            static_cast<unsigned long long>(rows[i].calls) * 1000000ull / wall);
    }
    return text;
}
#endif
