#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace affogato::cpu::espresso
{

class EspressoCore;

enum class HleDispatchResult
{
    not_hle,
    executed,
    unimplemented,
};

class HleDispatcher
{
public:
    using Handler = std::function<void(EspressoCore&)>;

    static constexpr std::uint32_t first_import_address = 0x03F00000U;
    static constexpr std::uint32_t import_address_limit = 0x04000000U;

    void register_function(std::string library, std::string symbol, Handler handler);
    [[nodiscard]] std::uint32_t bind_import(std::string library, std::string symbol);
    [[nodiscard]] HleDispatchResult dispatch(EspressoCore& core, std::uint32_t address);

    [[nodiscard]] const std::string& last_unimplemented_call() const noexcept
    {
        return last_unimplemented_call_;
    }

private:
    using ImportName = std::pair<std::string, std::string>;

    struct BoundImport
    {
        ImportName name;
        Handler handler;
    };

    std::map<ImportName, Handler> functions_;
    std::map<ImportName, std::uint32_t> addresses_;
    std::map<std::uint32_t, BoundImport> bound_imports_;
    std::uint32_t next_import_address_{first_import_address};
    std::string last_unimplemented_call_;
};

}
