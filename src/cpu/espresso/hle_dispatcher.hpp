#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace affogato::cpu::espresso
{

class EspressoCore;
class GuestMemory;

class HleExecutionError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

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
    using DataInitializer = std::function<void(GuestMemory&, std::uint32_t)>;

    static constexpr std::uint32_t first_data_address = 0x03E00000U;
    static constexpr std::uint32_t data_address_limit = 0x03F00000U;
    static constexpr std::uint32_t first_import_address = 0x03F00000U;
    static constexpr std::uint32_t import_address_limit = 0x04000000U;

    void register_function(std::string library, std::string symbol, Handler handler);
    void register_data(
        std::string library,
        std::string symbol,
        std::uint32_t size,
        std::uint32_t alignment,
        DataInitializer initializer = {});
    [[nodiscard]] std::uint32_t bind_import(std::string library, std::string symbol);
    [[nodiscard]] std::optional<std::uint32_t> bind_data_import(
        std::string library,
        std::string symbol,
        GuestMemory& memory);
    [[nodiscard]] std::optional<std::uint32_t> data_address(
        const std::string& library,
        const std::string& symbol) const;
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

    struct DataExport
    {
        std::uint32_t size{};
        std::uint32_t alignment{};
        DataInitializer initializer;
    };

    std::map<ImportName, Handler> functions_;
    std::map<ImportName, DataExport> data_exports_;
    std::map<ImportName, std::uint32_t> addresses_;
    std::map<ImportName, std::uint32_t> data_addresses_;
    std::map<std::uint32_t, BoundImport> bound_imports_;
    std::uint32_t next_import_address_{first_import_address};
    std::uint32_t next_data_address_{first_data_address};
    std::string last_unimplemented_call_;
};

}
