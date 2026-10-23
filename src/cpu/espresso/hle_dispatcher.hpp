#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

struct HleCallStatistic
{
    std::string name;
    std::uint64_t calls{};
};

struct HleCallTraceEntry
{
    std::uint32_t import_address{};
    std::uint32_t return_address{};
    std::string name;
    std::array<std::uint32_t, 6U> arguments{};
    std::array<std::uint32_t, 6U> results{};
    bool implemented{};
    bool completed{};
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

    void configure_guest_address_base(std::uint32_t data_base);
    [[nodiscard]] std::uint32_t data_range_begin() const noexcept
    {
        return data_range_begin_;
    }
    [[nodiscard]] std::uint32_t data_range_end() const noexcept
    {
        return data_range_end_;
    }
    [[nodiscard]] std::uint32_t import_range_begin() const noexcept
    {
        return import_range_begin_;
    }
    [[nodiscard]] std::uint32_t import_range_end() const noexcept
    {
        return import_range_end_;
    }
    [[nodiscard]] bool is_import_address(std::uint32_t address) const noexcept
    {
        return address >= import_range_begin_ && address < import_range_end_;
    }

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
    [[nodiscard]] std::vector<HleCallStatistic> call_statistics_snapshot() const;
    [[nodiscard]] std::vector<HleCallTraceEntry> recent_call_trace_snapshot() const;
    void set_recent_call_trace_enabled(bool enabled);
    void set_recent_call_trace_filter(std::string name);

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
    std::map<ImportName, std::uint64_t> call_counts_;
    // Startup paths can make more than 64 HLE calls before a later failure.
    // Keep a bounded but wider diagnostic window so earlier initialization
    // calls remain inspectable at the eventual stop.
    static constexpr std::size_t trace_capacity = 512U;
    std::vector<HleCallTraceEntry> recent_calls_;
    std::size_t recent_call_count_{};
    std::size_t next_recent_call_{};
    bool recent_call_trace_enabled_{};
    std::string recent_call_trace_filter_;
    std::uint32_t data_range_begin_{first_data_address};
    std::uint32_t data_range_end_{data_address_limit};
    std::uint32_t import_range_begin_{first_import_address};
    std::uint32_t import_range_end_{import_address_limit};
    std::uint32_t next_import_address_{first_import_address};
    std::uint32_t next_data_address_{first_data_address};
    std::string last_unimplemented_call_;
};

}
