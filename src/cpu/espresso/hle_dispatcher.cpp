#include "cpu/espresso/hle_dispatcher.hpp"

#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/guest_memory.hpp"

#include <algorithm>
#include <stdexcept>

namespace affogato::cpu::espresso
{

void HleDispatcher::register_function(std::string library, std::string symbol, Handler handler)
{
    if (library.empty() || symbol.empty() || !handler)
    {
        throw std::invalid_argument("HLE registration requires a library, symbol, and handler");
    }

    const ImportName name{std::move(library), std::move(symbol)};
    if (addresses_.contains(name) || data_exports_.contains(name) || data_addresses_.contains(name))
    {
        throw std::logic_error("HLE function was registered after its import was bound");
    }
    if (!functions_.emplace(name, std::move(handler)).second)
    {
        throw std::logic_error("HLE function is already registered");
    }
}

void HleDispatcher::register_data(
    std::string library,
    std::string symbol,
    std::uint32_t size,
    std::uint32_t alignment,
    DataInitializer initializer)
{
    if (library.empty() || symbol.empty() || size == 0 || alignment == 0 ||
        (alignment & (alignment - 1U)) != 0)
    {
        throw std::invalid_argument(
            "HLE data registration requires a library, symbol, nonzero size, and power-of-two alignment");
    }

    const ImportName name{std::move(library), std::move(symbol)};
    if (functions_.contains(name) || addresses_.contains(name) || data_addresses_.contains(name))
    {
        throw std::logic_error("HLE data export conflicts with an existing HLE export");
    }
    if (!data_exports_.emplace(name, DataExport{size, alignment, std::move(initializer)}).second)
    {
        throw std::logic_error("HLE data export is already registered");
    }
}

std::uint32_t HleDispatcher::bind_import(std::string library, std::string symbol)
{
    ImportName name{std::move(library), std::move(symbol)};
    if (const auto existing = addresses_.find(name); existing != addresses_.end())
    {
        return existing->second;
    }
    if (next_import_address_ > import_address_limit - 4U)
    {
        throw std::length_error("HLE import trampoline address range is exhausted");
    }

    const std::uint32_t address = next_import_address_;
    next_import_address_ += 4U;
    const auto registered = functions_.find(name);
    Handler handler = registered == functions_.end() ? Handler{} : registered->second;
    addresses_.emplace(name, address);
    bound_imports_.emplace(address, BoundImport{std::move(name), std::move(handler)});
    return address;
}

std::optional<std::uint32_t> HleDispatcher::bind_data_import(
    std::string library,
    std::string symbol,
    GuestMemory& memory)
{
    ImportName name{std::move(library), std::move(symbol)};
    if (const auto existing = data_addresses_.find(name); existing != data_addresses_.end())
    {
        return existing->second;
    }
    const auto registered = data_exports_.find(name);
    if (registered == data_exports_.end())
    {
        return std::nullopt;
    }

    const DataExport& data = registered->second;
    const std::uint64_t address =
        (static_cast<std::uint64_t>(next_data_address_) + data.alignment - 1U) &
        ~static_cast<std::uint64_t>(data.alignment - 1U);
    const std::uint64_t end = address + data.size;
    if (address < first_data_address || end > data_address_limit)
    {
        throw std::length_error("HLE data export address range is exhausted");
    }

    if (end > memory.size())
    {
        const std::uint64_t sparse_begin = std::max<std::uint64_t>(address, memory.size());
        memory.map_region(
            static_cast<std::uint32_t>(sparse_begin),
            static_cast<std::size_t>(end - sparse_begin));
    }
    const std::uint32_t guest_address = static_cast<std::uint32_t>(address);
    memory.zero_fill(guest_address, data.size);
    if (data.initializer)
    {
        data.initializer(memory, guest_address);
    }
    data_addresses_.emplace(name, guest_address);
    next_data_address_ = static_cast<std::uint32_t>(end);
    return guest_address;
}

std::optional<std::uint32_t> HleDispatcher::data_address(
    const std::string& library,
    const std::string& symbol) const
{
    const auto found = data_addresses_.find(ImportName{library, symbol});
    return found == data_addresses_.end()
        ? std::nullopt
        : std::optional<std::uint32_t>{found->second};
}

HleDispatchResult HleDispatcher::dispatch(EspressoCore& core, std::uint32_t address)
{
    const auto imported = bound_imports_.find(address);
    if (imported == bound_imports_.end())
    {
        return HleDispatchResult::not_hle;
    }

    const BoundImport& binding = imported->second;
    ++call_counts_[binding.name];
    HleCallTraceEntry* trace = nullptr;
    const std::string full_name = recent_call_trace_enabled_
        ? binding.name.first + "::" + binding.name.second
        : std::string{};
    if (recent_call_trace_enabled_ &&
        (recent_call_trace_filter_.empty() || full_name == recent_call_trace_filter_))
    {
        trace = &recent_calls_[next_recent_call_];
        *trace = {};
        trace->import_address = address;
        trace->return_address = core.state.lr;
        trace->name = full_name;
        for (std::size_t index = 0U; index < trace->arguments.size(); ++index)
        {
            trace->arguments[index] = core.state.gpr[index + 3U];
        }
        trace->implemented = static_cast<bool>(binding.handler);
    }
    if (!binding.handler)
    {
        if (trace != nullptr)
        {
            trace->results = trace->arguments;
            next_recent_call_ = (next_recent_call_ + 1U) % trace_capacity;
            recent_call_count_ = std::min(recent_call_count_ + 1U, trace_capacity);
        }
        last_unimplemented_call_ = binding.name.first + "::" + binding.name.second;
        return HleDispatchResult::unimplemented;
    }

    const std::uint32_t return_address = core.state.lr;
    const std::uint32_t previous_thread = core.current_thread_address;
    binding.handler(core);
    // A cooperative Cafe scheduler may have installed another guest thread's
    // saved context (including its CIA) while servicing this HLE call.
    if (core.current_thread_address == previous_thread)
    {
        core.state.cia = return_address;
    }
    if (trace != nullptr)
    {
        for (std::size_t index = 0U; index < trace->results.size(); ++index)
        {
            trace->results[index] = core.state.gpr[index + 3U];
        }
        trace->completed = true;
        next_recent_call_ = (next_recent_call_ + 1U) % trace_capacity;
        recent_call_count_ = std::min(recent_call_count_ + 1U, trace_capacity);
    }
    last_unimplemented_call_.clear();
    return HleDispatchResult::executed;
}

std::vector<HleCallStatistic> HleDispatcher::call_statistics_snapshot() const
{
    std::vector<HleCallStatistic> result;
    result.reserve(call_counts_.size());
    for (const auto& [name, calls] : call_counts_)
    {
        result.push_back({name.first + "::" + name.second, calls});
    }
    return result;
}

std::vector<HleCallTraceEntry> HleDispatcher::recent_call_trace_snapshot() const
{
    std::vector<HleCallTraceEntry> result;
    result.reserve(recent_call_count_);
    const std::size_t first = (next_recent_call_ + trace_capacity - recent_call_count_) %
        trace_capacity;
    for (std::size_t index = 0U; index < recent_call_count_; ++index)
    {
        result.push_back(recent_calls_[(first + index) % trace_capacity]);
    }
    return result;
}

void HleDispatcher::set_recent_call_trace_enabled(bool enabled)
{
    recent_call_trace_enabled_ = enabled;
    if (enabled)
    {
        recent_calls_.clear();
        recent_calls_.resize(trace_capacity);
        recent_call_count_ = 0U;
        next_recent_call_ = 0U;
    }
}

void HleDispatcher::set_recent_call_trace_filter(std::string name)
{
    recent_call_trace_filter_ = std::move(name);
    recent_call_count_ = 0U;
    next_recent_call_ = 0U;
}

}
