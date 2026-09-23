#include "cpu/espresso/hle_dispatcher.hpp"

#include "cpu/espresso/interpreter.hpp"

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
    if (addresses_.contains(name))
    {
        throw std::logic_error("HLE function was registered after its import was bound");
    }
    if (!functions_.emplace(name, std::move(handler)).second)
    {
        throw std::logic_error("HLE function is already registered");
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

HleDispatchResult HleDispatcher::dispatch(EspressoCore& core, std::uint32_t address)
{
    const auto imported = bound_imports_.find(address);
    if (imported == bound_imports_.end())
    {
        return HleDispatchResult::not_hle;
    }

    const BoundImport& binding = imported->second;
    if (!binding.handler)
    {
        last_unimplemented_call_ = binding.name.first + "::" + binding.name.second;
        return HleDispatchResult::unimplemented;
    }

    const std::uint32_t return_address = core.state.lr;
    binding.handler(core);
    core.state.cia = return_address;
    last_unimplemented_call_.clear();
    return HleDispatchResult::executed;
}

}
