#include "cpu/espresso/guest_mutex.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <limits>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t name_offset = 0x04U;
constexpr std::uint32_t owner_offset = 0x1CU;
constexpr std::uint32_t count_offset = 0x20U;

void validate_mutex(EspressoCore& core, std::uint32_t address)
{
    if (address == 0 || core.memory.read32_be(address) != os_mutex_tag)
    {
        throw HleExecutionError("OSMutex pointer is null or has an invalid tag");
    }
}

}

void initialize_guest_os_mutex(
    EspressoCore& core,
    std::uint32_t address,
    std::uint32_t name)
{
    if (address == 0)
    {
        throw HleExecutionError("OSInitMutex received a null OSMutex pointer");
    }

    core.memory.zero_fill(address, os_mutex_size);
    core.memory.write32_be(address, os_mutex_tag);
    core.memory.write32_be(address + name_offset, name);
}

std::uint32_t allocate_guest_os_mutex(EspressoCore& core, std::uint32_t name)
{
    const std::uint32_t address = core.allocate_guest_memory(os_mutex_size, 8U);
    if (address == 0)
    {
        return 0;
    }
    initialize_guest_os_mutex(core, address, name);
    return address;
}

bool lock_guest_os_mutex(EspressoCore& core, std::uint32_t address)
{
    validate_mutex(core, address);
    if (core.current_thread_address == 0)
    {
        throw HleExecutionError("OSLockMutex has no current guest OSThread");
    }

    const std::uint32_t owner = core.memory.read32_be(address + owner_offset);
    const std::uint32_t count = core.memory.read32_be(address + count_offset);
    if (owner == 0)
    {
        if (count != 0)
        {
            throw HleExecutionError("OSMutex has a recursion count without an owner");
        }
        core.memory.write32_be(address + owner_offset, core.current_thread_address);
        core.memory.write32_be(address + count_offset, 1U);
        return true;
    }
    if (owner != core.current_thread_address)
    {
        return false;
    }
    if (count == 0 || count == std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError("OSMutex recursion count is invalid or exhausted");
    }

    core.memory.write32_be(address + count_offset, count + 1U);
    return true;
}

void unlock_guest_os_mutex(EspressoCore& core, std::uint32_t address)
{
    validate_mutex(core, address);
    if (core.current_thread_address == 0)
    {
        throw HleExecutionError("OSUnlockMutex has no current guest OSThread");
    }

    const std::uint32_t owner = core.memory.read32_be(address + owner_offset);
    const std::uint32_t count = core.memory.read32_be(address + count_offset);
    if (owner != core.current_thread_address || count == 0)
    {
        throw HleExecutionError("OSUnlockMutex called by a thread that does not own the mutex");
    }

    if (count == 1U)
    {
        core.memory.write32_be(address + owner_offset, 0);
        core.memory.write32_be(address + count_offset, 0);
        return;
    }
    core.memory.write32_be(address + count_offset, count - 1U);
}

}
