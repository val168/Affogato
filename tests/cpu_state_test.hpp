#pragma once

#include "cpu/espresso/cpu_state.hpp"

#include <cassert>
#include <cstdint>

namespace affogato::tests
{

inline void cpu_state_tests()
{
    using affogato::cpu::espresso::CpuState;

    CpuState cpu{};

    for (std::uint32_t value : cpu.gpr)
    {
        assert(value == 0);
    }

    assert(cpu.cr == 0);
    assert(cpu.xer == 0);
    assert(cpu.lr == 0);
    assert(cpu.ctr == 0);
    assert(cpu.cia == 0);

    cpu.gpr[3] = 42;
    cpu.cr = 1;
    cpu.lr = 0x1000;

    assert(cpu.gpr[3] == 42);
    assert(cpu.cr == 1);
    assert(cpu.lr == 0x1000);

    cpu.reset();

    assert(cpu.gpr[3] == 0);
    assert(cpu.cr == 0);
    assert(cpu.lr == 0);

}

}
