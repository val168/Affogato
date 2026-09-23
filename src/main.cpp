#include "cpu/espresso/interpreter.hpp"

#include <iostream>

int main()
{
    using affogato::cpu::espresso::EspressoCore;
    using affogato::cpu::espresso::StepResult;

    EspressoCore core(0x100);
    core.memory.write32_be(0, 0x3860002AU); // addi r3, r0, 42

    if (core.step() != StepResult::executed)
    {
        return 1;
    }

    std::cout << "Affogato\n";
    std::cout << "GPR3: " << core.state.gpr[3] << '\n';

    return 0;
}
