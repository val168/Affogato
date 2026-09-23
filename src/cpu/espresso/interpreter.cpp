#include "cpu/espresso/interpreter.hpp"

#include "cpu/espresso/decoder.hpp"

namespace affogato::cpu::espresso
{

StepResult EspressoCore::step()
{
    const std::uint32_t cia = state.cia;
    const std::uint32_t instruction_word = memory.read32_be(cia);
    const DecodedInstruction instruction = decode(instruction_word);

    if (instruction.opcode == Opcode::unsupported)
    {
        return StepResult::unsupported_instruction;
    }

    const std::uint32_t fallthrough = cia + 4U;
    std::uint32_t next_cia = fallthrough;

    switch (instruction.opcode)
    {
    case Opcode::addi:
    {
        const std::uint32_t base =
            instruction.base == 0 ? 0U : state.gpr[instruction.base];
        state.gpr[instruction.destination] =
            base + static_cast<std::uint32_t>(instruction.immediate);
        break;
    }

    case Opcode::addis:
    {
        const std::uint32_t base =
            instruction.base == 0 ? 0U : state.gpr[instruction.base];
        const std::uint32_t shifted_immediate =
            static_cast<std::uint32_t>(instruction.immediate) << 16U;
        state.gpr[instruction.destination] = base + shifted_immediate;
        break;
    }

    case Opcode::ori:
        state.gpr[instruction.destination] =
            state.gpr[instruction.source] |
            static_cast<std::uint32_t>(instruction.immediate);
        break;

    case Opcode::branch:
        if (instruction.link)
        {
            state.lr = fallthrough;
        }

        if (instruction.absolute)
        {
            next_cia = static_cast<std::uint32_t>(instruction.immediate);
        }
        else
        {
            next_cia = cia + static_cast<std::uint32_t>(instruction.immediate);
        }
        break;

    case Opcode::unsupported:
        return StepResult::unsupported_instruction;
    }

    state.cia = next_cia;
    return StepResult::executed;
}

RunResult EspressoCore::run(std::size_t max_steps)
{
    RunResult result{};

    while (result.steps < max_steps)
    {
        if (step() == StepResult::unsupported_instruction)
        {
            result.reason = StopReason::unsupported_instruction;
            return result;
        }

        ++result.steps;
    }

    result.reason = StopReason::instruction_limit;
    return result;
}

}
