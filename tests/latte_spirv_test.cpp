#include "gpu/latte_spirv_vertex_subset.hpp"
#include "gpu/latte_spirv_fragment_subset.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

[[nodiscard]] affogato::gpu::LatteShaderIrProgram create_vertex_program()
{
    using namespace affogato::gpu;

    LatteAluOp2InstructionGroup move_group{};
    move_group.instructions.reserve(4U);
    for (std::uint8_t component = 0U; component < 4U; ++component)
    {
        LatteAluOp2Instruction move{};
        move.opcode = latte_alu_op2_move_opcode;
        move.source0_selector = 1U;
        move.source0_channel = component;
        move.destination_gpr = 3U;
        move.destination_component = component;
        move.write_mask = true;
        move.last_in_group = component == 3U;
        move_group.instructions.push_back(move);
    }

    LatteExportFields position_export{};
    position_export.type = LatteExportType::position;
    position_export.array_base = 0x3CU;
    position_export.read_write_gpr = 3U;
    position_export.burst_count = 1U;
    position_export.component_selectors = {0U, 1U, 2U, 3U};

    LatteShaderIrProgram program{};
    program.instructions.emplace_back(
        LatteShaderIrAluClause{{move_group}, {}, {}});
    program.instructions.emplace_back(
        LatteShaderIrExport{true, position_export});
    return program;
}

[[nodiscard]] affogato::gpu::LatteShaderIrProgram create_fragment_program()
{
    using namespace affogato::gpu;

    constexpr std::array<std::uint16_t, 4U> color_selectors{
        latte_alu_source_one_float_selector,
        latte_alu_source_zero_selector,
        latte_alu_source_one_float_selector,
        latte_alu_source_one_float_selector};

    LatteAluOp2InstructionGroup color_group{};
    color_group.instructions.reserve(color_selectors.size());
    for (std::uint8_t component = 0U; component < color_selectors.size(); ++component)
    {
        LatteAluOp2Instruction move{};
        move.opcode = latte_alu_op2_move_opcode;
        move.source0_selector = color_selectors[component];
        move.destination_gpr = 2U;
        move.destination_component = component;
        move.write_mask = true;
        move.last_in_group = component == 3U;
        color_group.instructions.push_back(move);
    }

    LatteExportFields pixel_export{};
    pixel_export.type = LatteExportType::pixel;
    pixel_export.array_base = 0U;
    pixel_export.read_write_gpr = 2U;
    pixel_export.burst_count = 1U;
    pixel_export.component_selectors = {0U, 1U, 2U, 3U};

    LatteShaderIrProgram program{};
    program.instructions.emplace_back(
        LatteShaderIrAluClause{{color_group}, {}, {}});
    program.instructions.emplace_back(
        LatteShaderIrExport{true, pixel_export});
    return program;
}

void write_module(const char* path, std::span<const std::uint32_t> module)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        throw std::runtime_error("Could not open SPIR-V output file");
    }

    output.write(
        reinterpret_cast<const char*>(module.data()),
        static_cast<std::streamsize>(module.size_bytes()));
    if (!output)
    {
        throw std::runtime_error("Could not write SPIR-V output file");
    }
}

[[nodiscard]] bool module_contains_opcode(
    std::span<const std::uint32_t> module,
    std::uint16_t expected_opcode)
{
    for (std::size_t word_offset = 5U; word_offset < module.size();)
    {
        const std::uint32_t instruction = module[word_offset];
        const std::size_t word_count = instruction >> 16U;
        if (word_count == 0U || word_count > module.size() - word_offset)
        {
            return false;
        }
        if (static_cast<std::uint16_t>(instruction & 0xFFFFU) ==
            expected_opcode)
        {
            return true;
        }
        word_offset += word_count;
    }
    return false;
}

[[nodiscard]] std::vector<std::uint32_t> read_float_constants(
    std::span<const std::uint32_t> module)
{
    std::vector<std::uint32_t> constants;
    for (std::size_t word_offset = 5U; word_offset < module.size();)
    {
        const std::uint32_t instruction = module[word_offset];
        const std::size_t word_count = instruction >> 16U;
        if (word_count == 0U || word_count > module.size() - word_offset)
        {
            throw std::runtime_error("Invalid SPIR-V while reading constants");
        }
        if (static_cast<std::uint16_t>(instruction & 0xFFFFU) == 43U &&
            word_count == 4U)
        {
            constants.push_back(module[word_offset + 3U]);
        }
        word_offset += word_count;
    }
    return constants;
}

} // namespace

int main(int argc, char* argv[])
{
    const auto program = create_vertex_program();
    const std::vector<std::uint32_t> module =
        affogato::gpu::emit_latte_spirv_vertex_position_mov_subset(
            program,
            {1U, 0U});
    assert(module.size() > 5U);
    assert(module[0U] == 0x07230203U);
    assert(module[1U] == 0x00010000U);
    assert(module[3U] > 1U);

    bool has_vertex_entry_point = false;
    bool has_input_load = false;
    bool has_component_extract = false;
    bool has_bitcast = false;
    bool has_position_store = false;
    for (std::size_t word_offset = 5U; word_offset < module.size();)
    {
        const std::uint32_t instruction = module[word_offset];
        const std::size_t word_count = instruction >> 16U;
        assert(word_count > 0U);
        assert(word_count <= module.size() - word_offset);

        const std::uint16_t opcode =
            static_cast<std::uint16_t>(instruction & 0xFFFFU);
        has_vertex_entry_point = has_vertex_entry_point || opcode == 15U;
        has_input_load = has_input_load || opcode == 61U;
        has_component_extract = has_component_extract || opcode == 81U;
        has_bitcast = has_bitcast || opcode == 124U;
        has_position_store = has_position_store || opcode == 62U;
        word_offset += word_count;
    }
    assert(has_vertex_entry_point);
    assert(has_input_load);
    assert(has_component_extract);
    assert(has_bitcast);
    assert(has_position_store);

    auto modified_program = program;
    auto& modified_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            modified_program.instructions[0U]);
    modified_clause.groups[0U].instructions[0U].source0_absolute = true;
    modified_clause.groups[0U].instructions[0U].source0_negate = true;
    const std::vector<std::uint32_t> modified_module =
        affogato::gpu::emit_latte_spirv_vertex_position_mov_subset(
            modified_program,
            {1U, 0U});
    assert(module_contains_opcode(modified_module, 199U));
    assert(module_contains_opcode(modified_module, 198U));

    auto unsupported_program = program;
    auto& unsupported_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            unsupported_program.instructions[0U]);
    unsupported_clause.groups[0U].instructions[0U].opcode =
        affogato::gpu::latte_alu_op2_add_float_opcode;
    bool rejected_unsupported_opcode = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_vertex_position_mov_subset(
                unsupported_program,
                {1U, 0U}));
    }
    catch (const std::runtime_error&)
    {
        rejected_unsupported_opcode = true;
    }
    assert(rejected_unsupported_opcode);

    auto add_program = program;
    auto& add_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            add_program.instructions[0U]);
    for (auto& operation : add_clause.groups[0U].instructions)
    {
        operation.opcode = affogato::gpu::latte_alu_op2_add_float_opcode;
        operation.source1_selector = 1U;
        operation.source1_channel = operation.destination_component;
    }
    const std::vector<std::uint32_t> add_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            add_program,
            {1U, 0U});
    assert(module_contains_opcode(add_module, 129U));
    assert(module_contains_opcode(add_module, 124U));

    add_clause.groups[0U].instructions[0U].source1_absolute = true;
    add_clause.groups[0U].instructions[0U].source1_negate = true;
    const std::vector<std::uint32_t> modified_add_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            add_program,
            {1U, 0U});
    assert(module_contains_opcode(modified_add_module, 199U));
    assert(module_contains_opcode(modified_add_module, 198U));

    auto multiply_program = add_program;
    auto& multiply_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            multiply_program.instructions[0U]);
    for (auto& operation : multiply_clause.groups[0U].instructions)
    {
        operation.opcode =
            affogato::gpu::latte_alu_op2_multiply_ieee_opcode;
        operation.source1_relative = false;
    }
    const std::vector<std::uint32_t> multiply_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            multiply_program,
            {1U, 0U});
    assert(module_contains_opcode(multiply_module, 133U));

    auto maximum_program = add_program;
    auto& maximum_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            maximum_program.instructions[0U]);
    for (auto& operation : maximum_clause.groups[0U].instructions)
    {
        operation.opcode =
            affogato::gpu::latte_alu_op2_maximum_dx10_opcode;
        operation.source1_relative = false;
    }
    const std::vector<std::uint32_t> maximum_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            maximum_program,
            {1U, 0U});
    assert(module_contains_opcode(maximum_module, 190U));
    assert(module_contains_opcode(maximum_module, 169U));

    auto minimum_program = add_program;
    auto& minimum_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            minimum_program.instructions[0U]);
    for (auto& operation : minimum_clause.groups[0U].instructions)
    {
        operation.opcode =
            affogato::gpu::latte_alu_op2_minimum_dx10_opcode;
        operation.source1_relative = false;
    }
    const std::vector<std::uint32_t> minimum_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            minimum_program,
            {1U, 0U});
    assert(module_contains_opcode(minimum_module, 184U));
    assert(module_contains_opcode(minimum_module, 169U));

    auto not_program = add_program;
    auto& not_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(not_program.instructions[0U]);
    for (auto& operation : not_clause.groups[0U].instructions)
    {
        operation.opcode = affogato::gpu::latte_alu_op2_not_integer_opcode;
        operation.source1_selector = 0xFFFFU;
        operation.source1_channel = 0xFFU;
        operation.source1_relative = false;
        operation.source1_absolute = false;
        operation.source1_negate = false;
    }
    const std::vector<std::uint32_t> not_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            not_program,
            {1U, 0U});
    assert(module_contains_opcode(not_module, 200U));

    constexpr std::array integer_operations{
        std::pair{
            affogato::gpu::latte_alu_op2_and_integer_opcode,
            std::uint16_t{199U}},
        std::pair{
            affogato::gpu::latte_alu_op2_or_integer_opcode,
            std::uint16_t{197U}},
        std::pair{
            affogato::gpu::latte_alu_op2_xor_integer_opcode,
            std::uint16_t{198U}},
        std::pair{
            affogato::gpu::latte_alu_op2_not_integer_opcode,
            std::uint16_t{200U}},
        std::pair{
            affogato::gpu::latte_alu_op2_add_integer_opcode,
            std::uint16_t{128U}},
        std::pair{
            affogato::gpu::latte_alu_op2_subtract_integer_opcode,
            std::uint16_t{130U}},
        std::pair{
            affogato::gpu::latte_alu_op2_maximum_signed_integer_opcode,
            std::uint16_t{175U}},
        std::pair{
            affogato::gpu::latte_alu_op2_minimum_signed_integer_opcode,
            std::uint16_t{179U}},
        std::pair{
            affogato::gpu::latte_alu_op2_maximum_unsigned_integer_opcode,
            std::uint16_t{174U}},
        std::pair{
            affogato::gpu::latte_alu_op2_minimum_unsigned_integer_opcode,
            std::uint16_t{178U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_equal_integer_opcode,
            std::uint16_t{170U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_not_equal_integer_opcode,
            std::uint16_t{171U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_greater_integer_opcode,
            std::uint16_t{173U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_greater_equal_integer_opcode,
            std::uint16_t{175U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_greater_unsigned_opcode,
            std::uint16_t{172U}},
        std::pair{
            affogato::gpu::latte_alu_op2_set_greater_equal_unsigned_opcode,
            std::uint16_t{174U}}};
    for (const auto& [latte_opcode, spirv_opcode] : integer_operations)
    {
        auto integer_operation_program = add_program;
        auto& integer_operation_clause = std::get<
            affogato::gpu::LatteShaderIrAluClause>(
                integer_operation_program.instructions[0U]);
        for (auto& operation : integer_operation_clause.groups[0U].instructions)
        {
            operation.opcode = latte_opcode;
            operation.source1_relative = false;
            operation.source1_absolute = false;
            operation.source1_negate = false;
        }

        const std::vector<std::uint32_t> integer_operation_module =
            affogato::gpu::emit_latte_spirv_vertex_position_subset(
                integer_operation_program,
                {1U, 0U});
        assert(module_contains_opcode(integer_operation_module, spirv_opcode));
        if (spirv_opcode >= 170U && spirv_opcode <= 179U)
        {
            assert(module_contains_opcode(integer_operation_module, 169U));
        }
    }

    auto integer_alu_program = add_program;
    auto& integer_alu_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            integer_alu_program.instructions[0U]);
    integer_alu_clause.groups.clear();
    for (std::size_t group_index = 0U; group_index < 4U; ++group_index)
    {
        auto group = add_clause.groups[0U];
        for (std::size_t lane = 0U; lane < group.instructions.size(); ++lane)
        {
            const std::size_t opcode_index = group_index * 4U + lane;
            if (opcode_index < integer_operations.size())
            {
                group.instructions[lane].opcode =
                    integer_operations[opcode_index].first;
                group.instructions[lane].source1_relative = false;
                group.instructions[lane].source1_absolute = false;
                group.instructions[lane].source1_negate = false;
            }
        }
        integer_alu_clause.groups.push_back(std::move(group));
    }
    const std::vector<std::uint32_t> integer_alu_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            integer_alu_program,
            {1U, 0U});

    auto conditional_move_program = program;
    auto& conditional_move_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            conditional_move_program.instructions[0U]);
    conditional_move_clause.groups.clear();
    affogato::gpu::LatteAluOp3InstructionGroup conditional_move_group{};
    for (std::uint8_t lane = 0U; lane < 4U; ++lane)
    {
        affogato::gpu::LatteAluOp3Instruction operation{};
        if (lane == 0U)
        {
            operation.opcode =
                affogato::gpu::latte_alu_op3_conditional_move_equal_opcode;
        }
        else if (lane == 1U)
        {
            operation.opcode =
                affogato::gpu::latte_alu_op3_conditional_move_greater_opcode;
        }
        else
        {
            operation.opcode = affogato::gpu::
                latte_alu_op3_conditional_move_greater_equal_opcode;
        }
        operation.source0_selector = 1U;
        operation.source0_channel = lane;
        operation.source1_selector =
            affogato::gpu::latte_alu_source_one_float_selector;
        operation.source2_selector =
            affogato::gpu::latte_alu_source_zero_selector;
        operation.destination_gpr = 3U;
        operation.destination_component = lane;
        operation.last_in_group = lane == 3U;
        conditional_move_group.instructions.push_back(operation);
    }
    conditional_move_clause.op3_groups.push_back(conditional_move_group);

    const std::vector<std::uint32_t> conditional_move_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            conditional_move_program,
            {1U, 0U});
    assert(module_contains_opcode(conditional_move_module, 180U));
    assert(module_contains_opcode(conditional_move_module, 186U));
    assert(module_contains_opcode(conditional_move_module, 190U));
    assert(module_contains_opcode(conditional_move_module, 169U));

    auto integer_conditional_move_program = conditional_move_program;
    auto& integer_conditional_move_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            integer_conditional_move_program.instructions[0U]);
    auto& integer_conditional_move_instructions =
        integer_conditional_move_clause.op3_groups[0U].instructions;
    integer_conditional_move_instructions[0U].opcode = affogato::gpu::
        latte_alu_op3_conditional_move_equal_integer_opcode;
    integer_conditional_move_instructions[1U].opcode = affogato::gpu::
        latte_alu_op3_conditional_move_greater_integer_opcode;
    integer_conditional_move_instructions[2U].opcode = affogato::gpu::
        latte_alu_op3_conditional_move_greater_equal_integer_opcode;
    integer_conditional_move_instructions[3U].opcode = affogato::gpu::
        latte_alu_op3_conditional_move_greater_equal_integer_opcode;
    const std::vector<std::uint32_t> integer_conditional_move_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            integer_conditional_move_program,
            {1U, 0U});
    assert(module_contains_opcode(integer_conditional_move_module, 170U));
    assert(module_contains_opcode(integer_conditional_move_module, 173U));
    assert(module_contains_opcode(integer_conditional_move_module, 175U));
    assert(module_contains_opcode(integer_conditional_move_module, 169U));

    auto mixed_program = program;
    auto& mixed_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            mixed_program.instructions[0U]);
    mixed_clause.groups.clear();
    affogato::gpu::LatteAluVectorInstructionGroup mixed_group{};
    mixed_group.instructions.emplace_back(
        std::get<affogato::gpu::LatteShaderIrAluClause>(
            program.instructions[0U]).groups[0U].instructions[0U]);
    for (std::size_t lane = 1U; lane < 4U; ++lane)
    {
        mixed_group.instructions.emplace_back(
            conditional_move_group.instructions[lane]);
    }
    mixed_clause.vector_groups.push_back(std::move(mixed_group));
    const std::vector<std::uint32_t> mixed_module =
        affogato::gpu::emit_latte_spirv_vertex_position_subset(
            mixed_program,
            {1U, 0U});
    assert(module_contains_opcode(mixed_module, 186U));
    assert(module_contains_opcode(mixed_module, 190U));
    assert(module_contains_opcode(mixed_module, 169U));

    conditional_move_clause.op3_groups[0U]
        .instructions[0U].source2_relative = true;
    bool rejected_relative_op3_source = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_vertex_position_subset(
                conditional_move_program,
                {1U, 0U}));
    }
    catch (const std::runtime_error&)
    {
        rejected_relative_op3_source = true;
    }
    assert(rejected_relative_op3_source);

    add_clause.groups[0U].instructions[0U].source1_relative = true;
    bool rejected_relative_add_source = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_vertex_position_subset(
                add_program,
                {1U, 0U}));
    }
    catch (const std::runtime_error&)
    {
        rejected_relative_add_source = true;
    }
    assert(rejected_relative_add_source);

    const auto fragment_program = create_fragment_program();
    const std::vector<std::uint32_t> fragment_module =
        affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
            fragment_program);
    assert(fragment_module[0U] == 0x07230203U);
    assert(module_contains_opcode(fragment_module, 15U));
    assert(module_contains_opcode(fragment_module, 16U));
    assert(module_contains_opcode(fragment_module, 44U));
    assert(module_contains_opcode(fragment_module, 62U));

    auto swizzled_fragment = fragment_program;
    auto& swizzled_export = std::get<
        affogato::gpu::LatteShaderIrExport>(
            swizzled_fragment.instructions[1U]);
    swizzled_export.fields.component_selectors = {1U, 2U, 0U, 4U};
    const std::vector<std::uint32_t> swizzled_fragment_module =
        affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
            swizzled_fragment);
    assert(read_float_constants(swizzled_fragment_module) ==
        (std::vector<std::uint32_t>{
            0U, 0x3F800000U, 0x3F800000U, 0U}));

    auto one_fragment = fragment_program;
    std::get<affogato::gpu::LatteShaderIrExport>(
        one_fragment.instructions[1U]).fields.component_selectors =
        {5U, 5U, 5U, 5U};
    const std::vector<std::uint32_t> one_fragment_module =
        affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
            one_fragment);
    assert(read_float_constants(one_fragment_module) ==
        (std::vector<std::uint32_t>{
            0x3F800000U,
            0x3F800000U,
            0x3F800000U,
            0x3F800000U}));

    auto masked_fragment = fragment_program;
    std::get<affogato::gpu::LatteShaderIrExport>(
        masked_fragment.instructions[1U]).fields.component_selectors[3U] = 7U;
    bool rejected_masked_fragment_component = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
                masked_fragment));
    }
    catch (const std::runtime_error&)
    {
        rejected_masked_fragment_component = true;
    }
    assert(rejected_masked_fragment_component);

    auto invalid_fragment = fragment_program;
    std::get<affogato::gpu::LatteShaderIrExport>(
        invalid_fragment.instructions[1U]).fields.component_selectors[3U] = 8U;
    bool rejected_invalid_fragment_component = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
                invalid_fragment));
    }
    catch (const std::runtime_error&)
    {
        rejected_invalid_fragment_component = true;
    }
    assert(rejected_invalid_fragment_component);

    auto unsupported_fragment = fragment_program;
    auto& unsupported_fragment_clause = std::get<
        affogato::gpu::LatteShaderIrAluClause>(
            unsupported_fragment.instructions[0U]);
    unsupported_fragment_clause.groups[0U].instructions[0U].source0_selector = 1U;
    bool rejected_fragment_register_input = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
                unsupported_fragment));
    }
    catch (const std::runtime_error&)
    {
        rejected_fragment_register_input = true;
    }
    assert(rejected_fragment_register_input);

    unsupported_fragment = fragment_program;
    std::get<affogato::gpu::LatteShaderIrExport>(
        unsupported_fragment.instructions[1U]).fields.array_base = 1U;
    bool rejected_nonzero_mrt = false;
    try
    {
        static_cast<void>(
            affogato::gpu::emit_latte_spirv_fragment_constant_color_subset(
                unsupported_fragment));
    }
    catch (const std::runtime_error&)
    {
        rejected_nonzero_mrt = true;
    }
    assert(rejected_nonzero_mrt);

    if (argc == 13 && std::string_view(argv[1]) == "--write-modules")
    {
        write_module(argv[2], module);
        write_module(argv[3], modified_module);
        write_module(argv[4], add_module);
        write_module(argv[5], multiply_module);
        write_module(argv[6], maximum_module);
        write_module(argv[7], minimum_module);
        write_module(argv[8], conditional_move_module);
        write_module(argv[9], integer_conditional_move_module);
        write_module(argv[10], mixed_module);
        write_module(argv[11], integer_alu_module);
        write_module(argv[12], fragment_module);
    }
    return 0;
}
