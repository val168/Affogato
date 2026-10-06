#include "cpu/espresso/nn_olv_hle.hpp"

#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

namespace affogato::cpu::espresso
{

void initialize_downloaded_data_base(GuestMemory& memory, std::uint32_t self)
{
    memory.zero_fill(self, downloaded_data_base_size);
}

void initialize_downloaded_topic_data(GuestMemory& memory, std::uint32_t self)
{
    // Validate all bytes before writing either field, so a boundary fault
    // cannot leave the first member modified and the second untouched.
    memory.validate_write_range(self, downloaded_topic_data_initialized_size);
    memory.write32_be(self + downloaded_topic_data_unk1_offset, 0U);
    memory.write32_be(self + downloaded_topic_data_community_id_offset, 0U);
}

void initialize_download_post_data_list_param(GuestMemory& memory, std::uint32_t self)
{
    memory.zero_fill(self, download_post_data_list_param_size);
}

void register_nn_olv_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "nn_olv.rpl",
        "__ct__Q3_2nn3olv18DownloadedPostDataFv",
        [](EspressoCore& core) {
            std::uint32_t self = core.state.gpr[3];
            if (self == 0)
            {
                self = core.allocate_guest_memory(downloaded_post_data_size, 8U);
                if (self == 0)
                {
                    core.state.gpr[3] = 0;
                    return;
                }
            }

            // Validate every byte this constructor will write before changing
            // guest memory. The derived object's unused padding is untouched.
            core.memory.validate_write_range(self, downloaded_post_data_initialized_size);
            initialize_downloaded_data_base(core.memory, self);

            // No GHS RTTI/vtable allocation exists yet. Base initialization
            // deliberately leaves this word zero rather than publishing a fake
            // host or guest pointer.
            core.memory.write32_be(
                self + downloaded_post_data_community_id_offset, 0U);
            core.memory.write32_be(
                self + downloaded_post_data_empathy_count_offset, 0U);
            core.memory.write32_be(
                self + downloaded_post_data_comment_count_offset, 0U);
            core.state.gpr[3] = self;
        });

    dispatcher.register_function(
        "nn_olv.rpl",
        "__ct__Q3_2nn3olv19DownloadedTopicDataFv",
        [](EspressoCore& core) {
            std::uint32_t self = core.state.gpr[3];
            if (self == 0)
            {
                self = core.allocate_guest_memory(downloaded_topic_data_size, 8U);
                if (self == 0)
                {
                    core.state.gpr[3] = 0;
                    return;
                }
            }

            initialize_downloaded_topic_data(core.memory, self);
            core.state.gpr[3] = self;
        });

    dispatcher.register_function(
        "nn_olv.rpl",
        "__ct__Q3_2nn3olv25DownloadPostDataListParamFv",
        [](EspressoCore& core) {
            const std::uint32_t self = core.state.gpr[3];
            initialize_download_post_data_list_param(core.memory, self);
            core.state.gpr[3] = self;
        });
}

}
