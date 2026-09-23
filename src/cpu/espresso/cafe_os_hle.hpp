#pragma once

namespace affogato::cpu::espresso
{

class HleDispatcher;

void register_coreinit_hle(HleDispatcher& dispatcher);

}
