# 11 — Graphics: GX2 / Latte -> host GPU

**Status:** later phase; do not implement before CPU, memory, loaders, and basic HLE work.

## Layers

```text
game/engine logic
    ->
models / materials / textures
    ->
GX2 API/state
    ->
Latte/GX2 guest GPU behavior
    ->
Affogato translation
    ->
Vulkan
    ->
host GPU
    ->
pixels
```

The game already owns its models, animation, scene graph, physics, and renderer logic. Affogato does not recreate them.

## Real Wii U

GX2 is the Wii U's main graphics processor/API environment in Wii U mode and is associated with Latte.

WiiUBrew documents the GX2 hardware as Radeon R7xx-family based and exposes GX2-related system-library functions through `gx2.rpl`.

## Affogato goal

Do not build a software rasterizer that mimics every physical Latte pipeline stage unless needed for debugging.

Instead, preserve guest-visible behavior and translate it to a host graphics API.

Possible conceptual mappings:

```text
guest vertex/index buffer -> VkBuffer
guest texture             -> VkImage
guest sampler             -> VkSampler
render target             -> Vulkan attachment/image
guest draw                 -> Vulkan draw command
guest shader               -> translated host shader / SPIR-V
guest synchronization      -> appropriate Vulkan synchronization
```

These mappings are conceptual, not guaranteed one-to-one.

## Research topics when this phase begins

- GX2 API surface used by games.
- command submission.
- guest buffer layouts.
- texture/surface formats.
- tiling/swizzling.
- shaders and instruction format.
- render targets.
- depth/stencil.
- blending.
- viewport/scissor.
- synchronization and cache-flush semantics.
- TV versus GamePad presentation.

## What the host GPU can do for Affogato

After correct translation, the host GPU should perform:

- vertex processing.
- primitive assembly.
- rasterization.
- texture sampling.
- fragment shading.
- depth/stencil tests.
- blending.
- final rendering/presentation.

## Sources

- WiiUBrew — GX2 hardware: https://wiiubrew.org/wiki/Hardware/GX2
- WiiUBrew — gx2.rpl: https://wiiubrew.org/wiki/Gx2.rpl
- WiiUBrew — Latte: https://wiiubrew.org/wiki/Hardware/Latte
