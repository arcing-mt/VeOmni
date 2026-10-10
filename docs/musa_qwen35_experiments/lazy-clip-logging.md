---
orphan: true
---

# [dist, logging] fix: defer gradient norm debug formatting

Use lazy logger argument formatting instead of tensor f-strings, preserving reduction groups, norm and clipping math. This removes formatting work when debug logging is disabled. Full50 experiments did not establish an independent step-time gain; this PR makes no performance promise.

Validation: the unchanged norm/clip math was already covered by controlled update gates and full training. The publication reruns make quality; no new GPU run or independent performance claim is added.

## Related material

[PR3](https://github.com/arcing-mt/VeOmni/pull/3), [PR4](https://github.com/arcing-mt/VeOmni/pull/4), and [TE compatibility branch](https://sh-code.mthreads.com/ai/TransformerEngine/-/tree/gl/mudnn3.4-torch2.11).
