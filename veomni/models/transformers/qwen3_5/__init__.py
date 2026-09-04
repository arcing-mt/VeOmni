# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ....utils.device import IS_MUSA_AVAILABLE, IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


# NPU branch is opt-in; everything else (CUDA, CPU-only) falls back to the GPU
# generated file. The GPU generated module imports cleanly without an active
# CUDA device, so a CPU-only environment (e.g. CI lint, doc build) can still
# register the class.


@MODELING_REGISTRY.register("qwen3_5")
def register_qwen3_5_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_npu import Qwen3_5ForConditionalGeneration, Qwen3_5Model
    else:
        from .generated import patched_modeling_qwen3_5_gpu as modeling_module

        if IS_MUSA_AVAILABLE:
            from ....ops.config.singleton import get_ops_config

            ops_config = get_ops_config()
            if ops_config is not None and (
                ops_config.rotary_pos_emb_implementation == "musa"
                or ops_config.rotary_pos_emb_vision_implementation == "musa"
            ):
                from .qwen3_5_musa_runtime_patch import install_qwen3_5_musa_rotary_patch

                install_qwen3_5_musa_rotary_patch(
                    modeling_module,
                    install_text=ops_config.rotary_pos_emb_implementation == "musa",
                    install_vision=ops_config.rotary_pos_emb_vision_implementation == "musa",
                )
        Qwen3_5ForConditionalGeneration = modeling_module.Qwen3_5ForConditionalGeneration
        Qwen3_5Model = modeling_module.Qwen3_5Model

    if "ForConditionalGeneration" in architecture:
        return Qwen3_5ForConditionalGeneration
    elif "Model" in architecture:
        return Qwen3_5Model
    else:
        return Qwen3_5ForConditionalGeneration


@MODELING_REGISTRY.register("qwen3_5_text")
def register_qwen3_5_text_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_npu import Qwen3_5ForCausalLM
    else:
        from .generated import patched_modeling_qwen3_5_gpu as modeling_module

        if IS_MUSA_AVAILABLE:
            from ....ops.config.singleton import get_ops_config

            ops_config = get_ops_config()
            if ops_config is not None and ops_config.rotary_pos_emb_implementation == "musa":
                from .qwen3_5_musa_runtime_patch import install_qwen3_5_musa_rotary_patch

                install_qwen3_5_musa_rotary_patch(modeling_module)
        Qwen3_5ForCausalLM = modeling_module.Qwen3_5ForCausalLM

    return Qwen3_5ForCausalLM
