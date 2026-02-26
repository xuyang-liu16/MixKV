from mixkv.qwen_model import (
    adakv_qwen_forward,
    prepare_inputs_for_generation_qwen,
    qwen_flash_attn2_forward_AdaKV,
    qwen_flash_attn2_forward_Mask,
    qwen_flash_attn2_forward_MixSparseMM,
    qwen_flash_attn2_forward_PyramidKV,
    qwen_flash_attn2_forward_SnapKV,
    qwen_flash_attn2_forward_SparseMM,
)

prepare_inputs_for_generation_qwen2_5_vl = prepare_inputs_for_generation_qwen
adakv_qwen2_5_vl_forward = adakv_qwen_forward
