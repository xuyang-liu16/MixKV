from importlib.metadata import version
import transformers

from mixkv.mistral_model import mistral_flash_attn2_forward_AdaKV, mistral_flash_attn2_forward_MixSparseMM,  mistral_flash_attn2_forward_PyramidKV, mistral_flash_attn2_forward_SnapKV, \
                                   mistral_flash_attn2_forward_SparseMM, mistral_flash_attn2_forward_Mask
from mixkv.mistral_model import prepare_inputs_for_generation_mistral_new, adaptive_MistralModel_forward

from mixkv.qwen2_self import flash_attn_forward_adakv, flash_attn_forward_snapkv, qwen2_forward_adakv,flash_attn_forward_pyramidkv
from mixkv.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_MixSparseMM, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from mixkv.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward
from mixkv.qwen2_5_vl_model import (
    adakv_qwen2_5_vl_forward,
    prepare_inputs_for_generation_qwen2_5_vl,
    qwen_flash_attn2_forward_AdaKV as qwen2_5_vl_flash_attn2_forward_AdaKV,
    qwen_flash_attn2_forward_Mask as qwen2_5_vl_flash_attn2_forward_Mask,
    qwen_flash_attn2_forward_MixSparseMM as qwen2_5_vl_flash_attn2_forward_MixSparseMM,
    qwen_flash_attn2_forward_PyramidKV as qwen2_5_vl_flash_attn2_forward_PyramidKV,
    qwen_flash_attn2_forward_SnapKV as qwen2_5_vl_flash_attn2_forward_SnapKV,
    qwen_flash_attn2_forward_SparseMM as qwen2_5_vl_flash_attn2_forward_SparseMM,
)





def replace_mistral(method):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SparseMM
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_MixSparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new



def replace_qwen(method):
    if method == 'snapkv':
        print("Using SnapKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_Mask
    
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_MixSparseMM
    if method not in ["fullkv"]:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen


def _require_qwen2_5_vl_module():
    module = getattr(transformers.models, "qwen2_5_vl", None)
    if module is None:
        raise RuntimeError(
            "Qwen2.5-VL support requires a transformers version that provides "
            "transformers.models.qwen2_5_vl."
        )
    modeling = getattr(module, "modeling_qwen2_5_vl", None)
    if modeling is None:
        raise RuntimeError(
            "Qwen2.5-VL support requires transformers.models.qwen2_5_vl.modeling_qwen2_5_vl."
        )
    return modeling


def _get_qwen2_5_vl_class(module, candidates, role):
    for name in candidates:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        f"Qwen2.5-VL support could not find {role} in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl. "
        f"Tried: {', '.join(candidates)}."
    )


def replace_qwen2_5_vl(method):
    qwen2_5_vl = _require_qwen2_5_vl_module()
    flash_attn_cls = _get_qwen2_5_vl_class(
        qwen2_5_vl,
        ("Qwen2_5_VLFlashAttention2", "Qwen2_5_VLFlashAttention"),
        "FlashAttention2 implementation",
    )
    model_cls = _get_qwen2_5_vl_class(
        qwen2_5_vl,
        ("Qwen2_5_VLModel", "Qwen2_5_VLBaseModel"),
        "model implementation",
    )
    for_causal_cls = _get_qwen2_5_vl_class(
        qwen2_5_vl,
        ("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForCausalLM"),
        "generation implementation",
    )
    if method == 'snapkv':
        print("Using SnapKV!")
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_PyramidKV

    if method == "adakv":
        print("Using AdaKV!")
        model_cls.forward = adakv_qwen2_5_vl_forward
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        model_cls.forward = adakv_qwen2_5_vl_forward
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_Mask

    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        model_cls.forward = adakv_qwen2_5_vl_forward
        flash_attn_cls.forward = qwen2_5_vl_flash_attn2_forward_MixSparseMM

    if method not in ["fullkv"]:
        for_causal_cls.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_5_vl

def replace_internvl(method):
    if method=="adakv":
        print("Using Adakv")
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward=qwen2_forward_adakv
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_adakv
    elif method=="snapkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_snapkv
        print("Using Snapkv")
    elif method=="pyramidkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_pyramidkv
        print("Using pyramidkv")
