from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(model, accelerator: Accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_model_hook(models, weights, output_dir, accelerator: Accelerator):
    if accelerator.is_main_process:
        for model in models:
            breakpoint()
