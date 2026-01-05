# LoRA Project

Lightweight example project showing how to fine-tune TinyLlama using PEFT/LoRA with 4-bit quantization. The repository contains a runnable Python script and a Jupyter notebook that demonstrate training an adapter and evaluating it on the GSM8K dataset.

**Warning:** This code expects an NVIDIA GPU and a CUDA-enabled environment. Quantization and bf16 usage may require specific hardware and driver support.

## Files
- **lora.py**: Main Python script that installs dependencies, prepares the model/tokenizer, configures LoRA, trains with `trl.SFTTrainer`, and runs evaluation.
- **LORA.ipynb**: Notebook version of the same workflow, broken into cells for step-by-step execution.

## Requirements
- Python 3.10+ (recommended)
- NVIDIA GPU with CUDA and proper drivers
- ~20 GB free GPU memory recommended for TinyLlama 1.1B with 4-bit quantization and adapters (actual needs vary)

Recommended Python packages (the notebook/script calls these):

```
torch
datasets
transformers
peft
bitsandbytes
trl
```

You can install the core packages with:

```bash
pip install torch datasets transformers peft bitsandbytes trl
```

If you need a pinned list, tell me and I will add a `requirements.txt`.

## Quickstart — Notebook
1. Open [LORA.ipynb](LORA.ipynb) in Jupyter/Colab/VS Code.  
2. Run cells in order (the notebook installs dependencies, loads the model, trains LoRA adapters, saves them to `./results-adapter`, then runs evaluation and generation examples).

## Quickstart — Script
Run the `lora.py` script with a CUDA-enabled Python environment. Example:

```bash
python lora.py
```

Notes in the script/notebook you may want to customize:
- `model_name` — current value: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
- LoRA hyperparams (`r`, `lora_alpha`, `lora_dropout`) in `LoraConfig`  
- Dataset split names and tokenizer `max_length`  
- `adapter_path` / output directories (`./results-adapter`)

## Training and Evaluation
- Training is performed via `trl.SFTTrainer` in the code. Adjust `TrainingArguments` for batch size, accumulation, learning rate, number of epochs, and bf16/fp16 settings.
- Evaluation computes perplexity on a small GSM8K slice and prints generation examples for comparison between the base and tuned model.

## Troubleshooting
- If you run out of GPU memory, reduce `per_device_train_batch_size` or enable gradient accumulation with `gradient_accumulation_steps` (already used in the example).  
- If bf16 is not supported on your GPU, change `bf16=True` to `fp16=True` or use CPU fallback (slower).  
- BitsAndBytes + 4-bit quantization requires `bitsandbytes` and compatible CUDA toolkit/drivers.

## Next steps
- I can add a `requirements.txt`, example SLURM/launcher script, or convert the notebook into a smaller reproducible example. Tell me which you want next.

---
