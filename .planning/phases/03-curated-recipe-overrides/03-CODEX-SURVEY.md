## 1. Recipe YAML field inventory

The inventory covers the 27 top-level YAML files in `/home/josh/spark-vllm-docker/recipes/`; cluster subdirectories were excluded.

| key | nesting path | type | how many recipes use it | example value | consumed by run-recipe.py how |
|---|---|---:|---:|---|---|
| `recipe_version` | `recipe_version` | string | 27 | `"1"` | Required; converted to string and checked against supported version `"1"` by [`run-recipe.py`](/home/josh/spark-vllm-docker/run-recipe.py). |
| `name` | `name` | string | 27 | `DeepSeek-V4-Flash` | Required; displayed in listings/status and written into the generated script comment. |
| `description` | `description` | string | 27 | `vLLM serving ...` | Optional; defaults to `""`; displayed only. |
| `model` | `model` | string | 27 | `openai/gpt-oss-120b` | Optional; defaults to `None`; used by setup/download/existence checks. It is not substituted automatically into `command`. |
| `container` | `container` | string | 27 | `vllm-node` | Required; selected unless overridden by CLI `--container`, then passed to image build/check and `launch-cluster.sh`. |
| `cluster_only` | `cluster_only` | boolean | 12 | `true` | Optional, default `false`; blocks solo execution. |
| `solo_only` | `solo_only` | boolean | 12 | `true` | Optional, default `false`; blocks cluster execution. |
| `mods` | `mods` | list of strings | 20 | `["mods/diffusiongemma"]` | Optional, default `[]`; each path is resolved relative to the repository and passed as `--apply-mod` to `launch-cluster.sh`. |
| `build_args` | `build_args` | list of strings | 1 | `["--exp-mxfp4"]` | Retrieved with default `[]`; appended to `build-and-copy.sh`. |
| `defaults` | `defaults` | mapping | 27 | `{port: 8000, host: 0.0.0.0, ...}` | Optional, default `{}`; merged with CLI overrides, then supplied to Python `str.format(**params)` for `command`. |
| `port` | `defaults.port` | integer | 27 | `8000` | Command placeholder; directly overridable with `--port`. |
| `host` | `defaults.host` | string | 27 | `0.0.0.0` | Command placeholder; directly overridable with `--host`. |
| `tensor_parallel` | `defaults.tensor_parallel` | integer | 22 | `2` | Command placeholder; directly overridable with `--tensor-parallel`/`--tp`; forced to `1` in solo mode unless explicitly overridden. Present but not referenced by the commands in [`nemotron-3-nano-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-nano-nvfp4.yaml) and [`openai-gpt-oss-120b.yaml`](/home/josh/spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml). |
| `gpu_memory_utilization` | `defaults.gpu_memory_utilization` | number: float or integer | 27 | `0.8`; also `108` | Command placeholder; directly overridable with `--gpu-memory-utilization`. Usually supplies `--gpu-memory-utilization`, but supplies patched `--gpu-memory-utilization-gb` in two recipes. |
| `max_model_len` | `defaults.max_model_len` | integer | 26 | `262144` | Command placeholder; directly overridable with `--max-model-len`. |
| `max_num_batched_tokens` | `defaults.max_num_batched_tokens` | integer | 14 | `8192` | Command placeholder only; no dedicated CLI override in `run-recipe.py`. |
| `max_num_seqs` | `defaults.max_num_seqs` | integer | 8 | `10` | Command placeholder only; no dedicated CLI override. |
| `block_size` | `defaults.block_size` | integer | 1 | `256` | Command placeholder only. |
| `num_speculative_tokens` | `defaults.num_speculative_tokens` | integer | 1 | `2` | Command placeholder embedded in `--speculative-config`. |
| `served_model_name` | `defaults.served_model_name` | string | 1 | `glm-4.7-flash` | Command placeholder used by `--served-model-name`. |
| `kv_cache_memory_bytes` | `defaults.kv_cache_memory_bytes` | integer | 1 | `2415919104` | Command placeholder used by patched `--kv-cache-memory-bytes`. |
| `env` | `env` | mapping or null | 26 | `{VLLM_MARLIN_USE_ATOMIC_ADD: 1}` | Optional, default `{}` only when the key is absent; nonempty mappings become shell `export KEY="VALUE"` lines. The null value in [`glm-4.7-flash-awq.yaml`](/home/josh/spark-vllm-docker/recipes/glm-4.7-flash-awq.yaml) is falsey and produces no exports. |
| `DG_JIT_USE_NVRTC` | `env.DG_JIT_USE_NVRTC` | string | 1 | `"0"` | Exported verbatim as an environment variable. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `env.VLLM_ALLOW_LONG_MAX_MODEL_LEN` | string or integer | 2 | `"1"` | Exported after string interpolation; quoted and unquoted YAML forms both become `export ...="1"`. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `env.VLLM_USE_BREAKABLE_CUDAGRAPH` | string | 1 | `"0"` | Exported verbatim. |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `env.VLLM_FLASHINFER_ALLREDUCE_BACKEND` | string | 1 | `trtllm` | Exported verbatim. |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `env.VLLM_MARLIN_USE_ATOMIC_ADD` | integer | 9 | `1` | Exported as `"1"`. |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `env.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | integer | 1 | `0` | Exported as `"0"`. |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `env.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | string | 1 | `"1"` | Exported verbatim. |
| `command` | `command` | multiline string | 27 | `vllm serve ... {host} ...` | Required; formatted with merged defaults/overrides. Extra CLI arguments after `--` are shell-quoted and appended. In solo or no-Ray mode, `--distributed-executor-backend` is removed; with `--ray`, Ray is preserved or appended. |

Mechanically stable fields are `recipe_version`, `model`, `container`, `cluster_only`, `solo_only`, `mods`, `build_args`, `defaults`, and `env`, subject to their documented optional defaults in [`run-recipe.py`](/home/josh/spark-vllm-docker/run-recipe.py). The named entries under `defaults` are mechanically usable placeholders when the corresponding `{name}` occurs in `command`.

Free-form fields are `name`, `description`, the contents of `command`, mod path strings, build arguments, model/container strings, environment-variable names and values, and any future keys placed in `defaults`.

`gpu_memory_utilization` changes meaning between files. In 25 recipes it is a fraction used with `--gpu-memory-utilization`; in [`qwen3.5-397b-int4-autoround.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.5-397b-int4-autoround.yaml) and [`step-3.7-flash-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/step-3.7-flash-fp8.yaml), the value `108` means GiB and is passed to patched `--gpu-memory-utilization-gb`. The first recipe explicitly comments that it is GiB rather than a percentage.

[`nemotron-3-super-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-super-nvfp4.yaml) contains `container: vllm-node` twice. Both occurrences have the same value; PyYAML exposes only one resulting key.

## 2. Recipe values table

“Other notable flags” excludes host, port, and flags already represented by dedicated columns.

| recipe file | model id | gpu_memory_utilization | max_model_len | max_num_batched_tokens | tensor_parallel | kv-cache-dtype | tool-call-parser | reasoning-parser | attention-backend | other notable flags | mods | solo_only |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| [`deepseek-v4-flash.yaml`](/home/josh/spark-vllm-docker/recipes/deepseek-v4-flash.yaml) | `deepseek-ai/DeepSeek-V4-Flash` | 0.8 | 500000 | 8192 | 2 | `fp8` | `deepseek_v4` | `deepseek_v4` | absent | block size 256; max seqs 4; MTP 2; prefix cache; DeepSeek-v4 tokenizer; InstantTensor; Ray | `[]` | absent |
| [`diffusion-gemma-bf16-thinking.yaml`](/home/josh/spark-vllm-docker/recipes/diffusion-gemma-bf16-thinking.yaml) | `google/diffusiongemma-26B-A4B-it` | 0.8 | 262144 | absent | absent | absent | `gemma4` | `gemma4` | `TRITON_ATTN` | max seqs 10; diffusion canvas 256; thinking true; prefix cache; fastsafetensors; Triton MoE | `mods/diffusiongemma` | true |
| [`diffusion-gemma-bf16.yaml`](/home/josh/spark-vllm-docker/recipes/diffusion-gemma-bf16.yaml) | `google/diffusiongemma-26B-A4B-it` | 0.8 | 262144 | absent | absent | absent | `gemma4` | `gemma4` | `TRITON_ATTN` | max seqs 10; diffusion canvas 256; thinking false; fixed chat template; prefix cache; Triton MoE | `mods/diffusiongemma` | true |
| [`diffusion-gemma-nvfp4-thinking.yaml`](/home/josh/spark-vllm-docker/recipes/diffusion-gemma-nvfp4-thinking.yaml) | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | 0.8 | 262144 | absent | absent | absent | `gemma4` | `gemma4` | `TRITON_ATTN` | diffusion canvas 256; thinking true; prefix cache; fastsafetensors | `mods/diffusiongemma` | true |
| [`diffusion-gemma-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/diffusion-gemma-nvfp4.yaml) | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | 0.8 | 262144 | absent | absent | absent | `gemma4` | `gemma4` | `TRITON_ATTN` | diffusion canvas 256; thinking false; fixed chat template; prefix cache | `mods/diffusiongemma` | true |
| [`gemma4-26b-a4b-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/gemma4-26b-a4b-nvfp4.yaml) | `nvidia/Gemma-4-26B-A4B-NVFP4` | 0.7 | 262144 | 8192 | 2 | `fp8` | `gemma4` | `gemma4` | absent | InstantTensor; prefix cache; 4-token MTP assistant; Ray | absent | true |
| [`gemma4-26b-a4b.yaml`](/home/josh/spark-vllm-docker/recipes/gemma4-26b-a4b.yaml) | `google/gemma-4-26B-A4B-it` | 0.8 | 262144 | 8192 | 2 | `fp8` | `gemma4` | `gemma4` | absent | online `--quantization fp8`; safetensors; prefix cache; Ray | absent | false |
| [`glm-4.7-flash-awq.yaml`](/home/josh/spark-vllm-docker/recipes/glm-4.7-flash-awq.yaml) | `cyankiwi/GLM-4.7-Flash-AWQ-4bit` | 0.8 | 202752 | 4096 | 1 | absent | `glm47` | `glm45` | absent | max seqs 64; served name `glm-4.7-flash` | absent | false |
| [`minimax-m2-awq.yaml`](/home/josh/spark-vllm-docker/recipes/minimax-m2-awq.yaml) | `QuantTrio/MiniMax-M2-AWQ` | 0.8 | 128000 | absent | 2 | absent | `minimax_m2` | `minimax_m2` | absent | fastsafetensors; Ray | `[]` | absent |
| [`minimax-m2.5-awq.yaml`](/home/josh/spark-vllm-docker/recipes/minimax-m2.5-awq.yaml) | `cyankiwi/MiniMax-M2.5-AWQ-4bit` | 0.8 | 128000 | absent | 2 | absent | `minimax_m2` | `minimax_m2` | absent | trust remote code; fastsafetensors; Ray | `[]` | absent |
| [`minimax-m2.7-awq.yaml`](/home/josh/spark-vllm-docker/recipes/minimax-m2.7-awq.yaml) | `cyankiwi/MiniMax-M2.7-AWQ-4bit` | 0.8 | 196608 | absent | 2 | absent | `minimax_m2` | `minimax_m2` | absent | trust remote code; fastsafetensors; Ray | `[]` | absent |
| [`nemotron-3-nano-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-nano-nvfp4.yaml) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 0.7 | 262144 | absent | 1 | `fp8` | `qwen3_coder` | `nano_v3` | absent | CUTLASS MoE; custom reasoning-parser plugin; prefix cache; fastsafetensors | `mods/nemotron-nano` | true |
| [`nemotron-3-super-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-super-nvfp4.yaml) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 0.8 | 262144 | absent | 2 | `fp8` | `qwen3_coder` | `nemotron_v3` | `TRITON_ATTN` | max seqs 10; CUTLASS MoE; Mamba cache float32; prefix cache; Ray | absent | false |
| [`openai-gpt-oss-120b.yaml`](/home/josh/spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml) | `openai/gpt-oss-120b` | 0.8 | absent | 8192 | 1 | `fp8` | `openai` | `openai_gptoss` | `FLASHINFER` | MXFP4/CUTLASS; selected MXFP4 layers; prefix cache; fastsafetensors | `[]` | true |
| [`qwen3-coder-next-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3-coder-next-fp8.yaml) | `Qwen/Qwen3-Coder-Next-FP8` | 0.8 | 131072 | absent | 2 | `fp8` | `qwen3_coder` | absent | `flashinfer` | fastsafetensors; prefix cache; Ray | absent | absent |
| [`qwen3-coder-next-int4-autoround.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3-coder-next-int4-autoround.yaml) | `Intel/Qwen3-Coder-Next-int4-AutoRound` | 0.8 | 262144 | absent | absent | absent | `qwen3_coder` | absent | absent | fastsafetensors; prefix cache | `mods/fix-qwen3-next-autoround` | true |
| [`qwen3.5-122b-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.5-122b-fp8.yaml) | `Qwen/Qwen3.5-122B-A10B-FP8` | 0.8 | 262144 | 8192 | 2 | absent | `qwen3_coder` | `qwen3` | absent | unsloth chat template; fastsafetensors; prefix cache; Ray | `mods/fix-qwen3.5-chat-template` | absent |
| [`qwen3.5-122b-int4-autoround.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.5-122b-int4-autoround.yaml) | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` | 0.8 | 262144 | 8192 | 2 | absent | `qwen3_xml` | `qwen3` | absent | trust remote code; unsloth chat template; fastsafetensors; prefix cache; Ray | `mods/fix-qwen3.5-chat-template` | absent |
| [`qwen3.5-35b-a3b-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.5-35b-a3b-fp8.yaml) | `Qwen/Qwen3.5-35B-A3B-FP8` | 0.8 | 262144 | 16384 | 2 | `fp8` | `qwen3_coder` | absent | `flashinfer` | unsloth chat template; fastsafetensors; prefix cache; Ray | `mods/fix-qwen3-coder-next`, `mods/fix-qwen3.5-chat-template` | absent |
| [`qwen3.5-397b-int4-autoround.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.5-397b-int4-autoround.yaml) | `Intel/Qwen3.5-397B-A17B-int4-AutoRound` | 108 GiB | 262144 | 4176 | 2 | `fp8` | `qwen3_xml` | `qwen3` | absent | max seqs 2; patched GPU-GiB and KV-byte flags; InstantTensor; unsloth template; prefix cache; Ray | four mods: chat-template, GPU-GiB, KV cleanup, drop-caches | absent |
| [`qwen3.6-35b-a3b-fp8-dflash.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8-dflash.yaml) | `Qwen/Qwen3.6-35B-A3B-FP8` | 0.8 | 262144 | 16384 | 2 | absent | `qwen3_xml` | `qwen3` | `flash_attn` | DFlash speculative model, 15 tokens; fixed chat template; fastsafetensors; Ray | `mods/fix-qwen3.6-chat-template` | absent |
| [`qwen3.6-35b-a3b-fp8-solo.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8-solo.yaml) | `Qwen/Qwen3.6-35B-A3B-FP8` | 0.55 | 262144 | 16384 | 1 | `fp8` | `qwen3_xml` | `qwen3` | `flashinfer` | served-name aliases; fixed chat template; fastsafetensors; prefix cache | `mods/fix-qwen3.6-chat-template` | true |
| [`qwen3.6-35b-a3b-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8.yaml) | `Qwen/Qwen3.6-35B-A3B-FP8` | 0.8 | 262144 | 16384 | 2 | `fp8` | `qwen3_xml` | `qwen3` | `flashinfer` | fixed chat template; fastsafetensors; prefix cache; Ray | `mods/fix-qwen3.6-chat-template` | absent |
| [`qwen3.6-35b-a3b-nvfp4-no-mtp.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-nvfp4-no-mtp.yaml) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | 0.4 | 262144 | 8192 | 2 | `fp8` | `qwen3_xml` | `qwen3` | `flashinfer` | max seqs 8; Marlin MoE; chunked prefill; async scheduling; prefix cache; no MTP; Ray | absent | absent |
| [`qwen3.6-35b-a3b-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-nvfp4.yaml) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | 0.4 | 262144 | 8192 | 2 | `fp8` | `qwen3_xml` | `qwen3` | `flashinfer` | max seqs 8; Marlin MoE; chunked prefill; async scheduling; MTP 3; prefix cache; Ray | absent | absent |
| [`step-3.7-flash-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/step-3.7-flash-fp8.yaml) | `stepfun-ai/Step-3.7-Flash-FP8` | 108 GiB | 262144 | absent | 2 | `fp8` | `step3p5` | `step3p5` | absent | patched GPU-GiB flag; max seqs 16; safetensors; Ray | `mods/step-3.7-flash`, `mods/gpu-mem-util-gb` | absent |
| [`step-3.7-flash-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/step-3.7-flash-nvfp4.yaml) | `stepfun-ai/Step-3.7-Flash-NVFP4` | 0.8 | 262144 | absent | 2 | absent | `step3p5` | `step3p5` | absent | trust remote code; InstantTensor; Ray | `mods/step-3.7-flash` | absent |

## 3. Recipe ↔ installed model coverage

Installed means a model directory containing nonempty weight files, not merely `config.json`.

**(a) Installed models that have a recipe**

| exact model id | matching recipe(s) | required match pattern |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | [`qwen3.6-35b-a3b-fp8-dflash.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8-dflash.yaml), [`qwen3.6-35b-a3b-fp8-solo.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8-solo.yaml), [`qwen3.6-35b-a3b-fp8.yaml`](/home/josh/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8.yaml) | Exact HF repository ID suffices. The profile serves that exact ID plus `Qwen--Qwen3.6-35B-A3B-FP8` and `vllm-active` in [`start_hf_qwen_qwen3.6-35b-a3b-fp8.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_qwen_qwen3.6-35b-a3b-fp8.sh). **Precedence hazard:** the same model ID selects three materially different recipes: DFlash, solo, and clustered FlashInfer. |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | [`nemotron-3-nano-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-nano-nvfp4.yaml) | Exact HF repository ID suffices for the underlying model. The profile’s served name is instead `nemotron-3-nano` in [`start_nemotron_nano.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_nemotron_nano.sh). |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | [`nemotron-3-super-nvfp4.yaml`](/home/josh/spark-vllm-docker/recipes/nemotron-3-super-nvfp4.yaml) | Exact HF repository ID suffices for the underlying model. Served aliases are `nvidia/nemotron-3-super`, `nemotron-3-super`, and `vllm-active` in [`start_nemotron_super.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_nemotron_super.sh). |
| `openai/gpt-oss-120b` | [`openai-gpt-oss-120b.yaml`](/home/josh/spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml) | Exact HF repository ID suffices; the profile also exposes `openai--gpt-oss-120b` and `vllm-active` in [`start_hf_openai_gpt-oss-120b.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_openai_gpt-oss-120b.sh). |

**(b) Installed models with no recipe**

- `Qwen/Qwen3-1.7B` — weighted snapshot under [`models--Qwen--Qwen3-1.7B`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B).
- `Qwen/Qwen3-8B` — weighted snapshot and profile [`start_hf_qwen_qwen3-8b.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_qwen_qwen3-8b.sh).
- `Qwen/Qwen3-14B` — weighted snapshot and profile [`start_hf_qwen_qwen3-14b.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_qwen_qwen3-14b.sh).
- `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4` — weighted snapshot and profile [`start_hf_lyf_qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-nvfp4.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_lyf_qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-nvfp4.sh).
- `Systran/faster-whisper-base` — weighted snapshot under [`models--Systran--faster-whisper-base`](/home/josh/.cache/huggingface/hub/models--Systran--faster-whisper-base).
- `Systran/faster-whisper-medium` — weighted snapshot under [`models--Systran--faster-whisper-medium`](/home/josh/.cache/huggingface/hub/models--Systran--faster-whisper-medium).
- `qwen2.5-14b-instruct-gptq-int8` — exact local directory ID; its [`README.md`](/mnt/models/qwen2.5-14b-instruct-gptq-int8/README.md) identifies the model as `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8`.
- `qwen3-coder-next-nvfp4` — exact local directory ID. Its [`README.md`](/mnt/models/qwen3-coder-next-nvfp4/README.md) identifies a local `Qwen3-Coder-Next-NVFP4-GB10` quantization based on `Qwen/Qwen3-Coder-Next`; it is not the FP8 or Intel INT4 artifact named by either recipe.
- `qwen3-next-80b-a3b-nvfp4` — exact local directory ID. Its [`README.md`](/mnt/models/qwen3-next-80b-a3b-nvfp4/README.md) identifies `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`.
- `qwen3-vl-4b-fp8` — exact local directory ID; its [`README.md`](/mnt/models/qwen3-vl-4b-fp8/README.md) identifies an FP8 quantization of `Qwen/Qwen3-VL-4B-Instruct`.
- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` — weighted snapshot under [`/opt/models/hub`](/opt/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B).
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` — weighted snapshot and profile [`start_hf_deepseek-ai_deepseek-r1-distill-qwen-14b.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_deepseek-ai_deepseek-r1-distill-qwen-14b.sh).
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` — weighted snapshot and profile [`start_hf_deepseek-ai_deepseek-r1-distill-qwen-32b.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_hf_deepseek-ai_deepseek-r1-distill-qwen-32b.sh).

**(c) Recipe model IDs that are not installed**

- `deepseek-ai/DeepSeek-V4-Flash` — `deepseek-v4-flash.yaml`
- `google/diffusiongemma-26B-A4B-it` — two BF16 recipes
- `nvidia/diffusiongemma-26B-A4B-it-NVFP4` — two NVFP4 recipes
- `nvidia/Gemma-4-26B-A4B-NVFP4`
- `google/gemma-4-26B-A4B-it`
- `cyankiwi/GLM-4.7-Flash-AWQ-4bit`
- `QuantTrio/MiniMax-M2-AWQ`
- `cyankiwi/MiniMax-M2.5-AWQ-4bit`
- `cyankiwi/MiniMax-M2.7-AWQ-4bit`
- `Qwen/Qwen3-Coder-Next-FP8`
- `Intel/Qwen3-Coder-Next-int4-AutoRound`
- `Qwen/Qwen3.5-122B-A10B-FP8`
- `Intel/Qwen3.5-122B-A10B-int4-AutoRound`
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Intel/Qwen3.5-397B-A17B-int4-AutoRound`
- `nvidia/Qwen3.6-35B-A3B-NVFP4` — two recipes
- `stepfun-ai/Step-3.7-Flash-FP8`
- `stepfun-ai/Step-3.7-Flash-NVFP4`

The following are metadata-only stubs and were not counted as installed: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-Coder-14B-Instruct` have `config.json` and tokenizer metadata but no weight files in their HF snapshots. `/opt/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B` is also a weightless duplicate stub; the weighted copy is under `/opt/models/hub/`.

## 4. Tool-parser capability map

“Absent” means no parser flag is supported by the on-disk evidence for that row. “Uncertain” marks mappings inferred from template protocol rather than an exact recipe.

| model id | architectures[] from config.json | model family | correct `--tool-call-parser` | correct `--reasoning-parser` | supports tool calling | evidence |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-1.7B` | `["Qwen3ForCausalLM"]` | Qwen3 dense | `hermes` (protocol match; uncertain) | `qwen3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json); `tokenizer_config.json` contains a tool-definition section, JSON `<tool_call>` format, and `<think>` handling. No recipe. |
| `Qwen/Qwen3-8B` | `["Qwen3ForCausalLM"]` | Qwen3 dense | `hermes` (protocol match; uncertain) | `qwen3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/config.json); its tokenizer chat template has the same JSON `<tool_call>` and `<think>` protocol. |
| `Qwen/Qwen3-14B` | `["Qwen3ForCausalLM"]` | Qwen3 dense | `hermes` (protocol match; uncertain) | `qwen3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json); tokenizer chat template injects tools and requests JSON inside `<tool_call>` tags. |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `["Qwen3_5MoeForConditionalGeneration"]` | Qwen3.6/Qwen3.5-MoE implementation | `qwen3_xml` | `qwen3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989/config.json); [`chat_template.jinja`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989/chat_template.jinja) uses `<function=...><parameter=...>` XML; all three exact-model recipes set `qwen3_xml`/`qwen3`. |
| `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4` | `["Qwen3_5MoeForConditionalGeneration"]` | Qwen3.6 derivative | `qwen3_xml` | `qwen3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--lyf--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4/snapshots/c9eb0a997ecfc688c468146404606ef6df49555f/config.json); its [`chat_template.jinja`](/home/josh/.cache/huggingface/hub/models--lyf--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4/snapshots/c9eb0a997ecfc688c468146404606ef6df49555f/chat_template.jinja) is byte-for-byte identical to the exact recipe-backed Qwen3.6 FP8 template. The current profile instead sets `qwen3_coder`. |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `["NemotronHForCausalLM"]` | Nemotron-H / Nemotron 3 Nano | `qwen3_coder` | `nano_v3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/snapshots/ce1b118ae66ec705d02c241525192832eb045fd3/config.json); chat template contains function/parameter tool XML; exact recipe sets both parsers and a plugin. |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `["NemotronHForCausalLM"]` | Nemotron-H / Nemotron 3 Super | `qwen3_coder` | `nemotron_v3` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6/config.json); exact recipe sets `qwen3_coder`/`nemotron_v3`. The profile conflicts by loading local [`super_v3_reasoning_parser.py`](/home/josh/super_v3_reasoning_parser.py) and selecting `super_v3`. |
| `openai/gpt-oss-120b` | `["GptOssForCausalLM"]` | GPT-OSS | `openai` | `openai_gptoss` | yes | [`config.json`](/home/josh/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json); [`chat_template.jinja`](/home/josh/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/chat_template.jinja) accepts tools and Harmony channels; exact recipe sets both parsers. |
| `qwen2.5-14b-instruct-gptq-int8` | `["Qwen2ForCausalLM"]` | Qwen2.5 Instruct | `hermes` (protocol match; uncertain) | absent | yes | [`config.json`](/mnt/models/qwen2.5-14b-instruct-gptq-int8/config.json); `tokenizer_config.json` injects tool definitions and requests JSON `<tool_call>` objects. No recipe/parser-bearing profile. |
| `qwen3-coder-next-nvfp4` | `["Qwen3NextForCausalLM"]` | Qwen3-Coder-Next | `qwen3_coder` | absent | yes | [`config.json`](/mnt/models/qwen3-coder-next-nvfp4/config.json); [`chat_template.jinja`](/mnt/models/qwen3-coder-next-nvfp4/chat_template.jinja) uses function/parameter XML; profile [`start_qwen3_coder_next.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_qwen3_coder_next.sh) sets `qwen3_coder`. |
| `qwen3-next-80b-a3b-nvfp4` | `["Qwen3NextForCausalLM"]` | Qwen3-Next Instruct | `hermes` (protocol match; uncertain) | absent | yes | [`config.json`](/mnt/models/qwen3-next-80b-a3b-nvfp4/config.json); [`chat_template.jinja`](/mnt/models/qwen3-next-80b-a3b-nvfp4/chat_template.jinja) injects tools and requests JSON `<tool_call>` objects. No parser appears in its profile. |
| `qwen3-vl-4b-fp8` | `["Qwen3VLForConditionalGeneration"]` | Qwen3-VL Instruct | `hermes` (protocol match; uncertain) | absent | yes | [`config.json`](/mnt/models/qwen3-vl-4b-fp8/config.json); [`chat_template.json`](/mnt/models/qwen3-vl-4b-fp8/chat_template.json) injects tools and requests JSON `<tool_call>` objects. |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | `["LlamaForCausalLM"]` | DeepSeek-R1 distill on Llama | absent; tool support uncertain | `deepseek_r1` (template match; uncertain) | none found — uncertain | [`config.json`](/opt/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e/config.json); tokenizer template contains DeepSeek `<think>` generation but no tool-definition injection. No recipe/profile parser evidence. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `["Qwen2ForCausalLM"]` | DeepSeek-R1 distill on Qwen2 | absent; tool support uncertain | `deepseek_r1` (template match; uncertain) | none found — uncertain | [`config.json`](/opt/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761/config.json); tokenizer template starts assistant generation with `<think>` but does not inject supplied tools. The generated profile’s `qwen3_coder` is supported only by the name substring. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `["Qwen2ForCausalLM"]` | DeepSeek-R1 distill on Qwen2 | absent; tool support uncertain | `deepseek_r1` (template match; uncertain) | none found — uncertain | [`config.json`](/opt/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746/config.json); same DeepSeek template evidence; generated profile sets `qwen3_coder` solely because the name contains Qwen. |
| `Systran/faster-whisper-base` | absent | CTranslate2 Whisper | absent | absent | no | [`config.json`](/home/josh/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66/config.json) has no `architectures` field and the snapshot has no chat template. |
| `Systran/faster-whisper-medium` | absent | CTranslate2 Whisper | absent | absent | no | [`config.json`](/home/josh/.cache/huggingface/hub/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f/config.json) has no `architectures` field and the snapshot has no chat template. |

The installed models that the substring test would wrongly give `qwen3_coder` are:

| model id | parser indicated by on-disk evidence instead |
|---|---|
| `Qwen/Qwen3-1.7B` | `hermes`; reasoning `qwen3` — inferred from its JSON `<tool_call>` template, not an exact recipe |
| `Qwen/Qwen3-8B` | `hermes`; reasoning `qwen3` — inferred from template |
| `Qwen/Qwen3-14B` | `hermes`; reasoning `qwen3` — inferred from template |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `qwen3_xml`; reasoning `qwen3` — exact recipe evidence |
| `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4` | `qwen3_xml`; reasoning `qwen3` — byte-identical template to the recipe-backed Qwen3.6 model |
| `qwen2.5-14b-instruct-gptq-int8` | `hermes`; no reasoning parser — inferred from template |
| `qwen3-next-80b-a3b-nvfp4` | `hermes`; no reasoning parser — inferred from template |
| `qwen3-vl-4b-fp8` | `hermes`; no reasoning parser — inferred from template |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | no supported tool parser found; reasoning protocol matches `deepseek_r1` |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | no supported tool parser found; reasoning protocol matches `deepseek_r1` |

`qwen3-coder-next-nvfp4` is the installed Qwen-named case for which `qwen3_coder` is supported by its template and existing profile.

The host filesystem contains vLLM `0.21.0`, established by [`vllm-0.21.0.dist-info/METADATA`](/home/josh/.local/lib/python3.12/site-packages/vllm-0.21.0.dist-info/METADATA). Its on-disk registries accept:

- Tool parsers from [`vllm/tool_parsers/__init__.py`](/home/josh/.local/lib/python3.12/site-packages/vllm/tool_parsers/__init__.py): `deepseek_v3`, `deepseek_v31`, `deepseek_v32`, `deepseek_v4`, `cohere_command3`, `cohere_command4`, `ernie45`, `glm45`, `glm47`, `granite-20b-fc`, `granite`, `granite4`, `hermes`, `poolside_v1`, `hunyuan_a13b`, `hy_v3`, `internlm`, `jamba`, `lfm2`, `kimi_k2`, `llama3_json`, `llama4_json`, `llama4_pythonic`, `longcat`, `mimo`, `minimax_m2`, `minimax`, `mistral`, `olmo3`, `openai`, `phi4_mini_json`, `pythonic`, `qwen3_coder`, `qwen3_xml`, `seed_oss`, `step3`, `step3p5`, `xlam`, `gigachat3`, `functiongemma`, `gemma4`.
- Reasoning parsers from [`vllm/reasoning/__init__.py`](/home/josh/.local/lib/python3.12/site-packages/vllm/reasoning/__init__.py): `deepseek_r1`, `deepseek_v3`, `deepseek_v4`, `poolside_v1`, `cohere_command3`, `cohere_command4`, `ernie45`, `gemma4`, `glm45`, `openai_gptoss`, `granite`, `holo2`, `hunyuan_a13b`, `hy_v3`, `kimi_k2`, `mimo`, `minimax_m2`, `minimax_m2_append_think`, `mistral`, `nemotron_v3`, `olmo3`, `qwen3`, `seed_oss`, `step3`, `step3p5`.

`nano_v3` and `super_v3` are plugin-provided values, not built into that registry. [`super_v3_reasoning_parser.py`](/home/josh/super_v3_reasoning_parser.py) registers `super_v3`; the Nano mod’s [`run.sh`](/home/josh/spark-vllm-docker/mods/nemotron-nano/run.sh) downloads `nano_v3_reasoning_parser.py`, but that downloaded file is not present on disk.

The profiles run several different container builds—vLLM `0.20.0`, an image documented in a profile as `0.23.1`, and `cu130-nightly`. Their parser registries are not present on the host filesystem, so their complete accepted-value sets could not be determined from disk.

## 5. kv-cache-dtype ground truth

Weight quantization such as FP8 or NVFP4 is not itself evidence of FP8 KV cache. The table distinguishes model configuration, recipe, and generated/profile evidence.

| installed model | on-disk KV evidence | status of blanket `--kv-cache-dtype fp8` |
|---|---|---|
| `Qwen/Qwen3-1.7B` | No `quantization_config` or `kv_cache_scheme` in [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json); no recipe/profile. | unproven |
| `Qwen/Qwen3-8B` | No KV declaration in [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/config.json); current profile omits the flag. | unproven |
| `Qwen/Qwen3-14B` | No KV declaration in [`config.json`](/home/josh/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json); current profile omits the flag. | unproven |
| `Qwen/Qwen3.6-35B-A3B-FP8` | Weight `quantization_config.quant_method` is `fp8`, but no config KV scheme. The solo and standard recipes explicitly set `--kv-cache-dtype fp8`; the DFlash recipe does not. | recipe-proven for two of three recipes; absent in the DFlash recipe |
| `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4` | `quantization_config.kv_cache_scheme` is explicitly null in [`config.json`](/home/josh/.cache/huggingface/hub/models--lyf--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4/snapshots/c9eb0a997ecfc688c468146404606ef6df49555f/config.json). Its generated profile sets FP8 KV; no exact recipe does. | profile-only; unproven by model config or recipe |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | Config has no KV scheme; exact recipe and [`start_nemotron_nano.sh`](/home/josh/DGX-Model-Manager/profiles/vLLM/start_nemotron_nano.sh) explicitly set FP8 KV. | recipe/profile-proven |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | [`config.json`](/home/josh/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6) has `quantization_config.kv_cache_scheme = {dynamic:false, num_bits:8, type:"float"}`; recipe and profile set FP8 KV. | declared and recipe/profile-proven |
| `openai/gpt-oss-120b` | Config declares MXFP4 weights but no KV scheme. The recipe sets FP8 KV, while the hand-corrected current profile explicitly documents and uses full-precision KV with no FP8 flag. | conflicting disk evidence; blanket FP8 is wrong for the current known-good profile |
| `qwen2.5-14b-instruct-gptq-int8` | [`config.json`](/mnt/models/qwen2.5-14b-instruct-gptq-int8/config.json) declares GPTQ weights and no KV scheme; generated profile sets FP8 KV. | profile-only; unproven |
| `qwen3-coder-next-nvfp4` | [`config.json`](/mnt/models/qwen3-coder-next-nvfp4/config.json) explicitly has `quantization_config.kv_cache_scheme: null`; existing profile sets FP8 KV. | profile-only; unproven by config or exact recipe |
| `qwen3-next-80b-a3b-nvfp4` | [`config.json`](/mnt/models/qwen3-next-80b-a3b-nvfp4/config.json) declares an 8-bit floating `quantization_config.kv_cache_scheme`; profile also sets FP8 KV. | declared and profile-proven |
| `qwen3-vl-4b-fp8` | [`config.json`](/mnt/models/qwen3-vl-4b-fp8/config.json) declares FP8 weight quantization but no KV scheme; generated profile sets FP8 KV. | profile-only; FP8 weights do not prove FP8 KV |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | No quantization or KV scheme in [`config.json`](/opt/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e/config.json); no recipe/profile. | unproven |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | No quantization or KV scheme in [`config.json`](/opt/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761/config.json); generated profile sets FP8 KV. | profile-only; unproven |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | No quantization or KV scheme in [`config.json`](/opt/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746/config.json); generated profile sets FP8 KV. | profile-only; unproven |
| `Systran/faster-whisper-base` | No KV declaration or recipe; this is a CTranslate2 speech model. | unproven and not a vLLM chat-model setting |
| `Systran/faster-whisper-medium` | No KV declaration or recipe; this is a CTranslate2 speech model. | unproven and not a vLLM chat-model setting |

The installed models with model-level evidence for 8-bit floating KV are `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` and `qwen3-next-80b-a3b-nvfp4`. Recipe command blocks additionally establish FP8 KV for Qwen3.6 FP8 standard/solo, Nemotron Nano, Nemotron Super, and GPT-OSS, although GPT-OSS conflicts with its newer hand-corrected profile.

## 6. Uncertainties

- The exact originating HF repository IDs for the locally named `qwen3-coder-next-nvfp4` and `qwen3-vl-4b-fp8` directories are not recorded in `config.json`; their README files identify base models only. Preserved HF snapshot metadata or an explicit source-repository field would settle this.
- `hermes` for the installed Qwen3 dense, Qwen2.5, Qwen3-Next, and Qwen3-VL models is inferred from their JSON `<tool_call>` chat-template protocol and the local vLLM parser implementation; no exact recipe confirms it. An exact-machine known-good recipe or parser integration test would settle each row.
- Tool-calling support for the three DeepSeek-R1 distill models is uncertain. Their templates expose reasoning and can render existing tool-result roles, but do not inject supplied tool definitions. A model-specific recipe or successful tool-call integration test would settle it.
- The complete accepted parser sets inside the `v0.20.0`, profile-described `v0.23.1`, and `cu130-nightly` container images cannot be read from the host source. The parser registry files or `vllm serve --help` output from each exact image would settle this.
- `nano_v3_reasoning_parser.py` is downloaded at mod-application time and is not present on disk. The downloaded plugin file would settle its exact behavior.
- GPT-OSS has conflicting KV evidence: its recipe sets FP8 KV, while its newer hand-corrected profile explicitly uses full-precision KV. A synchronized current recipe or retained validation record for the recipe command would settle which artifact is authoritative for present operation.
- FP8 KV correctness for models supported only by auto-generated/profile flags—rather than a recipe or `kv_cache_scheme`—cannot be established from the files read. A known-good measured recipe, model-declared KV scheme, or retained runtime validation output would settle each case.