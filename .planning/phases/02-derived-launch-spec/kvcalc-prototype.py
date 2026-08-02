import json, glob, os

POOL = 121.0  # GB unified

def load(p):
    c = json.load(open(p)); return c, (c.get("text_config") or c)

def attn_layers(t):
    """(full_attn_layers, sliding_layers, sliding_window) — hybrid-aware."""
    n = t.get("num_hidden_layers") or 0
    lt = t.get("layer_types")
    if isinstance(lt, list) and lt:
        full = sum(1 for x in lt if x == "full_attention")
        slide = sum(1 for x in lt if "sliding" in str(x))
        return full, slide, t.get("sliding_window")
    pat = t.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat:
        return pat.count("*"), 0, None
    iv = t.get("full_attention_interval")
    if iv:
        return n // iv, 0, None
    sw = t.get("sliding_window")
    if sw and t.get("use_sliding_window"):
        return 0, n, sw
    return n, 0, None

def kv_bytes_per_token(t, kv_dtype_bytes):
    full, slide, sw = attn_layers(t)
    kvh = t.get("num_key_value_heads") or t.get("num_attention_heads") or 0
    hd = t.get("head_dim") or ((t.get("hidden_size") or 0) // (t.get("num_attention_heads") or 1))
    per_layer = 2 * kvh * hd * kv_dtype_bytes
    return full * per_layer, slide * per_layer, sw, full, slide

rows = []
paths = sorted(set(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*/snapshots/*/config.json"))
      + glob.glob("/mnt/models/*/config.json") + glob.glob("/opt/models/models--*/snapshots/*/config.json")))
for p in paths:
    try: c, t = load(p)
    except Exception: continue
    if not t.get("num_hidden_layers"): continue
    name = p.split("models--")[-1].split("/snapshots")[0].replace("--","/") if "models--" in p else p.split("/")[-2]
    d = os.path.dirname(p)
    size = sum(os.path.getsize(f) for f in glob.glob(d+"/*.safetensors")) / 1e9
    if size == 0:
        root = p.split("/snapshots")[0]
        size = sum(os.path.getsize(f) for f in glob.glob(root+"/blobs/*") if os.path.isfile(f)) / 1e9
    for kvb, lbl in ((1,"fp8"),(2,"bf16")):
        fullb, slideb, sw, nf, ns = kv_bytes_per_token(t, kvb)
        maxpos = t.get("max_position_embeddings") or 0
        naive_layers = t.get("num_hidden_layers")
        naive = 2*(t.get("num_key_value_heads") or 0)*(t.get("head_dim") or ((t.get("hidden_size") or 0)//(t.get("num_attention_heads") or 1)))*kvb*naive_layers
        if lbl != "fp8": continue
        # KV at declared max ctx (fp8), hybrid-aware vs naive
        kv_max = (fullb*maxpos + slideb*(sw or 0)) / 1e9
        kv_naive = naive*maxpos/1e9
        rows.append((name[:46], f"{nf}+{ns}s/{naive_layers}", maxpos, round(size,1),
                     round(kv_max,1), round(kv_naive,1)))

print(f"{'model':46} {'attn/total':13} {'maxctx':>7} {'wt GB':>6} {'KVfp8@max':>9} {'naive':>7}")
for r in sorted(rows, key=lambda x:-x[3]):
    print(f"{r[0]:46} {r[1]:13} {r[2]:>7} {r[3]:>6} {r[4]:>9} {r[5]:>7}")

print("\n=== derived recommendations (pool 121 GiB, 8 GiB OS margin) ===")
print(f"{'model':44} {'ctx':>7} {'wt':>5} {'KV':>5} {'ovh':>4} {'need':>5} {'util':>5}")
for p in paths:
    try: c,t = load(p)
    except Exception: continue
    if not t.get("num_hidden_layers"): continue
    d=os.path.dirname(p); root=p.split("/snapshots")[0]
    size=sum(os.path.getsize(f) for f in glob.glob(d+"/*.safetensors"))/1e9
    if size==0: size=sum(os.path.getsize(f) for f in glob.glob(root+"/blobs/*") if os.path.isfile(f))/1e9
    if size < 3: continue
    name=p.split("models--")[-1].split("/snapshots")[0].replace("--","/") if "models--" in p else p.split("/")[-2]
    ctx=t.get("max_position_embeddings") or 0
    fullb,slideb,sw,nf,ns=kv_bytes_per_token(t,1)   # fp8 KV
    kv=(fullb*ctx + slideb*(sw or 0))/1e9
    ovh=6.0   # CUDA ctx + graphs + activations
    need=size+kv+ovh
    util=min(0.95,round((need/POOL)+0.04,2))
    print(f"{name[:44]:44} {ctx:>7} {size:>5.1f} {kv:>5.1f} {ovh:>4.0f} {need:>5.1f} {util:>5.2f}")
