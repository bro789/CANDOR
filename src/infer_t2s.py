"""Text-to-SQL inference with vLLM over several BIRD-format roots in one model load (OmniSQL prompt format).
python infer_t2s.py --model PATH --tag omni7b --roots R1 R2 ... [--n 1 --temperature 0]
writes R/pred_{tag}.json : [{'pred_sqls': [...], 'responses': [...]}] aligned with R/dev.json
"""
import argparse, json, re, os, time


def parse_response(response):
    blocks = re.findall(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--gpu_util', type=float, default=0.90)
    ap.add_argument('--max_model_len', type=int, default=8192)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, dtype='float16' if 'AWQ' in a.model else 'bfloat16', tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=a.temperature, max_tokens=2048, n=a.n, stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None)
    for r in a.roots:
        out_f = f'{r}/pred_{a.tag}.json'
        if os.path.exists(out_f):
            print('skip', out_f)
            continue
        data = json.load(open(f'{r}/dev_bird.json'))
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': d['input_seq']}], add_generation_prompt=True, tokenize=False)
                   for d in data]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        res = []
        for o in outs:
            resp = [x.text for x in o.outputs]
            res.append({'pred_sqls': [parse_response(x) for x in resp], 'responses': resp})
        json.dump(res, open(out_f, 'w'), ensure_ascii=False)
        print('wrote', out_f, round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
