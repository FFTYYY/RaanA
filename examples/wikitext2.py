import torch as tc
from argparse import ArgumentParser
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import math
import os 

from raana import quantize, zeroshot_calibration
from raana.tricks import trick_centralize, trick_norm_row

try:
    from .change_hfhome import change_hfhome # type: ignore
    change_hfhome()
except:
    try:
        from change_hfhome import change_hfhome # type: ignore
        change_hfhome()
    except:
        print ("cache path: default")

def get_config():
    parser = ArgumentParser()
    parser.add_argument("--model" , type = str, default = "meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lowbit", action = "store_true", default = False)
    parser.add_argument("--avgbits", type = float, default = 3.3)
    return parser.parse_args()

def get_llama(model_name: str, device: str = "auto"):
    def skip(*args, **kwargs):
        pass
    tc.nn.init.kaiming_uniform_  = skip
    tc.nn.init.uniform_          = skip
    tc.nn.init.normal_           = skip

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast            = False , 
    )

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path   = model_name, 
        torch_dtype                     = tc.float16 , 
        low_cpu_mem_usage               = True,
            cache_dir                   = os.environ.get("HF_HOME", None),
        device_map                      = device ,
    )
    
    model.seqlen = 2048
    model = model.eval()

    return model, tokenizer

def get_wikitext2(tokenizer):

    testdata  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testenc  = tokenizer("\n\n".join(testdata ["text"]), return_tensors="pt")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    return testdata, testenc.input_ids, traindata, trainenc.input_ids

def evaluate(model, test_data):
    seqlen  = model.seqlen or 2048
    n_data = test_data.size(1) // seqlen

    losses = []
    sizes  = []
    ppl = -1
    pbar = tqdm(range(n_data), desc = "Evaluating")
    for eval_idx in pbar:

        input = test_data[:, eval_idx * seqlen : (eval_idx + 1) * seqlen].cuda()

        # loss = model(input, labels = input).loss
        output  = model(input)
        lm_logits    = output.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input[:, 1:]
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses.append(float(loss))
        sizes .append(int(input.view(-1).size(0)))

        total_loss = sum([loss * size for loss, size in zip(losses, sizes)])
        total_size = sum(sizes)
        avg_loss   = total_loss / total_size
        ppl = math.exp(float(avg_loss))

        desc = f"[{eval_idx: <3}] now ppl is {ppl:.4f}"
        pbar.set_description(desc)

        tc.cuda.empty_cache()
    
    print ("Evaluation done.")
    return ppl


if __name__ == "__main__":
    config = get_config()

    model , tokenizer = get_llama(config.model)
    testdata, test_ids, traindata, train_ids = get_wikitext2(tokenizer)
    
    model = model.eval()
    quantize_result = quantize(
        model,
        b_candidates    = [1,2,3,4,5,6,] + ([0.5, 0.75] if config.lowbit else []),
        calibrate_data  = zeroshot_calibration(tokenizer),
        avg_bits        = config.avgbits,
    )
    q_model         = quantize_result["model"]
    bits_allocation = quantize_result["bits"]

    print ("\n\n" + "-" * 50)
    print ("bits allocation:")
    for layer_name, bit in bits_allocation.items():
        print (f"{layer_name}: {bit}")
    print ("\n\n" + "-" * 50)

    loss_q = evaluate(q_model, test_ids)

    print (f"quantized ppl: {loss_q:.4f}")
