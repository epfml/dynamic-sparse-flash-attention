# Dynamic Sparse FlashAttention

Code to reproduce results for the paper "Faster Causal Attention Over Large Sequences Through Sparse Flash Attention"

Arxiv link.

# Setup

To install the required python dependencies, first run:

```bash
pip install -r ./requirements.txt
```

Then install Triton:
```bash
git clone https://github.com/openai/triton.git
cd triton 
git checkout b2a757d00028fe844a93904036a18e8670bfe92f
cd python 
pip install cmake
pip install -e .
```
In the command above we set the Triton library to the commit used in our experiments. Feel free to experiment with later Triton versions. 

# Reproducing our LM experiments on OpenWebText2

**GPU requirements:** Prefeerably, you need at least one A100. Some of our experiments use data-parallelism with up to 3 A100s. You should have no problem running those experiments on any GPU supporting `bfloat16`, you might have to change the model parameters to adapt to the memory available. 

Go in the `openwebtext2-experiments` folder and run the `script/train-LMs.sh` command.

# Reproducing our runtime results

**GPU requirements:** We used one A100. 

For the Hash-sparse and QK-sparse results, go in the `runtime-experiments` folder and check the `timeperf-hash-and-qk-sparse.ipynb` notebook.

# Reproducing our Reformer results

Coming soon

