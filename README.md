# LMPL 2025: A Case Study on the Effectiveness of LLMs in Verification with Proof Assistants

This repository contains the source code and the results for the experiment conducted in the paper.

## Results

To inspect the results directly, see the `results` folder. Each file contains the ablation, the theorem / lemma name, the file it is contained in, the result, and the error. Under `results/<project logs>/stats` directory, you can find the specific tactic used by each model for each ablation, and in the file `original_proof_stats.csv`, you can find the tactic statistics of the original proofs.

Inside `logs.zip`, you can find the full logs of the experiment, including the generated objects and the LLM responses. Note that log folder names may contain the temperature, but that is ignored if the model does not support the parameter.

## Setup

The experiment is run inside a Docker container, which only supports x86 machines as previous Coq versions do not support ARM.

### Setting Up Environment Variables

Fill in the two environment variables, as found in `.env.example`, into the `.env` file. `OPENAI_BASE_URL` should be an OpenAI-compatible endpoint, such as [OpenRouter](https://openrouter.ai/) which we have used for our experiment. `OPENAI_API_KEY` should be your API key for the endpoint.
```
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...
```

> [!IMPORTANT]  
> If you're not using OpenRouter and encounter an error when replicating, you may need to change the model names in `main.py` model list to match those supported by your endpoint.

### Getting Started With Docker

Pull the Docker image from GHCR. 

```bash
docker pull ghcr.io/bbayazit16/lmpl-2025-artifact:latest
```

Then, run the container with:
```bash
docker run --name lmpl-2025-artifact -it --env-file .env ghcr.io/bbayazit16/lmpl-2025-artifact:latest
```

If you don't want to pull the image, you can also build it locally using the provided `Dockerfile` by running `docker build -t ghcr.io/bbayazit16/lmpl-2025-artifact:latest .` in the root directory of this repository.

### Getting Started Without Docker

Setup without Docker is possible. You must have `opam`, `haskell stack`, `pkg-config`, and `python 3.12.3` installed. To do so, run:
```bash
apt-get update && apt-get install -y opam haskell-stack pkg-config python3.12 python3-pip
```

Then, follow the steps below to install and build the projects:

```bash
chmod +x coq-setup.sh install-hs-to-coq.sh install-verdi.sh
./coq-setup.sh
./install-hs-to-coq.sh
./install-verdi.sh
```

Then, activate the virtual environment and install Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Replication Instructions

Make sure you are inside the docker container. See the setup steps above if you are not.

For the `hs-to-coq` experiment, first switch to the correct opam switch:
```bash
opam switch coq-8.10.2
eval $(opam env)
```

For `Verdi`, use:
```bash
opam switch coq-8.11.0
eval $(opam env)
```

To run the experiment, first construct the objects as follows:
```bash
python3 main.py <project> --logs-dir <logs_dir> --dry-run
```
where `<project>` is either:
1) `hs-to-coq/base-thy` for `hs-to-coq`, or
2) `verdi` for `Verdi`.

and `<logs_dir>` is either:
1) `logs-hs-to-coq` for `hs-to-coq`, or
2) `logs-verdi` for `Verdi`.

After the objects are constructed, run the following command to evaluate for a given model:
```bash
python3 main.py <project> --logs-dir <logs_dir> --models <model_name>
```

Supported models are (space-separated to run for multiple models):
```
gpt-4o-mini
gpt-4o
o4-mini-medium
deepseek/deepseek-prover-v2
deepseek/deepseek-r1-0528:free
```

and more are listed in `main.py`. By default, omitting `--models` runs the experiment for all models.

Note that LLM results are cached under the `<logs-dir>` directory. Feel free to interrupt the experiment and resume later without having to redo the LLM API calls again. You may also use the logs directory to inspect the generated proofs.

### Ablations

You can adjust context with the following flags passed to `main.py`:
1. Informed mode (default): no flags
2. No external dependencies: `--no-dependencies`
3. No in-file context: `--no-lines`
4. No context: both `--no-dependencies --no-lines`

For example, if you want to run `hs-to-coq` experiment for gpt-4o-mini and gpt-4o with no external dependencies but with in-file context, run:
```bash
python3 main.py hs-to-coq/base-thy --logs-dir logs-hs-to-coq --models gpt-4o-mini gpt-4o --no-dependencies
```

### Additional Help
See `main.py --help` for more details, including how to configure thread counts (defaulting to your CPU count), temperature, maximum tokens, counting tactic counts, and more.

### Aggregating Results

Run `python3 sheets_util.py` to automatically aggregate all `results.csv` files under logs into a single folder called `results`.

### Modifying the Experiment

To add/remove models, you can simply extend the list of models in `main.py` using the classes we have provided.

To modify the Rocq prover checking and evaluation logic (including a possible refinement / feedback  mechanism), you must modify `eval.py`. The main entry point is the `eval_coq_objects` function inside `eval.py`.

To modify dependency extraction logic, see `coq_dependencies.py`.

# Citation

Barış Bayazıt, Yao Li, and Xujie Si. 2025. A Case Study on the Effectiveness of LLMs in Verification with Proof Assistants. In Proceedings of the 1st ACM SIGPLAN International Workshop on Language Models and Programming Languages (LMPL ’25), October 12–18, 2025, Singapore, Singapore. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3759425.3763391

# Acknowledgements

We thank all the anonymous reviewers of LMPL 2025 for their thoughtful and constructive comments on this paper and their suggestions for potential future directions for this work. We thank Yiming Lin for his feedback on a draft of this paper.

This work was partially supported by the University of Toronto Department of Computer Science Research Award. This work has also partially benefited from the Microsoft Accelerate Foundation Models Research (AFMR) grant program.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```Bash
opam switch coq-8.10.2
eval $(opam env)
python3 main.py hs-to-coq/base-thy --logs-dir logs-hs-to-coq --dry-run
python3 main.py hs-to-coq/base-thy --logs-dir logs-hs-to-coq --models gpt-4o-mini
```
