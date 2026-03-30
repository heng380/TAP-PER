# TAP-PER

## Dataset ##

We use publicly available data from the [LaMP](https://arxiv.org/abs/2304.11406) benchmark. You can download the our processed data [here](https://drive.google.com/file/d/1bJ3Rh_sqrw3suwwweFbra5CTV7GVjgxF/view?usp=sharing), unzip it, and place it under the ```./data``` folder


## Installation ##
Please install the dependencies via conda, using the following command:

```bash
pip install -r requirements.txt
```

## Experiment ##
```task_name``` can be selected from ```[citation, movie_tagging, news_categorize, news_headline, product_rating, scholarly_title, tweet_paraphrase]```.

### Three-Stage Training and Evaluation

#### Stage 1: Global memory

Training:

```bash
torchrun --nproc_per_node=8 task_LoRA.py --task_name movie_tagging
```

Evaluation:

```bash
python eval.py --task_name movie_tagging --ckpt_path ./ckpt/movie_tagging/k0-movie_tagging-llama3.1-8B-task_LoRA_ckpt/ --k 1 --profile
```

#### Stage 2: Rag prefix + mediator

Training:

```bash
torchrun --nproc_per_node=8 task_LoRA_ragpag.py --task_name tweet_paraphrase --k 10 --disable_pag --use_time_bias --use_order_bias
```

Evaluation:

```bash
python eval_ragpag.py --task_name tweet_paraphrase --k 10 --use_time_bias --use_order_bias --disable_pag
```

#### Stage 3: Rag prefix + pag prefix + mediator

Training:

```bash
torchrun --nproc_per_node=8 task_LoRA_ragpag.py --task_name movie_tagging --k 10 --use_time_bias --use_order_bias
```

Evaluation:

```bash
python eval_ragpag.py --task_name movie_tagging --k 10 --use_time_bias --use_order_bias
```