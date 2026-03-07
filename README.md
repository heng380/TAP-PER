# Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning


This is source code of our EMNLP 2024 paper

[**Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning**](https://arxiv.org/abs/2402.04401).


## Overview ##

* **Ownership**: Existing methods are processed centralized, where user history is encoded in a personalized prompt and processed by centralized LLMs. This paradigm limits the model's customization and ability to provide deep, personalized experiences tailored to individual users. Moreover, when using a centralized model, users often have to share personal data with the service provider, which raises concerns about how user data are stored, used, and protected.

* **Behavior Pattern Generalization**: As is revealed by existing research, LLMs can be easily distracted by irrelevant context information that retrieval can hardly avoid. In LLM personalization, where the retrieval corpus is confined to a specific user's behaviors, retrieval augmentation might underperform, especially when the user's past behaviors do not closely mirror the patterns needed for the query at hand.

<div  align="center">    
<img src="./asset/teaser.png" width="50%" height="100%">
</div>

Personalization in large language models (LLMs) is increasingly important, aiming to align the LLMs' interactions, content, and recommendations with individual user preferences. Recent advances have highlighted effective prompt design by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these methods faced limitations due to a lack of model ownership, resulting in constrained customization and privacy issues, and often failed to capture complex, dynamic user behavior patterns. To address these shortcomings, we introduce One PEFT Per User (OPPU), employing personalized parameter-efficient fine-tuning (PEFT) modules to store user-specific behavior patterns and preferences. By plugging in personal PEFT parameters, users can own and use their LLMs individually. OPPU integrates parametric user knowledge in the personal PEFT parameters with non-parametric knowledge from retrieval and profiles, adapting LLMs to user behavior shifts. Experimental results demonstrate that OPPU significantly outperforms existing prompt-based methods across seven diverse tasks in the LaMP benchmark. Further studies reveal OPPU's enhanced capabilities in handling user behavior shifts, modeling users at different activity levels, maintaining robustness across various user history formats, and displaying versatility with different PEFT methods.

<div  align="center">    
<img src="./asset/overview.png" width="70%" height="100%">
</div>

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

#### Stage 2: Group memory

Training:

```bash
python /home/ubuntu/repos/agent/OPPU/cluster_profiles.py --task_name movie_tagging

torchrun --nproc_per_node=8 /home/ubuntu/repos/agent/OPPU/task_LoRA_group.py --task_name movie_tagging
```

Evaluation:

```bash
python /home/ubuntu/repos/agent/OPPU/eval_group.py --task_name movie_tagging --k 1 --profile
```

#### Stage 3: Local memory + mediator

Training:

```bash
torchrun --nproc_per_node=8 /home/ubuntu/repos/agent/OPPU/task_LoRA_local_memory.py --task_name movie_tagging --group_mode 1
```

Evaluation:

```bash
python /home/ubuntu/repos/agent/OPPU/eval_local.py --task_name movie_tagging --group_mode 1
```

## Citation ##
If you find this paper or codebase useful in your research, please kindly cite the following paper.

```bibtex
@article{tan2024democratizing,
  title={Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning},
  author={Tan, Zhaoxuan and Zeng, Qingkai and Tian, Yijun and Liu, Zheyuan and Yin, Bing and Jiang, Meng},
  journal={arXiv preprint arXiv:2402.04401},
  year={2024}
}
```
