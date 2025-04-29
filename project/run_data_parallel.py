import sys
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(cousin_dir))

from functools import partial
import time
import os
import argparse
import tqdm
import json
import datasets
import numpy as np
from transformers import AutoConfig, GPT2LMHeadModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.distributed as dist
from torch.multiprocessing import Process

from data_parallel.dataset import partition_dataset
from utils import get_tokenizer, evaluate_bleu, save_grad_weights, collate_batch, evaluate_loss, generate, train

PYTEST = False

import wandb

# ASSIGNMENT 4.1
def average_gradients(model):
    '''Aggregate the gradients from different GPUs
    
    1. Iterate through the parameters of the model 
    2. Use `torch.distributed` package and call the reduce fucntion to aggregate the gradients of all the parameters
    3. Average the gradients over the world_size (total number of devices)
    '''
    world_size = torch.distributed.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= world_size
    # # END SOLUTION

# ASSIGNMENT 4.1
def setup(rank, world_size, backend):
    '''Setup Process Group

    1. Set the environment variables `MASTER_ADDR` as `localhost` or `127.0.0.1`  and `MASTER_PORT` as `11868`
    2. Use `torch.distributed` to init the process group
    '''
    # BEGIN SOLUTION
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11868'
    torch.distributed.init_process_group(rank=rank, world_size=world_size, backend=backend)
    # END SOLUTION

def is_conversational(example) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False

def apply_chat_template(
    example, #: dict[str, list[dict[str, str]]],
    tokenizer, #: PreTrainedTokenizerBase,
    tools, #: Optional[list[Union[dict, Callable]]] = None,
): # -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        
    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output


def maybe_apply_chat_template(
    example, #: dict[str, list[dict[str, str]]],
    tokenizer, #: PreTrainedTokenizerBase,
    tools, #: Optional[list[Union[dict, Callable]]] = None,
): # -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]` or `None`, *optional*, defaults to `None`):
            A list of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    
    new_example = dict()
    prompt_text = example["question"]["full_text"]
    new_example["prompt"] = [{"role": "user", "content": prompt_text}]
    if example["score_0"] >= example["score_1"]:
        chosen_text = example["answer_0"]
        rejected_text = example["answer_1"]
    else:
        chosen_text = example["answer_1"]
        rejected_text = example["answer_0"]
    new_example["chosen"] = [{"role": "assistant", "content": chosen_text}]
    new_example["rejected"] = [{"role": "assistant", "content": rejected_text}]
    
    if is_conversational(new_example):
        return apply_chat_template(new_example, tokenizer, tools)
    else:
        return new_example

def extract_prompt(example):#e: dict[str, Sequence]) -> dict[str, Sequence]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    For more details, see [`maybe_extract_prompt`].
    """
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if example["chosen"][idx - 1] == " ":  # remove space before the prompt
                idx -= 1
            break
    return {
        "prompt": example["chosen"][:idx],
        "chosen": example["chosen"][idx:],
        "rejected": example["rejected"][idx:],
    }

def maybe_extract_prompt(example): #: dict[str, list]) -> dict[str, list]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. Else, the function
    identifies the longest common sequence (prefix) of conversation turns between the "chosen" and "rejected"
    completions and extracts this as the prompt. It then removes this prompt from the respective "chosen" and
    "rejected" completions.

    Args:
        example (`dict[str, list]`):
            A dictionary representing a single data entry in the preference dataset. It must contain the keys
            `"chosen"` and `"rejected"`, where each value is either conversational or standard (`str`).

    Returns:
        `dict[str, list]`: A dictionary containing:
            - `"prompt"`: The longest common prefix between the "chosen" and "rejected" completions.
            - `"chosen"`: The remainder of the "chosen" completion, with the prompt removed.
            - `"rejected"`: The remainder of the "rejected" completion, with the prompt removed.

    Examples:

    ```python
    >>> example = {
    ...     "chosen": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is blue."}
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."}
    ...     ]
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of `datasets.Dataset`:
    
    ```python
    >>> from trl import extract_prompt
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "chosen": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is blue."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sky."},
    ...         ],
    ...     ],
    ...     "rejected": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is green."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sea."},
    ...         ],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = dataset.map(extract_prompt)
    >>> dataset[0]
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```
    """
    # Some dataset add a `"prompt"` column, even though the prompt is implicit and included in the "chosen" and
    # "rejected" completions. E.g.:
    # {"prompt": "What color is the sky?",
    #  "chosen": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
    #  "rejected": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}]}
    # That's why we check if the prompt is also conversational before deciding not to extract it.
    if "chosen" not in example or "rejected" not in example:  # not a preference example
        return example
    if "prompt" in example:
        # Both conversational or both non-conversational
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})
    
def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        # chosen = features["answer_0"] if features["score_0"] >= features["score_1"] else features["answer_1"]
        # rejected = features["answer_1"] if features["score_0"] >= features["score_1"] else features["answer_0"]
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }
        
def prepare_dataset(
        dataset, #: Union[Dataset, IterableDataset],
        processing_class, #: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args, #: DPOConfig,
        dataset_name, #: str,
    ) :# -> Union[Dataset, IterableDataset]:
        # # Build the kwargs for the `map` function
        # map_kwargs = {"writer_batch_size": 10}
        map_kwargs = dict()
        # if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
        #     map_kwargs["num_proc"] = args.dataset_num_proc


        from accelerate import PartialState
        with PartialState().main_process_first():
                
            # # Extract prompt if needed
            # if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            #     map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            # if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            #     map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": None}, **map_kwargs
            )

            # # Tokenize the dataset
            # if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            #     map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                tokenize_row,
                remove_columns=["question", "answer_0", "answer_1"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": 512, # TODO: args.max_prompt_length,
                    "max_completion_length": 512, # TODO: args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

def pad_to_length(
    tensor, #: torch.Tensor, 
    length: int, 
    pad_value, #: Union[int, float], 
    dim): #: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
        
def dpo_collate_fn(dataset, tokenizer):
    chosen_input_ids = [prompt + chosen for (prompt, chosen) in zip(dataset["prompt_input_ids"], dataset["chosen_input_ids"])]
    rejected_input_ids = [prompt + rejected for (prompt, rejected) in zip(dataset["prompt_input_ids"], dataset["rejected_input_ids"])]

    # loss_mask_chosen: zeros_like(prompt) + ones_like(chosen)
    loss_mask_chosen = [[0] * len(prompt) + [1] * len(chosen)
                    for prompt, chosen in zip(dataset["prompt_input_ids"], dataset["chosen_input_ids"])]
    loss_mask_rejected = [[0] * len(prompt) + [1] * len(rejected)
                    for prompt, rejected in zip(dataset["prompt_input_ids"], dataset["rejected_input_ids"])]

    # Pad to same length within batch
    padded = tokenizer.pad(
        {"input_ids": chosen_input_ids + rejected_input_ids},
        padding=True,
        return_tensors="pt"
    )
    mask_padded = tokenizer.pad(
        {"input_ids": loss_mask_chosen + loss_mask_rejected},
        padding=True,
        return_tensors="pt"
    )

    batch_size = len(chosen_input_ids)
    return {
        "input_ids_chosen": padded["input_ids"][:batch_size],
        "input_ids_rejected": padded["input_ids"][batch_size:],
        "attention_mask_chosen": padded["attention_mask"][:batch_size],
        "attention_mask_rejected": padded["attention_mask"][batch_size:],
        "loss_mask_chosen": mask_padded["input_ids"][:batch_size],
        "loss_mask_rejected": mask_padded["input_ids"][batch_size:],
    }


def run_dp(
    rank, world_size, backend,
    dataset_name='',
    model_max_length=128,
    n_epochs=10,
    batch_size=128,
    learning_rate=1e-6):
    workdir = f'./workdir'
    os.makedirs(workdir, exist_ok=True)

    config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
    config.save_pretrained(workdir)
    
    ### Distributed Training Setup
    setup(rank, world_size, backend)
    
    # model = GPT2LMHeadModel(config=config).to(rank)
    # load model = meta-llama/Llama-3.2-1B-Instruct
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float32,
    ).to(rank)
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float32,
    ).to(rank)
    ref_model.eval()  # Set to eval mode
    for param in ref_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = datasets.load_dataset(dataset_name, use_auth_token=True, split="train")
    # dataset = dataset.select(range(1024))
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token        

    # ### MAKE SMALLER
    # dataset = {}
    # dataset['train'] = all_dataset[:5000]
    # dataset['validation'] = all_dataset[:1000]
    # dataset['test'] = all_dataset[:100]
    # ###
    
    dataset = prepare_dataset(
        dataset=dataset,
        processing_class=tokenizer,
        args=None,
        dataset_name=dataset_name,
    )
    
    dataset = dpo_collate_fn(dataset, tokenizer)
    
    # convert dictionary to dataset
    dataset = datasets.Dataset.from_dict(dataset)
    
    # dataset to device
    dataset.set_format(type='torch', device=rank)
    
    
    ### Get Partition of the Training Dataset on Device {rank}
    train_loader = partition_dataset(rank, world_size, dataset, batch_size=batch_size, collate_fn=None)
    # val_loader = partition_dataset(rank, world_size, val_dataset, batch_size=batch_size, collate_fn=None)
    # test_loader = partition_dataset(rank, world_size, test_dataset, batch_size=batch_size, collate_fn=None)
    # val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=None)
    # test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=None)
    

    total_time = []
    total_tokens_per_sec = []

    for epoch_idx in range(n_epochs):
        desc = f'rank {rank}/{world_size} epoch {epoch_idx}/{n_epochs}'

        start = time.time()
        avg_tokens_per_sec, _  = train(
                                    model=model,
                                    ref_model=ref_model,
                                    tokenizer=tokenizer,
                                    optimizer=optimizer,
                                    examples=train_loader,
                                    batch_size=batch_size,
                                    collate_fn=None,
                                    desc=desc,
                                    rank=rank,
                                    average_gradients_fn=average_gradients)
        end = time.time()
    #     if not PYTEST:
    #         training_time = end - start
    #         print(f'Epoch {epoch_idx} on Rank {rank}: Training Time = {training_time}, Tokens_per_sec = {avg_tokens_per_sec}')
    #         total_time.append(training_time)
    #         total_tokens_per_sec.append(avg_tokens_per_sec)

    #         validation_loss = evaluate_loss(
    #             model=model,
    #             examples=val_loader,
    #             batch_size=batch_size,
    #             collate_fn=None,
    #             desc=desc)

    #         print(f'Epoch {epoch_idx} on Rank {rank}: Validation Loss = {validation_loss}')

    #         gen_sents = generate(
    #             model=model,
    #             examples=dataset['test'],
    #             src_key=src_key,
    #             tgt_key=tgt_key,
    #             tokenizer=tokenizer,
    #             model_max_length=model_max_length,
    #             device=rank,
    #             desc=desc)

    #         gen_examples = []
    #         for example, gen_sent in zip(dataset['test'], gen_sents):
    #             gen_examples.append({'example': example, 'gen': gen_sent})
    #         json.dump(gen_examples, open(
    #             f'{workdir}/rank{rank}_gen_epoch{epoch_idx}.json', 'w'), indent=4)

    #         eval_scores = evaluate_bleu(
    #             examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
    #         print(f'Epoch {epoch_idx} on Rank {rank}: {eval_scores}')

    #         json.dump(
    #             {'validation_loss': validation_loss, **eval_scores, 'training_time': training_time, 'tokens_per_sec': avg_tokens_per_sec},
    #             open(f'{workdir}/rank{rank}_results_epoch{epoch_idx}.json', 'w'))
    #     else:
    #         save_grad_weights(model, rank)
    #         break
    # if not PYTEST:
    #     # You only get the average training time and tokens_per_second per device
    #     # To compute the throughput, you need to sum up the tokens_per_sec across all the devices based on epochs
    #     print(f'Rank {rank} training time: avg:{np.mean(total_time)}, std:{np.std(total_time)}, \
    #     tokens_per_second: avg: {np.mean(total_tokens_per_sec)}, std:{np.std(total_tokens_per_sec)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytest', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='openai/webgpt_comparisons') #'argilla/ultrafeedback-binarized-preferences-cleaned')
    parser.add_argument('--model_max_length', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    if args.pytest:
        PYTEST = True
    else:
        PYTEST = False

    processes = []
    
    # init wandb
    wandb.init(project="hw")
    

    # ASSIGNMENT 4.1
    '''Create Process to start distributed training

    Hint:
    1. You can use Process from torch.distributed to define the process
    2. You should start the processes to work and terminate resources properly
    '''
    # BEGIN SOLUTION
    world_size = args.world_size  # TODO: Define the number of GPUs
    backend = 'nccl'  # TODO: Define your backend for communication, we suggest using 'nccl'
    
    for rank in range(world_size):
        p = Process(
            target=run_dp, 
            args=(
                rank, 
                world_size, 
                backend,
                args.dataset, 
                args.model_max_length, 
                args.n_epochs, 
                args.batch_size, 
                args.learning_rate
            ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        p.terminate()
    # END SOLUTION