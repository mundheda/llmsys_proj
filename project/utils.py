import os
import sys
from pathlib import Path
from sacrebleu.metrics import BLEU
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import torch
import tqdm
import time
import numpy as np

import wandb

cousin_dir = Path(__file__).resolve().parents[1]


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    assert os.path.exists(f'{workdir}/config.json')
    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def evaluate_bleu(examples, gen_sents, tgt_key):
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }

def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, device):
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length + 1
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    token_ids = torch.tensor(token_ids, device=device)
    tgt_token_mask = torch.tensor(tgt_token_mask, device=device)
    
    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'label_token_weights': tgt_token_mask[:, 1:]
    }

def loss_fn(batch, model):
    logits = model(input_ids=batch['input_ids']).logits

    loss = torch.nn.functional.cross_entropy(
        input=logits.reshape((-1, logits.shape[-1])),
        target=batch['labels'].reshape(-1),
        reduction='none')

    return (torch.sum(loss * batch['label_token_weights'].reshape(-1)) /
            torch.sum(batch['label_token_weights']))


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    model.eval()
    losses = []

    for batch in (prog_bar := tqdm.tqdm(examples, desc=f'Evaluating ({desc})')):
        with torch.no_grad():
            loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(model, examples, src_key, tgt_key, tokenizer, model_max_length, device, desc):
    model.eval()

    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            with torch.no_grad():
                logits = model(
                    input_ids=torch.tensor([token_ids], device=device)
                ).logits[0, -1]
                gen_id = torch.argmax(logits).item()

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def get_batch_logps(
    logits, #: "torch.Tensor", 
    labels, #: "torch.Tensor", 
    label_pad_token_id, #: int = IGNORE_INDEX
    loss_mask,
): # -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    loss_mask = loss_mask[:, 1:].clone()
    logits = logits[:, :-1, :]
    end_mask = labels != label_pad_token_id
    loss_mask = loss_mask * end_mask
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # print("mean logps (after loss mask)", (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1))
    
    reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    
    return (per_token_logps * loss_mask).sum(-1), reward

def compute_preference_loss(
        policy_chosen_logps, #: "torch.Tensor",
        policy_rejected_logps, #: "torch.Tensor",
        reference_chosen_logps, #: Optional["torch.Tensor"],
        reference_rejected_logps, #: Optional["torch.Tensor"],
    ): # -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = logratios - ref_logratios
        label_smoothing = 0.1
        
        from torch.nn import functional as F
        label_smoothing = 0.0
        beta = 0.1
        losses = -F.logsigmoid(beta * logits)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
        

def train(model, ref_model, tokenizer, optimizer, examples, batch_size, collate_fn, desc, rank=0, average_gradients_fn=None):
    model.train()
    
    tokens_per_sec = []
    tokens_num = []
        
    gradient_accumulation_steps = 4
    
    for i, batch in enumerate(prog_bar := tqdm.tqdm(examples, desc=f'Training ({desc})')):
        
        t0 = time.time()
        optimizer.zero_grad()
        
        if i % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            
            chosen_logits = model(
                input_ids=batch['input_ids_chosen'],
                attention_mask=batch['attention_mask_chosen']
            ).logits
            # loss_mask: remove the system and user prompt
            chosen_logp, chosen_avg_logp = get_batch_logps(
                chosen_logits, 
                batch['input_ids_chosen'], 
                label_pad_token_id=tokenizer.pad_token_id,
                loss_mask=batch['loss_mask_chosen'],
            )
        
            rejected_logits = model(
                input_ids=batch['input_ids_rejected'],
                attention_mask=batch['attention_mask_rejected']
            ).logits
            rejected_logp, rejected_avg_logp = get_batch_logps(
                rejected_logits, 
                batch['input_ids_rejected'], 
                label_pad_token_id=tokenizer.pad_token_id,
                loss_mask=batch['loss_mask_rejected'],
            )
        
        with torch.no_grad():
            ref_chosen_logits = ref_model(
                input_ids=batch['input_ids_chosen'],
                attention_mask=batch['attention_mask_chosen']
            ).logits
            ref_chosen_logp, _ = get_batch_logps(
                ref_chosen_logits, 
                batch['input_ids_chosen'], 
                label_pad_token_id=tokenizer.pad_token_id,
                loss_mask=batch['loss_mask_chosen'],
            )
            
            ref_rejected_logits = ref_model(
                input_ids=batch['input_ids_rejected'],
                attention_mask=batch['attention_mask_rejected'],
            ).logits
            ref_rejected_logp, _ = get_batch_logps(
                ref_rejected_logits, 
                batch['input_ids_rejected'], 
                label_pad_token_id=tokenizer.pad_token_id,
                loss_mask=batch['loss_mask_rejected'],
            )
            
        loss, chosen_reward, rejected_reward = compute_preference_loss(
            policy_chosen_logps=chosen_logp,
            policy_rejected_logps=rejected_logp,
            reference_chosen_logps=ref_chosen_logp.detach(), 
            reference_rejected_logps=ref_rejected_logp.detach(), 
        )
        
        # print("chosen_logp.shape", chosen_logp[:1])
        # print("rejected_logp.shape", rejected_logp[:1])
        # print("ref_chosen_logps.shape", ref_chosen_logps[i*batch_size:(i+1)*batch_size][:1])
        # print("ref_rejected_logps.shape", ref_rejected_logps[i*batch_size:(i+1)*batch_size][:1])
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        ''' Call the `average_gradients_fn` function to reduce and broadcast the gradients in Data Parallel
            Just few lines of code. Think simply.
        '''
        # BEGIN SOLUTION
        if(average_gradients_fn is not None):
            average_gradients_fn(model)
        # END SOLUTION
        
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(examples):
            optimizer.step()
        # optimizer.step()
        batch_time = time.time() - t0
        tokens = np.prod(batch['input_ids_chosen'].shape) + np.prod(batch['input_ids_rejected'].shape)
        tokens_per_sec.append(tokens / batch_time)
        tokens_num.append(tokens)
        prog_bar.set_postfix(
            tokens_per_sec=tokens / batch_time,
            loss=loss.item())
    
        acc = (chosen_avg_logp > rejected_avg_logp)
        acc_mean = torch.mean(acc.float())
        
        # push loss to wandb
        wandb.log({"loss": loss.item()})
        wandb.log({"tokens_per_sec": np.mean(tokens_per_sec)})
        wandb.log({"acc_mean": acc_mean.item()})
        wandb.log({"chosen_reward": chosen_reward.mean().item()})
        wandb.log({"rejected_reward": rejected_reward.mean().item()})
        
    # save model
    if rank == 0:
        model.save_pretrained(f'/u/wchen11/llmsys_s25_project/model{rank}')
        tokenizer.save_pretrained(f'/u/wchen11/llmsys_s25_project/model{rank}')
        
    return np.mean(tokens_per_sec), tokens_num


def save_grad_weights(model, rank):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.data.detach().cpu()
    torch.save(gradients, f'{cousin_dir}/tests/model{rank}_gradients.pth')