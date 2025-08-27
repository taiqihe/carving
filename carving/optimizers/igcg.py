"""Improved GCG optimizer from Improved Techniques for Optimization-Based Jailbreaking on Large Language Models"""

import itertools

import numpy as np
import torch

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)
max_retries = 20
progress_threshold = 0.5  # only valid for progressive expansion. Hardcoded for now


class IGCGOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        steps=500,
        batch_size=512,
        topk=256,
        top_p=7,
        temp=1,
        patience=3,
        filter_cand=True,
        freeze_objective_in_search=False,
        progressive_expansion=False,
        compute_combined_loss=True,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.steps = steps
        self.batch_size = batch_size
        self.topk = topk
        self.top_p = top_p
        self.temp = temp
        self.patience = patience
        self.filter_cand = filter_cand
        self.freeze_objective_in_search = freeze_objective_in_search
        self.progressive_expansion = progressive_expansion
        self.compute_combined_loss = compute_combined_loss
        self.rng = np.random.default_rng()


    def token_gradients(self, sigil, input_ids, state=None):
        """
        Computes gradients of the loss with respect to the coordinates.
        Operating with constraint mapped indices
        """
        sigil.model.train()  # necessary to trick HF transformers to allow gradient checkpointing
        # one_hot: attack_len x n_constraint_toks
        one_hot = torch.zeros(input_ids.shape[1], sigil.num_constraint_embeddings, **self.setup)
        one_hot.scatter_(1, input_ids[0].unsqueeze(1), 1)
        one_hot.requires_grad_()
        inputs_embeds = (one_hot @ sigil.constraint_embeddings.weight).unsqueeze(0)
        adv_loss = sigil.objective(inputs_embeds=inputs_embeds, mask_source=input_ids, state=state).mean()
        (input_ids_grads,) = torch.autograd.grad(adv_loss, [one_hot])
        sigil.model.eval()  # necessary to trick HF transformers to disable gradient checkpointing
        return input_ids_grads
        # adv_loss.backward() # alternative for reentrant grad checkpointing
        # return one_hot.grad


    @torch.no_grad()
    def sample_candidates(self, input_ids, grad, batch_size, topk=256):
        """
        We sample batch_size sequnences from topk by sampling the token position and id independently.
        """
        # grad: attack_len x n_constraint_toks
        top_indices = torch.topk(-grad, k=topk, dim=-1).indices.to(input_ids.dtype) # attack_len x topk
        
        idxes = list(itertools.product(range(top_indices.shape[0]), range(topk)))
        new_seq_pos, new_token_pos = torch.tensor(self.rng.choice(idxes, size=batch_size, replace=False).T, device=top_indices.device, dtype=torch.int32)
        
        new_input_ids = input_ids.repeat(batch_size, 1)
        new_input_ids[torch.arange(batch_size), new_seq_pos] = top_indices[new_seq_pos, new_token_pos]
        return new_input_ids, new_seq_pos

    @torch.no_grad()
    def get_filtered_cands(self, sigil, candidate_ids, threshold):
        if self.filter_cand:
            candidate_ids = sigil.map_to_tokenizer_ids(candidate_ids)
            candidate_is_valid = sigil.constraint.is_tokenization_safe(candidate_ids)
            if sum(candidate_is_valid) > threshold:
                return candidate_is_valid, True
            else:
                print(f"Not enough valid candidate accepted out of {len(candidate_ids)} candidates.")
                return candidate_is_valid, False
        else:
            return torch.ones(candidate_ids.shape[0]), True
    
    def combine_top_p_samples(self, loss, candidates, pos_changed, candidate_is_valid):
        ranking = loss.argsort()
        selections = []
        for i in ranking:
            if candidate_is_valid[i]:
                selections.append(i)
                if len(selections) >= self.top_p:
                    break
        combined_candidate = candidates[selections[-1]].clone()
        for j in range(len(selections)-2, -1, -1):
            i = selections[j]
            pos = pos_changed[i]
            combined_candidate[pos] = candidates[i][pos]
        estimated_loss = loss[selections].mean().detach()
        return combined_candidate, estimated_loss

    def solve(self, sigil, initial_guess=None, initial_step=0, dryrun=False, **kwargs):
        if len(sigil.constraint) < self.topk:
            new_topk = len(sigil.constraint) // 2
            print(f"Constraint space of size {len(sigil.constraint)} too small for {self.topk} topk entries. Reducing to {new_topk}.")
            self.topk = new_topk

        if self.progressive_expansion:
            if hasattr(sigil, "progressive_expansion"):
                sigil.progressive_expansion = True
            else:
                raise ValueError(f"Sigil {sigil} does not support progressive expansion.")

        # Initialize solver
        best_loss = float("inf")
        if initial_guess is None:
            prompt_ids = sigil.constraint.draw_random_sequence(device=self.setup["device"])
        else:
            if len(initial_guess) != sigil.num_tokens:
                raise ValueError(f"Initial guess does not match expected number of tokens ({sigil.num_tokens}).")
            else:
                prompt_ids = torch.as_tensor(initial_guess, device=self.setup["device"]).unsqueeze(0)
        # prompt_ids: 1 x attack_len

        # print(f"Initial Prompt is: {sigil.tokenizer.decode(prompt_ids[0])}")
        best_prompt_ids = prompt_ids.clone()
        init_state = initial_step if self.freeze_objective_in_search else None
        # objective w/ ids: tokenizer space; w/ embs: constraint space
        best_loss = sigil.objective(input_ids=best_prompt_ids, state=init_state).to(dtype=torch.float32).mean().item()
        prompt_ids = sigil.map_to_constraint_ids(prompt_ids)

        for iteration in range(initial_step, self.steps):
            # Optionally freeze objective state
            state = iteration if self.freeze_objective_in_search else None
            if self.progressive_expansion and best_loss < progress_threshold:
                print(f"Loss threshold reached with loss {best_loss} in step {iteration}, expanding target length.")
                state = f"expand_{state}"
                best_loss = float("inf")
            # Aggregate gradients
            grad = self.token_gradients(sigil, prompt_ids, state=state)
            normalized_grad = grad / grad.norm(dim=-1, keepdim=True)

            # Select candidates
            for _ in range(max_retries):
                # Sample candidates
                candidates, changed_pos = self.sample_candidates(prompt_ids, normalized_grad, self.batch_size, self.topk)
                # Filter candidates:
                candidate_is_valid, valid_candidates_found = self.get_filtered_cands(sigil, candidates, self.batch_size//2)
                if valid_candidates_found:
                    break

            # Search
            # loss = torch.zeros(len(candidates), dtype=torch.float32, device=self.setup["device"])
            # print(f"Unique fwd passes: {len(unique_candidates)}")
            with torch.no_grad():
                loss = sigil.objective(inputs_embeds=sigil.constraint_embeddings(candidates), state=state).to(dtype=torch.float32).mean(dim=1)

            # Return best from batch:
            best_candidate, estimated_loss = self.combine_top_p_samples(loss, candidates, changed_pos, candidate_is_valid)
            prompt_ids = best_candidate.unsqueeze(0)
            
            best_candidate = sigil.map_to_tokenizer_ids(prompt_ids)
            if self.top_p == 1 or not self.compute_combined_loss:
                loss_for_best_candidate = estimated_loss
            else:
                loss_for_best_candidate = sigil.objective(input_ids=best_candidate, state=None).to(dtype=torch.float32).mean().detach()

            if loss_for_best_candidate < best_loss:
                best_loss = loss_for_best_candidate.item()
                best_prompt_ids = best_candidate

            self.callback(sigil, best_candidate, best_prompt_ids, loss_for_best_candidate, iteration, **kwargs)
            if dryrun:
                break

        return best_prompt_ids  # always return with leading dimension
