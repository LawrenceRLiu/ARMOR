import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
import itertools
import hydra
import tqdm
import random
import time
import wandb
from functools import partial # <-- Import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Optional, Union, List, Literal
from dataclasses import dataclass, field

from src.compression_parent import CompressedLinear
from src.sparse_compress import SparseLinear
from src.utils import normalizer as normalize
from src.utils import utils
from src.utils.blockwise_diag_matricies import BlockwiseDiagMatrix
    
class BlockCompressLearnable(nn.Module):
    original_weight: torch.FloatTensor
    importance_weight: Union[None, torch.FloatTensor] #shape of (d_in) if not None
    A: BlockwiseDiagMatrix #shape of (d_out, d_in)
    B: BlockwiseDiagMatrix #shape of (d_in, d_out)
    naive_compression_module: CompressedLinear
    block_size: int
    eps: float = 1e-8
    prev_dim: int = -1
    
    
    
    def __init__(self, original_weight: torch.FloatTensor, 
                 naive_compression_module: CompressedLinear,
                 block_size: Union[int, Tuple[int, int]],
                 importance_weight: Optional[torch.FloatTensor] = None,
    ) -> None:
        """Initializes the PermutedSparseWeight class.

        Args:
            original_weight (torch.FloatTensor): The original weight matrix to be approximated.
            naive_compression_module (CompressedLinear): The naive compression module (e.g., SparseLinear).
            block_size (Union[int, Tuple[int, int]]): The size of the blocks for the block-diagonal matrices A and B. 
                If an integer is provided, both dimensions will use the same block size, otherwise a tuple specifying (block_size_A, block_size_B).
            importance_weight (Optional[torch.FloatTensor], optional): Importance weights for the input dimensions. Defaults to None.
        """
        
        super(BlockCompressLearnable, self).__init__()
        
        
        d_out, d_in = original_weight.shape
        
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
            
            
        assert d_in % block_size[1] == 0, f"d_in = {d_in} must be divisible by block_size = {block_size[1]}"
        assert d_out % block_size[0] == 0, f"d_out = {d_out} must be divisible by block_size = {block_size[0]}"
        
        self.d_out = d_out
        self.d_in = d_in
            
        self.block_size = block_size
        
        #initalize each of the permutation matricies
        self.A = BlockwiseDiagMatrix(
            d = d_out,
            block_size=block_size[0],
            initalize_as_identity=True
        )
        self.B = BlockwiseDiagMatrix(
            d = d_in,
            block_size=block_size[1],
            initalize_as_identity=True
        )
                                                         
            
        self.original_weight = original_weight.detach().clone()
        self.original_weight.requires_grad = False
        
        self.importance_weight = importance_weight
        
        with torch.no_grad():
            self.loss_scaling = {"mean":1.0, "sum": 1.0}    
            self.loss_scaling = {"mean":self.recon_loss(reduction = "mean", zero_sub = True).item(),
                                    "sum": self.recon_loss(reduction = "sum", zero_sub = True).item()}
        
        self.naive_compression_module = naive_compression_module
        self.to(original_weight.device)
            

        
    #turn of gradient t
    def forward(self, permutations_to_ignore: Tuple[set[int],set[int]] = [{},{}], 
                mask_kwargs: Optional[dict]= {}):
        out = self.A @ self.naive_compression_module.reconstruct() @ self.B
        return out
            
        
    def recon_loss(self, reduction: Literal["mean", "sum", "none"] = "mean",
                   zero_sub:bool = False, recon_weight: Optional[torch.FloatTensor] = None,
                   **kwargs):
        if recon_weight is None:
            if zero_sub:
                recon_weight = torch.zeros_like(self.original_weight) #used to get the loss scaling
            else:
                recon_weight = self(**kwargs)
        
        #reconstruction loss
        recon_loss_elementwise = (recon_weight - self.original_weight) ** 2 if self.importance_weight is None else (recon_weight - self.original_weight) ** 2 * self.importance_weight.unsqueeze(0)

        if reduction == "mean":
            return recon_loss_elementwise.mean()/self.loss_scaling["mean"] #scale the loss by the scaling factor
        elif reduction == "sum":
            return recon_loss_elementwise.sum()/self.loss_scaling["sum"] #scale the loss by the scaling factor
        elif reduction == "none":
            return recon_loss_elementwise


@dataclass
class SelectionConfig:
    random: bool = False #random or gradient based
    masked: bool = False #whether to only select from the masked elements
    norm: Literal["L1, L2", "Linf"] = "L1" #the norm to use for the gradient based selection
    greedy: bool = True #whether to use greedy or random selection for the gradient based selection
    
    @classmethod
    def from_name(cls, name: str):
        #if the name is just random
        if name == "random":
            return cls(random=True, masked=False)
        prefix,name = name.split("_",1)
        assert prefix=="gradient", f"Unknown selection method {name}, expected 'random' or 'gradient_*'."
        
        terms = name.split("_")
        defaults = {"masked": False, "norm": "L1", "greedy": False}
        for term in terms:
            if term == "masked":
                defaults["masked"] = True
            if term == "all":
                defaults["masked"] = False
            elif term in ["L1", "L2", "Linf"]:
                defaults["norm"] = term
            elif term in ["greedy", "random"]:
                defaults["greedy"] = term == "greedy"
            else:
                raise ValueError(f"Unknown selection method {name}, expected 'random' or 'gradient_*_masked_*_{'L1'|'L2'|'Linf'}'.")
        
        return cls(random=False, **defaults)
    


@torch.no_grad()         
def sparse_core_step(trainable_sparse: BlockCompressLearnable,
                            n_times: int = 1,
                            select: Literal["random", "gradient_greedy", "gradient_random"] = "random",
                            )-> None:
    
    
    """Performs n_times of discrete optimization on the sparse core of the compression module.

    Args:
        trainable_sparse (BlockCompressLearnable): The trainable overall module representing the ARMOR decomposition.
        n_times (int, optional): The number of times to perform discrete optimization. Defaults to 1.
        select (Literal["random", "gradient_greedy", "gradient_random"], optional): The method to use for selecting the groups to update. Defaults to "random".
    """
    
    #check that the naive compression module is SparseLinear
    assert isinstance(trainable_sparse.naive_compression_module, SparseLinear), \
        f"Currently only SparseLinear is supported, but got {type(trainable_sparse.naive_compression_module)}."
        
    #calculate the number or blocks
    block_size_0 = trainable_sparse.block_size[0]
    block_size_1 = trainable_sparse.block_size[1]
    n_blocks_0  = trainable_sparse.original_weight.shape[0] // block_size_0 #number of blocks in the 0th dimension
    n_blocks_1  = trainable_sparse.original_weight.shape[1] // block_size_1 #number of blocks in the 1st dimension
    n_blocks_total = n_blocks_0 * n_blocks_1 #total number of blocks
    
    #get the group size
    group_size = trainable_sparse.naive_compression_module.sparse_group
    
    
    #check that the the input shape is divisible by group_size
    assert trainable_sparse.original_weight.shape[1] % group_size == 0, f"Expected that d_in = {trainable_sparse.original_weight.shape[1]} to be divisible by group_size = {group_size}. non divisible input shapes are currently not supported."
    #check that the block_size_1 is divisible by group_size
    assert block_size_1 % group_size == 0, f"Expected that block_size_1 = {block_size_1} to be divisible by group_size = {group_size}. non divisible block sizes are currently not supported."
    
    n_groups_per_row = trainable_sparse.original_weight.shape[1]// group_size #number of codes per row, this is the number of groups we can select from in each block
    n_groups_per_block_row = block_size_1 // group_size #number of codes per row in each block, this is the number of groups we can select from in each block
    #check our counts are correct
    
    A_diag = trainable_sparse.A.diag_blocks #shape of (n_blocks_0, block_size_0, block_size_0)
    B_diag = trainable_sparse.B.diag_blocks #shape of (n_blocks_1, block_size_1, block_size_1)
    if trainable_sparse.importance_weight is not None:
        B_diag = B_diag * torch.sqrt(
            trainable_sparse.importance_weight.view(n_blocks_1, -1, block_size_1)
        )

    
    #adjust the full matrices by the normalizer
    naive_normalizer = trainable_sparse.naive_compression_module.normalizer
    for j,i in enumerate(naive_normalizer.norm_order):
        if i == 1:
            A_diag = A_diag * naive_normalizer.norms[j].view(n_blocks_0, 1, block_size_0)
        elif i == 0:
            B_diag = B_diag * naive_normalizer.norms[j].view(n_blocks_1, block_size_1, 1)
    
    
   
    #calculate all possible non-zero indices
    n_nonzero = trainable_sparse.naive_compression_module.n_non_zero_per_group
    possible_non_zero_idxs = torch.combinations(torch.arange(group_size, device=trainable_sparse.original_weight.device),
                                                r=n_nonzero) #shape of (n_possible, n_nonzero)
    #if the  is greater than 100 raise a warning
    n_possible = possible_non_zero_idxs.shape[0]
    if n_possible > 100:
        print(f"Warning: {n_possible} possible codes, this may take a long time to optimize. Consider reducing the group size or number of non-zero elements per group.")
        
   
    selection_config = SelectionConfig.from_name(select)

    for i in range(n_times):
        
        # selection code
        idxs = torch.arange(n_blocks_total, device=trainable_sparse.original_weight.device)
        j = idxs // n_blocks_1 #this is the block index in the 0th dimension
        k = idxs % n_blocks_1 #this is the block index in the 1st dimension

        if selection_config.random:
            idx_0 = j*block_size_0 + torch.randint(0, block_size_0, (n_blocks_total,), device=trainable_sparse.original_weight.device) #shape of (n_blocks_total,)
            idx_1 = k * block_size_1 + torch.randint(0, n_groups_per_block_row, (n_blocks_total,), device=trainable_sparse.original_weight.device) * d
            
            
        else:
            # raise NotImplementedError("Gradient greedy selection is not implemented yet.")
            with torch.enable_grad():
                naive_non_normalized = trainable_sparse.naive_compression_module.reconstruct_(denormalize=False).detach().clone() #shape of (d_out, d_in)
                #turn on the gradient for the naive non-normalized weight
                naive_non_normalized.requires_grad = True
                #use it to get the loss
                loss = trainable_sparse.recon_loss(reduction="mean",recon_weight = trainable_sparse.A @ trainable_sparse.naive_compression_module.normalizer.denormalize(naive_non_normalized) @ trainable_sparse.B)
                loss.backward()
                grad = naive_non_normalized.grad #shape of (d_out, d_in)
            #reblock the gradient
            if not selection_config.masked:
                grad = grad.view(n_blocks_0, block_size_0, n_blocks_1, block_size_1).transpose(1, 2).reshape(n_blocks_total, block_size_0, block_size_1) #shape of (n_blocks_total, block_size_0, block_size_1)
                #reshape to the indvidual groups
                grad = grad.view(n_blocks_total, block_size_0 * n_groups_per_block_row, group_size) #shape of (n_blocks_total, block_size_0 * n_groups_per_block_row, d)
                if selection_config.norm == "L2":
                    #get the l2 norm of the gradient
                    grad_norm = torch.norm(grad, p=2, dim=-1) #shape of (n_blocks_total, block_size_0 * n_groups_per_block_row)
                elif selection_config.norm == "Linf":
                    #get the linf norm of the gradient
                    grad_norm = torch.norm(grad, p=float('inf'), dim=-1) #shape of (n_blocks_total, block_size_0 * n_groups_per_block_row)
                elif selection_config.norm == "L1":
                    #get the l1 norm of the gradient
                    grad_norm = torch.norm(grad, p=1, dim=-1) #shape of (n_blocks_total, block_size_0 * n_groups_per_block_row)
            elif "masked" in select:
                raise NotImplementedError("Gradient masked selection is not implemented yet.")
            else:
                raise ValueError(f"Unknown select method {select}, expected 'gradient_all' or 'gradient_masked'.")
            #get the max index for each block
            if selection_config.greedy:
                max_grads, selected_idxs = torch.max(grad_norm, dim=-1) #shape of (n_blocks_total, )
            else:
                selected_idxs = torch.multinomial(grad_norm, num_samples=1).squeeze(-1) #shape of (n_blocks_total, )
            #convert to idx_0 and idx_1
            idx_0 = j * block_size_0 + selected_idxs // n_groups_per_block_row #shape of (n_blocks_total,)
            idx_1 = k * block_size_1 + (selected_idxs % n_groups_per_block_row) * group_size #shape of (n_blocks_total,)
        
            
        group_idxs = torch.stack([idx_0, idx_1], dim=1) #shape of (n_blocks_total, 2)

        original_naive = trainable_sparse.naive_compression_module.reconstruct().clone()
        #zero out the groups selected

        original_naive[group_idxs[:, 0].unsqueeze(1), 
                      group_idxs[:, 1].unsqueeze(1) + torch.arange(group_size, device = group_idxs.device).unsqueeze(0)] = 0.0

        #use this to get W_remaining, the remaining weight matrix after subtracting the untouched subvectors
        W_remaining = trainable_sparse.A @ original_naive @ trainable_sparse.B - trainable_sparse.original_weight #shape of (d_out, d_in)

        W_remaining = W_remaining * torch.sqrt(trainable_sparse.importance_weight.unsqueeze(0)) if trainable_sparse.importance_weight is not None else W_remaining

        #block W_remaining
        W_remaining = W_remaining.view(n_blocks_0, block_size_0, n_blocks_1, block_size_1).transpose(
                    1, 2).reshape(n_blocks_total, block_size_0, block_size_1) #shape of (n_blocks_total, block_size_0, block_size_1)

        # get a and B
        a = A_diag[j, :, idx_0 % block_size_0] #shape of (n_blocks_total, block_size_0)
        assert a.shape == (n_blocks_total, block_size_0), f"Expected a to have shape {(n_blocks_total, block_size_0)}, but got {a.shape}."

        #get the B matrix for the selected groups
        
        B = B_diag[k.unsqueeze(1), (group_idxs[:,1] % block_size_1).unsqueeze(1) + torch.arange(group_size, device = idx_1.device).unsqueeze(0),:] #shape of (n_blocks_total, d, block_size_1)
        assert B.shape == (n_blocks_total, group_size, block_size_1), f"Expected B to have shape {(n_blocks_total, group_size, block_size_1)}, but got {B.shape}."
        #shape of (n_blocks_total, group_size, block_size_1)

        
        #calculate the first order term
        first_order_term = 2.0 * torch.bmm(torch.bmm(B, W_remaining.transpose(1, 2)), #shape of (n_blocks_total, group_size, block_size_0)
                                            a.unsqueeze(2)).squeeze(2) #shape of (n_blocks_total, group_size)
        


        B_selected = B[:, possible_non_zero_idxs, :] #shape of (n_blocks_total, n_possible, n_non_zero, block_size_1)
        B_squared = torch.bmm(B_selected.view(-1, n_nonzero, block_size_1),
                                B_selected.view(-1,  n_nonzero, block_size_1).transpose(1, 2)) #shape of (n_blocks_total*n_possible, n_non_zero, n_non_zero)
        #add damping to the diagonal
        B_squared += torch.eye(n_nonzero, device=B.device) * 1e-9 
        #get the inverse with cholesky
        B_squared_inv = torch.cholesky_inverse(torch.linalg.cholesky_ex(B_squared)[0]) #shape of (n_blocks_total*n_possible, n_non_zero, n_non_zero)
        #shape of (n_blocks_total*n_possible, n_non_zero, n_non_zero)
        first_order_selected = 1/2 * first_order_term[:, possible_non_zero_idxs] #shape of (n_blocks_total, n_possible, n_non_zero)
        first_order_selected = first_order_selected.view(n_blocks_total * n_possible, n_nonzero, 1) #shape of (n_blocks_total*n_possible, n_non_zero, 1)
        #calculate the cost
        cost = -torch.bmm(first_order_selected.transpose(1, 2),
                            torch.bmm(B_squared_inv, first_order_selected)).squeeze(2).squeeze(1) #shape of (n_blocks_total*n_possible, )
        
        #the actual cost is the current cost divided by the squared l2 norm of a and with W_remaining's F norm squared add
        #we are ignoring the constant term since it does not affect the optimization
        #reshape 
        cost = cost.view(n_blocks_total, n_possible)
        
        #get the optimal mask
        optimal_mask = torch.argmin(cost, dim=1) #shape of (n_blocks_total, )

        #get the optimal non-zero indices
        optimal_non_zero_idxs = possible_non_zero_idxs[optimal_mask] #shape of (n_blocks_total, n_nonzero)
        
        #update the naive sparse
        
        #update the mask
        #first we zero out the groups selected in the mask 
        trainable_sparse.naive_compression_module.sparse_mask[group_idxs[:,0].unsqueeze(1),
                                                        group_idxs[:,1].unsqueeze(1) + torch.arange(group_size, device = group_idxs.device).unsqueeze(0)] = False

        #now we set the non-zero indices in the mask to True
        trainable_sparse.naive_compression_module.sparse_mask[group_idxs[:,0].unsqueeze(1),
                                                        optimal_non_zero_idxs + group_idxs[:,1].unsqueeze(1)] = True #shape of (n_blocks_total, n_nonzero)
        
        
        #check the mask 
        # assert torch.all(trainable_sparse.naive_compression_module.sparse_mask == o_mask), "Mask has changed, this should not happen."
        trainable_sparse.naive_compression_module.check_mask()
        
        #update the underlying/transformed matrix, called X in NoWag
        #first we calculate the sparse values 
        B_inv_optimal = B_squared_inv.view(n_blocks_total, n_possible, n_nonzero, n_nonzero)[torch.arange(n_blocks_total, device=B.device), optimal_mask] #shape of (n_blocks_total, n_non_zero, n_non_zero)
        first_order_optimal = first_order_selected.view(n_blocks_total, n_possible, n_nonzero)[
            torch.arange(n_blocks_total, device=B.device), optimal_mask].unsqueeze(2) #shape of (n_blocks_total, n_non_zero, 1)
        #calculate the sparse values
        sparse_values = -torch.bmm(B_inv_optimal, first_order_optimal).squeeze(2) #shape of (n_blocks_total, n_non_zero)
        #now scale by square of a's norm
        sparse_values = sparse_values / (torch.sum(a**2, dim = 1, keepdim = True) + trainable_sparse.eps) #shape of (n_blocks_total, n_non_zero)
        #update the sparse values
        trainable_sparse.naive_compression_module.X.data[group_idxs[:,0].unsqueeze(1),
                                                        optimal_non_zero_idxs + group_idxs[:,1].unsqueeze(1)] = sparse_values #shape of (n_blocks_total, n_non_zero)
            
        
        
                
        
        
            
        
        
        
        
        
        
        
        


        
@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.

    Attributes:
        n_iters (int): Number of training iterations to run.
        n_continous_updates_per_iter (int): Number of continuous updates per iteration.
        n_sparse_core_updates_per_iter (int): Number of sparse core updates per iteration. Must be non-negative.
        sparse_core_step_select str: Strategy for selecting group to update in the sparse core step, must be able to be parsed by SelectionConfig.from_name.
        overall_patience (int): Number of iterations to wait for improvement before stopping.
        loss_atol (float): Absolute tolerance for loss improvement to consider as progress.
        loss_rtol (float): Relative tolerance for loss improvement to consider as progress.
        logfile (Optional[str]): Path to the log file. If None, logging is disabled.
        log_freq (int): Frequency (in iterations) at which to log training progress.
        iter_save_path (Optional[str]): Path to save iteration checkpoints for the iteration ablation. If None, checkpoints are not saved.
        iter_save_freq (int): Frequency (in iterations) at which to save checkpoints. If negative, checkpoints are not saved.

    """
    n_iters: int = 100
    n_continous_updates_per_iter: int = 1
    n_sparse_core_updates_per_iter: int = 0
    sparse_core_step_select: str = "random"
    overall_patience: int = 10
    loss_atol: float = 1e-8
    loss_rtol: float = 1e-6
    logfile: Optional[str] = None
    log_freq: int = 1
    iter_save_path: Optional[str] = None
    iter_save_freq: int = -1
    
    def __post_init__(self):
        self.iter_ablation = (self.iter_save_path is not None) and (self.iter_save_freq > 0)
        print("self.iter_ablation", self.iter_ablation)


def initalize_optimizer(
    trainable_sparse: BlockCompressLearnable,
    optimizer_config: DictConfig):
    
    
    # #create the optimizers
    # trainable_sparse.A.init_optimizers(optimizer_config)
    # trainable_sparse.B.init_optimizers(optimizer_config)
    
    # params = []
    # #get all the parameters that are not in trainable_sparse.A and trainable_sparse.B
    # for name, param in trainable_sparse.named_parameters():
    #     if name.startswith("A.") or name.startswith("B."):
    #         continue
    #     if param.requires_grad:
    #         params.append(param)
            
    optimizer = instantiate(
        optimizer_config, 
        params=trainable_sparse.parameters(),
    )
    return optimizer

def get_divisors(x):
    divisors = []
    for i in range(1, int(x**0.5) + 1):
        if x % i == 0:
            divisors.append(i)
            if i != x // i:  # Avoid adding the square root twice for perfect squares
                divisors.append(x // i)
    return sorted(divisors)

class ARMOR_Linear(CompressedLinear):
    name = "ARMOR_Linear"

    
    def ARMOR_sparse_(
        self,
        naive_compression_config: DictConfig,
        block_diagonal_config: DictConfig,
        optimizer_config: DictConfig,
        training_config: DictConfig,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        training_config_overrides: Optional[dict] = {},
    ):
        
        torch.set_num_threads(1)
        
        training_config:TrainingConfig = instantiate(training_config,
                                      **training_config_overrides) if training_config_overrides is not None else \
            instantiate(training_config)
        
        if training_config.logfile is not None:
            os.makedirs(os.path.dirname(training_config.logfile), exist_ok=True)
        
        
        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )
        
        #naive compression config consits of both the init config and the compression config
        self.naive_compression_module = instantiate(
            naive_compression_config.init_config,
            weight=normalized_weight,
            verbose=self.verbose,
        )

        self.naive_compression_module.hessianDiag = self.get_hessianDiag()
        
        #compress the weight with the naive compression module
        self.naive_compression_module.compress(
            **naive_compression_config.compression_config)
        #clean the naive compression module
        self.naive_compression_module.clean()
        
        if isinstance(self.naive_compression_module, SparseLinear):
            self.naive_compression_module.uncompress_sparse_values() #so we can do discrete optimization easier
            
        #remove loss_weighting from the config
        del block_diagonal_config.importance_weight
        print("block diagonal config block size:", block_diagonal_config.block_size)
        trainable_sparse = BlockCompressLearnable(
            normalized_weight,
            naive_compression_module=self.naive_compression_module,
            importance_weight=self.get_hessianDiag(),
            # block_size=training_config.block_size_start,
            **block_diagonal_config,
        )
        #create the optimizers
        optimizer = initalize_optimizer(
            trainable_sparse=trainable_sparse,
            optimizer_config=optimizer_config,
        )
            
        
        start_time = time.time()
        #create a simple logger
        with torch.no_grad():
            # trainable_sparse()
            prev_iter_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs={"add_noise": False, "return_hard": True}).item()
            if self.verbose:
                print(f"Initial loss: {prev_iter_loss}")
            if self.use_wandb:
                self.wandb_queue.put({self.metric_name: prev_iter_loss,
                           self.step_metric: 0})
            # best_state_dict = copy.deepcopy(trainable_sparse.state_dict())
            
        remaining_patience = training_config.overall_patience
        for i in tqdm.tqdm(range(training_config.n_iters), disable = not self.verbose):
            
            #optimizer step
            for j in tqdm.tqdm(range(training_config.n_continous_updates_per_iter),  disable = (not self.verbose or training_config.n_continous_updates_per_iter<10)):
                
                #reset the optimizers
                optimizer.zero_grad()   
                
                recon_loss = trainable_sparse.recon_loss(reduction="mean") 
                loss = recon_loss
             
                loss.backward()
                #step the optimizers
                optimizer.step()
                    

            if training_config.n_sparse_core_updates_per_iter != 0:
                with torch.no_grad():
                    sparse_core_step(
                        trainable_sparse,
                        n_times=training_config.n_sparse_core_updates_per_iter,
                        select=training_config.sparse_core_step_select,
                    )
                    # raise ValueError("stop here, we are done with training")
            
            
            #loss stuff
            with torch.no_grad():
                current_loss = trainable_sparse.recon_loss(reduction="mean").item()
                if current_loss > (1 - training_config.loss_rtol) * prev_iter_loss or current_loss > prev_iter_loss - training_config.loss_atol:
                    remaining_patience -= 1

                    if remaining_patience == 0:
                        if self.verbose:
                            print("Loss converged, stopping early")
                        break
                else:
                    remaining_patience = training_config.overall_patience
                    # best_state_dict = copy.deepcopy(trainable_sparse.state_dict())
                prev_iter_loss = current_loss
            # )
                
            if i%training_config.log_freq == training_config.log_freq-1 or i==0:
                log_str = f"Iter: {i}, Loss: {current_loss}"
                if self.verbose:
                    print(log_str)
                if self.use_wandb:
                    log = {self.metric_name: current_loss,
                           self.step_metric: i+1}
                    if self.direct_wandb_log:
                        wandb.log(log)
                    else:
                        #put the log in the queue
                        self.wandb_queue.put(log)
                if training_config.logfile is not None:
                    with open(training_config.logfile, "a") as f:
                        f.write(log_str + "\n")
                        
                    
            if training_config.iter_ablation:
                if i % training_config.iter_save_freq == training_config.iter_save_freq - 1  or (i==0 or i==training_config.n_iters-1):
                    print("saving iter", i)
                    #save the first iter and last iter and every iter_save_freq iterations
                    state_dict= {"A": trainable_sparse.A.state_dict(),
                                    "B": trainable_sparse.B.state_dict(),
                                    "naive_compression_module": trainable_sparse.naive_compression_module.state_dict()}
                    save_path = training_config.iter_save_path.replace("{iter}", str(i))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print("save_path", save_path)
                    torch.save(state_dict, save_path)
                    
            
        # trainable_sparse.load_state_dict(best_state_dict)
        
        if self.verbose:
            print("time taken to train:", time.time() - start_time)
            print("final loss:", trainable_sparse.recon_loss(reduction="mean").item())
            print("Finished training") 
            
        
        
        if isinstance(trainable_sparse.naive_compression_module, SparseLinear):
            #we need to compress the sparse values
            trainable_sparse.naive_compression_module.compress_sparse_values()
        self.A = trainable_sparse.A
        self.B = trainable_sparse.B
        

        
        assert torch.allclose(
            self.reconstruct_(denormalize=False),
            trainable_sparse(),
            rtol=1e-5, atol=1e-5
        )
        
        del trainable_sparse.naive_compression_module

    def load_iter_state(self, state_dict_path: str):
        state_dict = torch.load(state_dict_path, map_location=self.original_weight.device)
        self.A.load_state_dict(state_dict["A"])
        self.B.load_state_dict(state_dict["B"])
        self.naive_compression_module.uncompress_sparse_values()
        self.naive_compression_module.load_state_dict(state_dict["naive_compression_module"])
        self.naive_compression_module.compress_sparse_values()
        self.compressed = True
        
    def compress(self, 
               naive_compression_config: DictConfig,
        block_diagonal_config: DictConfig,
        optimizer_config: DictConfig,
        training_config: DictConfig,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        training_config_overrides: Optional[dict] = {},
    ):
        self.compressed = True
        return self.ARMOR_sparse_(
            naive_compression_config = naive_compression_config,
            block_diagonal_config = block_diagonal_config,
            optimizer_config = optimizer_config,
            training_config = training_config,
            normalizer = normalizer,
            normalizer_kwargs = normalizer_kwargs,
            training_config_overrides = training_config_overrides,
        )
    
        
    
    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        if self.forward_method == "reconstruct":
            if self.denormalization_method == "otf":
                y = F.linear(
                    self.normalizer.denormalize_otf_in(x),
                    self.reconstruct(denormalize=False),
                )
                y = self.normalizer.denormalize_otf_out(y) + (
                    self.bias if self.bias is not None else 0
                )
            else:
                # tqdm.tqdm.write(f"x dtype {x.dtype}, denormalize dtype {self.reconstruct(denormalize = self.denormalization_method == 'reconstruct').dtype}")
                y = F.linear(
                    x,
                    self.reconstruct(
                        denormalize=self.denormalization_method == "reconstruct"
                    ),
                    self.bias,
                )
        else:
            assert (
                self.denormalization_method == "otf"
            ), "on the fly denormalization is only supported for on the fly sparsity"
            
            x = self.normalizer.denormalize_otf_in(x)
            #multiply by B
            y = self.B(x.transpose(-2, -1)).transpose(-2, -1)
            #apply through the naive compression module
            y = self.naive_compression_module(y)
            #apply through A
            y = self.A(y.transpose(-2, -1)).transpose(-2, -1) #shape of (batch_size, d_out, d_in) 
            #denormalize the output
            y = self.normalizer.denormalize_otf_out(y)

        return y
    
    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        #reconstruct the weight
        weight = self.A(self.B(self.naive_compression_module.reconstruct(), leading=False), leading=True)
        

        if denormalize:
            weight = self.normalizer.denormalize(weight)
            

            
        return weight
    
    def blank_recreate(self,
                       block_diagonal_config: DictConfig,
                        normalizer: Optional[normalize.Normalizer] = None,
                        normalizer_kwargs: Optional[dict] = None,
                        naive_compression_config: DictConfig = None,
                        **kwargs
                            ):
        

            
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )
        
        self.naive_compression_module = utils.blank_init(
            naive_compression_config,
            n_in = self.original_weight.shape[1],
            n_out = self.original_weight.shape[0],
            dtype=self.original_weight.dtype,
            device=self.original_weight.device,
        )
        
        if isinstance(block_diagonal_config.block_size, int):
            block_diagonal_config.block_size = (block_diagonal_config.block_size, block_diagonal_config.block_size)
        self.A = BlockwiseDiagMatrix(
            d=self.original_weight.shape[0],
            block_size=block_diagonal_config.block_size[0],
            initalize_as_identity=True,
            device=self.original_weight.device,
        )
        self.B = BlockwiseDiagMatrix(
            d=self.original_weight.shape[1],
            block_size=block_diagonal_config.block_size[1],
            initalize_as_identity=True,
            device=self.original_weight.device,
        )
        
        
            
            
        self.to(self.original_weight.device)
        
    
        
        self.compressed = True
        
    def get_additional_bits(self):
        
        additional_bits = self.A.get_n_bits() + self.B.get_n_bits()
        
        return additional_bits
    
    def get_n_bits(self):
        n_bits = self.naive_compression_module.get_n_bits() 
        n_bits += self.get_additional_bits() #no need to consider the normalizer as we can fold it into A and B
        return n_bits
    
    def get_n_nonzero(self):
        assert hasattr(self.naive_compression_module, "get_n_nonzero"), "Naive compression module does not have get_n_nonzero method"
        n_nonzero = self.naive_compression_module.get_n_nonzero()
        n_nonzero += (self.get_additional_bits())//16
        return n_nonzero
        
    @property
    def compression_measure(self):
        return self.naive_compression_module.compression_measure
        
        
        
#testing main fn 
if __name__ == "__main__":
    @hydra.main(config_path="../config/compress", config_name="ARMOR")
    def testing_main(cfg: DictConfig):
        utils.seed(0)
        device = "cuda:0"
        print("current_directory:", os.getcwd())
        model_name = "Qwen/Qwen2.5-7B"
        # weight_path = "/data/lliu/NoWAG/models/meta-llama/Llama-2-7b-hf/original_weights/layer_28/mlp.up_proj.pt"
        proj_name = "layer_0/self_attn.q_proj"
        weight_path = f"/data/lliu/LLM_data/{model_name}/original_weights/{proj_name}.pt"
        hessian_diag = weight_path.replace("original_weights", "hessian_diag/SlimPajama-627B/n_samples_128_ctx_len_8192/seed_0")

        
        weight = torch.load(weight_path, map_location=device)["weight"].to(torch.float32).detach()
        hessian_diag = torch.load(hessian_diag, map_location=device)["hessianDiag"].to(torch.float32 )
        
        #take the mean of all the finite valid hessian diag values
        with torch.no_grad():
            valid = torch.isfinite(hessian_diag) & (hessian_diag > 0)
            print("non-valid hessian diag values:", (~valid).sum().item(), "out of", hessian_diag.numel())
            mean_hessian_diag = hessian_diag[valid].mean().item()
            hessian_diag[~valid] = mean_hessian_diag
            hessian_diag = hessian_diag.detach()
        
        
        # assert torch.all(torch.isfinite(hessian_diag)), "Hessian diag contains non-finite or non-positive values"
        print("weight:", weight)
        print("hessian_diag:", hessian_diag)
        # weight = weight[:,:2048]
        # hessian_diag = hessian_diag[:2048]
        print("weight shape:", weight.shape)
        print("hessian_diag shape:", hessian_diag.shape)
        # raise ValueError("stop here")
        
        # print("created compression module gpu stats:")
        # print(utils.get_gpu_memory(weight.device))
        print("cfg:")
        #print out the cfg
        print(OmegaConf.to_yaml(cfg))
        # raise ValueError("stop here, we are done with training")
        compression_module = instantiate(
            cfg.init_config,
            weight = weight)
        compression_module.hessianDiag = hessian_diag
        # raise ValueError("stop here")
        torch.set_printoptions(linewidth = 240)
        compression_module.compress(
            training_config_overrides = {"iter_save_path": "test/test_run/permute_{iter}.pt",
                                         "logfile": "test/test_run/log.txt",
                                         "iter_save_freq": 1000},
            **cfg.compression_config
        )
        # torch.save({"A": compression_module.A.get_dense().to("cpu"),
        #             "B": compression_module.B.get_dense().to("cpu")},
        #         "/data/lliu/PermPrune/test/permute_compress_A_B.pt")
        # raise ValueError("stop here, we are done with training")
        
        
        torch.set_printoptions(sci_mode=False)
        
        #run some checks:
        print("reconstructued_weight:", compression_module.reconstruct(denormalize=True))
        print("original_weight:", weight)
        
        #create a random input 
        x = torch.randn(1, weight.shape[1]).to(device)
        
        #try several different forward pass methods
        y_naive = compression_module(x)
        
        
        compression_module.forward_method = "otf"
        compression_module.denormalization_method = "otf"
        y_otf = compression_module(x)
        print("y_naive:", y_naive)
        print("y_otf:", y_otf)
        print("maximum difference:", torch.max(torch.abs(y_naive - y_otf)))
        assert torch.allclose(y_naive, y_otf, atol=1e-5), "Naive and otf forward pass do not match"
        
        #try the blank recreate method
        
        state_dict = compression_module.state_dict()    
        
        new_compression_module = ARMOR_Linear(weight)
        
        new_compression_module.blank_recreate(
            **cfg.compression_config)
        
        new_compression_module.load_state_dict(state_dict)
        
        assert torch.allclose(
            new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
        ), "Weight does not match"
        
        y_blank_recreate = new_compression_module(x)
        assert torch.allclose(y_naive, y_blank_recreate, atol=1e-5), "Naive and blank recreate forward pass do not match"
        
        y_orig = F.linear(x, weight)
        print("y_orig:", y_orig)
        
        #test out load_iter_state 
        new_compression_module = ARMOR_Linear(weight)
        new_compression_module.blank_recreate(
            **cfg.compression_config)
        
        print(f"Path: test/test_run/permute_{cfg.compression_config.training_config.n_iters-1}.pt")
        new_compression_module.load_iter_state(
            f"test/test_run/permute_{cfg.compression_config.training_config.n_iters-1}.pt")
        
        new_compression_module.load_state_dict(state_dict)
        
        assert torch.allclose(
            new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
        ), "Weight does not match"
        
        y_blank_recreate = new_compression_module(x)
        assert torch.allclose(y_naive, y_blank_recreate, atol=1e-5), "Naive and blank recreate forward pass do not match"
        
        y_orig = F.linear(x, weight)
        print("y_orig:", y_orig)
        
        if compression_module.compression_measure == "bits":
            print("average number  of bits", compression_module.get_n_bits()/compression_module.get_n_original_parameters())
        elif compression_module.compression_measure == "parameters":
            print("relative sparsity:", compression_module.get_n_nonzero()/compression_module.get_n_original_parameters())
        

    
    testing_main()