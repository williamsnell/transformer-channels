import torch as t
from pathlib import Path
from typing import Tuple, Literal
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import ticker
import einops
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer
from jaxtyping import Float

device = "cuda" if t.cuda.is_available() else "cpu"

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = HookedTransformer.from_pretrained(model_name)

def qr_svd(out_mat: Float[Tensor, "d_head d_model"], 
           in_mat: Float[Tensor, "d_model d_head"]
           ) -> Tuple[Float[Tensor, "d_head"], 
                      Float[Tensor, "d_head"],
                      Float[Tensor, "d_model d_head"],
                      Float[Tensor, "d_model d_head"],
                      Float[Tensor, "d_head d_head"],
                      Float[Tensor, "d_head d_head"]]:
    """Usage:
        out_mat is W_O
        in_mat can be W_Q, W_K, or W_V

       Purpose:

           Calculate the principal vectors of the subspaces
           spanned by out_mat and in_mat. 

           Also return the angle between each set of principal
           vectors (aka the principal angles, theta). 
           theta[0] <= ... <= theta[-1]

       Returns:
           (theta, cos(theta), principal_vectors_out_mat, principal_vectors_out_mat

       Follows the procedure in https://helper.ipam.ucla.edu/publications/glws1/glws1_15465.pdf

       Assumptions:
           Assumes the first n columns in a
           m x n (m >= n) matrix are linearly
           independent.

           E.g. in W_O.T, shape [768, 64],
           the first 64 columns should be 
           linearly independent.
    """
    # Q is the set of orthonormal basis vectors
    # for the subspace spanned by each matrix.

    q_out, r_out = t.linalg.qr(out_mat.transpose(-1, -2))
    q_in, r_in = t.linalg.qr(in_mat)

    # Compute the deviation between the 
    # two subspaces using SVD

    U, S, Vh = t.linalg.svd(q_out.transpose(-1, -2) @ q_in)

    principal_vectors_out_mat = q_out @ U
    principal_vectors_in_mat = q_in @ Vh.transpose(-1, -2)

    theta = t.arccos(S)

    return t.arccos(S), S, principal_vectors_out_mat, principal_vectors_in_mat, U, Vh



def calculate_all_connections(model: HookedTransformer, head_layer: int = 0):
    output = t.zeros(model.cfg.n_heads,
                     model.cfg.d_head, 
                     4,
                     model.cfg.n_layers - 1 - head_layer,
                     model.cfg.n_heads)

    # I use tx and rx for transmit and receive, respectively
    for tx_head in range(model.cfg.n_heads):
        tx = model.W_O[head_layer, tx_head].to(device)
        q_tx, r_tx = t.linalg.qr(tx.T)

        for layer in range(1 + head_layer, model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                for i, rx in enumerate((model.W_Q[layer, head].to(device), 
                           model.W_K[layer, head].to(device),
                           model.W_V[layer, head].to(device),
                           model.W_O[layer, head].to(device).T,
                           )):
                   q_rx, r_rx = t.linalg.qr(rx)
                   U, S, Vh = t.linalg.svd(q_tx.transpose(-2, -1) @ q_rx)

                   # S is cosine similarity
                   output[tx_head, :, i, layer - 1 - head_layer, head] = S

    
    return output

def plot_all_connections(output, starting_layer, clip=0.9):
    plottable = einops.rearrange(output,
        "tx_head d_head qkv layer rx_head -> qkv (layer rx_head) (tx_head d_head)")

    plottable[plottable < clip] = 0

    n_heads = output.shape[4]
    d_head = output.shape[1]


    for (i, comp_type) in enumerate(("Q", "K", "V", "O")):
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        fig.suptitle(f"Transmissions from Layer {starting_layer}")
        ax.set_title(f"{comp_type}-composition")
        map = ax.imshow((plottable[i]).detach().cpu(), vmin=0, vmax=1)
        fig.colorbar(map, ax=ax)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: 
            f"{(x // n_heads + starting_layer + 1):.0f}, {(x % n_heads):.0f})"
        ))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: f"{(x // d_head):.0f}"
        ))

        ax.set_xlabel("Transmitting Head")
        ax.set_ylabel("Receiving Head")

        plt.tight_layout()
        dir = Path(f"figures/{model_name}").mkdir(exist_ok=True)
        plt.savefig(f"figures/{model_name}/{comp_type}_layer_{starting_layer}.png")

    # plt.show()


def calc_incoming_connections(model: HookedTransformer, layer: int = 0, head: int = 0, type: Literal["Q", "K", "V", "O"] = "Q"):
    output = t.zeros(model.cfg.n_layers,
                     model.cfg.n_heads,
                     4,
                     model.d_head)

    # I use tx and rx for transmit and receive, respectively
    for tx_head in range(model.cfg.n_heads):
        for tx_layer in range(1 + layer, model.cfg.n_layers):
            tx = model.W_O[head_layer, tx_head].to(device)
            q_tx, r_tx = t.linalg.qr(tx.T)

            for i, rx in enumerate((model.W_Q[layer, head].to(device), 
                       model.W_K[layer, head].to(device),
                       model.W_V[layer, head].to(device),
                       model.W_O[layer, head].to(device).T,
                       )):
               q_rx, r_rx = t.linalg.qr(rx)
               U, S, Vh = t.linalg.svd(q_tx.transpose(-2, -1) @ q_rx)

               # S is cosine similarity
               output[tx_head, :, i, layer - 1 - head_layer, head] = S

    
    return output


def plot_incoming_connections(output, clip=0.9):
    pass


if __name__ == '__main__':
    # heads we think compose

    # (2.2, 4.11) are previous-token heads.
    # These should compose with (5.5, 0.9),
    # and to a lesser extent (5.8, 5.9),
    # which are induction heads

    # heads we don't think compose?
    #     hard to say, but we could
    #     benchmark against two random
    #     matrices perhaps?
    
#    output = calculate_all_connections(model)
#    plot_all_connections(output)
    

    # calculate the expected "background"
    # correlation
#     batch_size = 2
#     _, s_background, _, _ = qr_svd(t.randn(batch_size, 64, 768), t.randn(batch_size, 768, 64))
# 
#     plt.plot(s_background.mean(0).detach().cpu())
#     plt.show()

#
    for starting_layer in range(11):
        output = calculate_all_connections(model, starting_layer)
        plot_all_connections(output, starting_layer, clip=0.0)
#

    # ------- Explore the anomalously high 4 output dims
    #         of head (0,9) ---------------------------
#    [plt.plot(qr_svd(model.W_O[0,9], model.W_K[3, i])[1].detach().cpu(), label=i) for i in range(12)]
#    plt.legend()
#    plt.show()

    # compare the vector spaces spanned by the 4 vectors
    # to see if they overlap
    
    # all the heads in layer 3 except for (3, 0) and (3, 4) 
    # compose with (0, 9)

    _, S, out_mat, in_mat, U, Vh = qr_svd(model.W_O[0,9], 
                                   model.W_K[3])


    # First, see if W_O is writing to the same space
    # for all the heads

    # Should be all close to 1 if they're the same
    # sub-space between heads
    print("Should be close to 1 if the two heads are "
          "reading from the same subspace.")
    print(qr_svd(out_mat[1, :, :4].T, out_mat[2, :, :4])[1])

    # Shouldn't be all close to 1, based on head (3, 0)
    # not reading from the same subspace as head (0, 9)
    # transmitted to
    print("Shouldn't be close to 1 if the two heads aren't"
          " sharing a subspace.")
    print(qr_svd(out_mat[0, :, :4].T, out_mat[2, :, :4])[1])


    # Does anyone else transmit to this subspace?

    # We saw a similar pattern of transmission from 
    # other layer 0 heads, so let's see if they share the 
    # same subspace as (0, 9)

    # Yes, looks like some layer 1 heads transmit into it


    # Does this subspace get activated on particular
    # kinds of information?
    #       Get the channel vector:
    subspace = out_mat[1, :, :4]

    # Positional encodings?

    #   Run the same token through
    #   the model in 100 positions,
    logits, cache = model.run_with_cache("".join(["the"]*99))
    #   and see whether the information
    #   in the channel changes
    
    residual_stream = cache["blocks.3.hook_resid_pre"]

    def inspect_stream(string, subspace_vecs, resid_layer):
        logits, cache = model.run_with_cache(string)

        residual_stream = cache[f"blocks.{resid_layer}.hook_resid_pre"]

        projected = residual_stream[0] @ subspace_vecs

        return projected 


