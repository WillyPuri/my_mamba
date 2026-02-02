import torch
import torch.nn as nn
import transformers
from transformers.models.mamba.modeling_mamba import MambaMixer, MambaCache
if torch.cuda.is_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    import transformers.models.mamba.modeling_mamba as mm

    mamba_inner_fn = mm.mamba_inner_fn
    selective_scan_fn = mm.selective_scan_fn
    selective_state_update = mm.selective_state_update

from typing import Optional
import itertools
import os
import shutil

def patched_no_conv1d_cuda(
    self,
    hidden_states: torch.Tensor,
    cache_params: MambaCache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
):
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(hidden_states).transpose(1, 2)

    #--------------------------PARTE MODIFICATA------------------------
    use_conv1d = bool(getattr(self.config, "use_conv1d", True))
    conv_kernel = int(getattr(self.config, "conv_kernel", 0))  # assumed 2..4
    use_conv_bias = bool(getattr(self.config, "use_conv_bias", True))
    use_bias = bool(getattr(self.config, "use_bias", False))

    # Faccio diventare sempre falso questo if
    if self.training and cache_params is None and False:  # Doesn't support outputting the states -> used for training
        #contextualized_states = mamba_inner_fn(
        #    projected_states,
        #    self.conv1d.weight,
        #    self.conv1d.bias if self.use_conv_bias else None,
        #    self.x_proj.weight,
        #    self.dt_proj.weight,
        #    self.out_proj.weight,
        #    self.out_proj.bias.float() if self.use_bias else None,
        #    -torch.exp(self.A_log.float()),
        #    None,  # input-dependent B
        #    None,  # input-dependent C
        #    self.D.float(),
        #    delta_bias=self.dt_proj.bias.float(),
        #    delta_softplus=True,
        #)
        pass

    else:
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        #conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        #if cache_params is not None and cache_position[0] > 0:
        #    hidden_states = causal_conv1d_update(
        #        hidden_states.squeeze(-1),
        #        cache_params.conv_states[self.layer_idx],
        #        conv_weights,
        #        self.conv1d.bias,
        #        self.activation,
        #    )
        #    hidden_states = hidden_states.unsqueeze(-1)
        #else:
        #    if cache_params is not None:
        #        conv_states = nn.functional.pad(
        #            hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
        #        )
        #        cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
        #    hidden_states = causal_conv1d_fn(
        #        hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
        #    )

        # Lascio comunque la funzione di attivazione
        hidden_states = self.act(hidden_states)                                     # Ho controllato e type(self.act)=SiLUActivation()

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)
        #--------------------------PARTE MODIFICATA------------------------

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        if cache_params is not None and cache_position[0] > 0:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(self.layer_idx, ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
    return contextualized_states

token_id = -1
temp = None
save_dir = "/content/M_matrices/"

def patched_slow_save(self, input_states, cache_params: Optional[MambaCache]=None, cache_position:Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            # use `cache_position.shape[0]` to check whether we are in prefill
            # stage, it's equivalent to check `cache_position[0] == 0`, which
            # breaks dynamo fullgraph constraints
            if cache_position.shape[0] == self.conv_kernel_size:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )

                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
            else:
                conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
                conv_state = conv_state.to(self.conv1d.weight.device)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        if self.use_mambapy and self.training and cache_params is None:
            hs = pscan(discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)) # [batch, seq_len, intermediate_size, ssm_state_size]

            scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2) # [batch, intermediate_size, seq_len]
            scan_output = scan_output + hidden_states * self.D[None, :, None]
            scan_output = scan_output * self.act(gate)
        else:
            scan_outputs = []
            for i in range(seq_len):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediate_size, ssm_state]
                scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediate_size, 1]
                scan_outputs.append(scan_output[:, :, 0])
            scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, intermediate_size, seq_len]
            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = (scan_output * self.act(gate))

            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        #--------------------------PARTE MODIFICATA------------------------
        # Contatori per salvare la matrice con un nome sensato
        layer_id = getattr(self, "layer_idx", 0)
        if layer_id == 0:
          global token_id
          token_id+=1

        # Salvo le matrici discrete_A, discrete_B e C per un layer fissato

        save_sub_dir_A = os.path.join(save_dir, f"A_bar/layer{layer_id}/")
        save_sub_dir_B = os.path.join(save_dir, f"B_bar/layer{layer_id}/")
        save_sub_dir_C = os.path.join(save_dir, f"C/layer{layer_id}/")
        save_sub_dir_D = os.path.join(save_dir, f"D/layer{layer_id}/")

        os.makedirs(save_sub_dir_A, exist_ok=True)
        os.makedirs(save_sub_dir_B, exist_ok=True)
        os.makedirs(save_sub_dir_C, exist_ok=True)
        os.makedirs(save_sub_dir_D, exist_ok=True)

        file_path_A = os.path.join(save_sub_dir_A, f"M_{token_id}.pt")
        file_path_B = os.path.join(save_sub_dir_B, f"M_{token_id}.pt")
        file_path_C = os.path.join(save_sub_dir_C, f"M_{token_id}.pt")
        file_path_D = os.path.join(save_sub_dir_D, f"M_{token_id}.pt")

        torch.save(discrete_A, file_path_A)
        torch.save(discrete_B, file_path_B)
        torch.save(C, file_path_C)
        torch.save(self.D, file_path_D)                 # self.D.size() = torch.Size([intermediate_size])

        print(file_path_A)
        print(file_path_B)
        print(file_path_C)
        print(file_path_D)

        #--------------------------PARTE MODIFICATA------------------------

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states
