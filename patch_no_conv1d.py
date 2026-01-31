import torch
import transformers
from transformers.models.mamba.modeling_mamba import MambaMixer, MambaCache
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import transformers.models.mamba.modeling_mamba as mm

mamba_inner_fn = mm.mamba_inner_fn
selective_scan_fn = mm.selective_scan_fn
selective_state_update = mm.selective_state_update

def patched_cuda_kernels_forward(
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
