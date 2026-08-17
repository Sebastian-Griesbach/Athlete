import flax
import optax


def target_network_update(
    target_network_variables: flax.core.FrozenDict,
    q_value_function_variables: flax.core.FrozenDict,
    tau: float = 1.0,
) -> flax.core.FrozenDict:
    target_network_variables = optax.incremental_update(
        new_tensors=q_value_function_variables,
        old_tensors=target_network_variables,
        step_size=tau,
    )
    return target_network_variables
