from typing import Tuple

import flax
import optax


def target_network_update(
    target_network_variables: flax.core.FrozenDict,
    q_value_function_variables: flax.core.FrozenDict,
    tau: float = 1.0,
    update_collections: Tuple[str, ...] = ("params",),
) -> flax.core.FrozenDict:
    updated_target_variables = target_network_variables

    # For loop is unrolled once during jitting, as long as update collection is static, has no performance impact
    for collection_name in update_collections:
        updated_collection = optax.incremental_update(
            new_tensors=q_value_function_variables[collection_name],
            old_tensors=target_network_variables[collection_name],
            step_size=tau,
        )

        updated_target_variables = updated_target_variables.copy(
            {
                collection_name: updated_collection,
            }
        )

    return updated_target_variables
