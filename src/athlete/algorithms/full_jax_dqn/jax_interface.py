from typing import Callable, Dict, Tuple

import chex
import flax
import jax

# TODO think about auto reset modes similar to gymnasium, agent gets two observations and immediately
# calls step for final observation and reset step for initial observation, how exactly?


@chex.dataclass(frozen=True)
class JaxAgent:
    step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    reset_step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    make_evaluation_agent: Callable[
        [flax.struct.PyTreeNode, flax.struct.PyTreeNode],
        Tuple[flax.struct.PyTreeNode, "JaxEvaluationAgent"],
    ]


@chex.dataclass(frozen=True)
class JaxEvaluationAgent:
    step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    reset_step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
