from typing import Callable, Dict, Tuple

import chex
import flax
import jax

# TODO maybe make it so that instead of having eval step and eval reset step, the main agent can generate an evaluation
# instance that than also only has step and reset step


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
        [flax.struct.PyTreeNode, flax.struct.PyTreeNode], "JaxEvaluationAgent"
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
