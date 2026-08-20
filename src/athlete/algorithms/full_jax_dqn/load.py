from athlete.algorithms.full_jax_dqn.interface import Agent
from athlete.algorithms.full_jax_dqn.jax_interface import JaxAgent

# TODO put this in the top level after restructuring such that save and load can directly be called with athlete.save...
# make the eval agents somehow smartly using the same route so no extra function needed, they are also just agents after all.

load_agent_from_file = Agent.load_from_file
load_jax_agent_from_file = JaxAgent.load_from_file
