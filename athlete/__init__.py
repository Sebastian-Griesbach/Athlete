from athlete.agent import Agent
from athlete.algorithms.registry import AlgorithmRegistry

# This is done such that we can call these functions directly with athlete.<function>
make = Agent.make
from_checkpoint = Agent.from_checkpoint
get_default_configuration = AlgorithmRegistry.get_default_configuration
register = AlgorithmRegistry.register
list_algorithms = AlgorithmRegistry.list_algorithms

# Import all algorithms to register them, registration happens in according __init__.py files
import athlete.algorithms.q_learning
import athlete.algorithms.dqn
import athlete.algorithms.ddpg
import athlete.algorithms.td3
import athlete.algorithms.sac
import athlete.algorithms.ppo
