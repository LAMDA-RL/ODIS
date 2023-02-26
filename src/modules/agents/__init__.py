REGISTRY = {}

# normal agents
from .rnn_agent import RNNAgent

REGISTRY["rnn"] = RNNAgent
