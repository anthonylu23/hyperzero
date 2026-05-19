"""Agents for N-dimensional Connect-K."""

from hyperzero.agents.alphazero_agent import AlphaZeroAgent
from hyperzero.agents.base import Agent
from hyperzero.agents.heuristic_agent import HeuristicAgent
from hyperzero.agents.mcts_agent import MCTSAgent
from hyperzero.agents.random_agent import RandomAgent
from hyperzero.agents.tactical_agent import TacticalAgent

__all__ = [
    "Agent",
    "AlphaZeroAgent",
    "HeuristicAgent",
    "MCTSAgent",
    "RandomAgent",
    "TacticalAgent",
]
