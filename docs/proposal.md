# Project Proposal: HyperZero

## Title

HyperZero: AlphaZero-Style Self-Play for N-Dimensional Connect-K

## Summary

HyperZero explores reinforcement learning in a generalized family of Connect-K games played on N-dimensional boards with gravity. The project will implement a configurable game environment, baseline agents, and an AlphaZero-style training system using neural-guided Monte Carlo Tree Search. The central research goal is to understand how search, learning, and model architecture scale as classic 2D Connect-style games are extended into 3D, 4D, and potentially higher dimensions.

The project is designed to be valuable even if the hardest 4D setting proves computationally expensive. The main contribution is a rigorous experimental platform and a set of controlled results about what makes higher-dimensional self-play tractable.

## Motivation

AlphaZero demonstrated that self-play reinforcement learning can produce strong agents in perfect-information games without handcrafted strategy. Most well-known examples use games with established rule sets and rich human history, such as Go, chess, and shogi. HyperZero instead studies a clean game family whose complexity can be tuned directly through dimensionality, board size, connect length, and gravity-axis geometry.

N-dimensional Connect-K is attractive because it is simple to define, easy to visualize at small sizes, and combinatorially rich. Moving from 2D to 3D and 4D increases the number of cells, possible lines, tactical threats, and branching factors. This creates a natural testbed for studying how AlphaZero-style methods scale.

## Problem Statement

Classic Connect Four is well-studied, but higher-dimensional Connect-K variants introduce larger state spaces and less obvious strategic structure. It is unclear how far a practical AlphaZero-style implementation can scale in these variants, what architecture is best suited to N-dimensional boards, and whether techniques such as curriculum learning or symmetry augmentation improve sample efficiency.

This project asks:

> Can neural-guided MCTS learn strong play in N-dimensional Connect-K, and what design choices make training feasible as dimensionality increases?

## Research Questions

1. How does AlphaZero-style self-play scale from 2D to 3D and 4D Connect-K?
2. How much MCTS search is needed before the neural network provides useful policy guidance?
3. Which neural architecture is most effective for configurable N-dimensional boards?
4. Does curriculum learning from smaller or lower-dimensional boards improve training on harder variants?
5. Do board symmetries improve sample efficiency and final playing strength?
6. How do gravity-axis choices and board shapes affect branching factor, game length, and learnability?

## Methodology

The project will proceed in four layers:

1. Build a general N-dimensional Connect-K game engine.
2. Implement non-neural baselines, including random play, tactical heuristics, and pure MCTS.
3. Implement AlphaZero-style self-play with a policy-value network and PUCT-based MCTS.
4. Run controlled experiments across board dimensions, rulesets, architectures, and training settings.

## Initial Scope

The minimum successful project is not to solve every possible variant. It is to produce a working experimental system and demonstrate meaningful learning on a nontrivial higher-dimensional game.

Primary success target:

> A trained AlphaZero-style agent beats random, heuristic, and pure-MCTS baselines on 3D 4x4x4 Connect-4.

Stretch success target:

> The system runs on 4D 4x4x4x4 Connect-4 and produces useful analysis, even if full mastery is out of reach.

Current status:

- The 3D target has been met for 4x4x4 Connect-4: the guarded line-ResNet
  checkpoint beats random, tactical, heuristic, and MCTS-32 baselines under
  fixed final-eval budgets.
- The 4D stretch target has produced a clear result: the system runs and learns
  stably in 4D, but current specialist agents remain weak against tactical and
  heuristic threat/fork play.
- The next research extension is a universal agent: one shared checkpoint that
  can play selected 2D, 3D, and 4D Connect-K variants. The current
  residual-recovery checkpoint is promoted for the public demo and has passed
  train-time eval floors across the selected variants. This tests whether
  tactical concepts learned in cheaper dimensions can transfer into 4D.

## Expected Contributions

- A configurable N-dimensional Connect-K environment.
- Programmatic generation of legal moves and winning lines.
- Baseline agents for controlled comparison.
- AlphaZero-style self-play implementation.
- Experiments measuring scaling behavior across game variants.
- Documentation and visualizations suitable for a research-style final report.

## Risks

The largest risk is computational cost. AlphaZero-style systems can be expensive, and 4D variants may be too large for fast iteration. The mitigation is to build upward from small settings: 2D validation, 3D target, and 4D stretch analysis.

Another risk is implementation complexity. The project should prioritize a minimal complete training loop before adding advanced architectures, distributed workers, or polished interfaces.

## Proposed Abstract

This project explores AlphaZero-style reinforcement learning in a generalized family of N-dimensional Connect-K games with gravity. We implement a configurable game engine supporting arbitrary board dimensions, connect lengths, and gravity axes, then train neural-guided MCTS agents through self-play. Using 2D Connect Four as a validation benchmark, we scale to 3D and 4D variants to study how dimensionality affects search complexity, learning efficiency, and model architecture requirements. We compare pure MCTS, heuristic agents, and learned policy-value networks, and evaluate whether curriculum learning and symmetry augmentation improve training. The goal is to understand how well self-play reinforcement learning transfers from classic board games to higher-dimensional abstract strategy environments.
