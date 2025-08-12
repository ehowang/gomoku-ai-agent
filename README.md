# Gomoku MCTS AI Agent

An advanced AI agent for playing Gomoku (5-in-a-row) using Monte Carlo Tree Search (MCTS), inspired by AlphaGo Zero's approach.

## Overview

This project implements a sophisticated MCTS algorithm for Gomoku that combines:
- **Tree Search**: Systematically explores possible future game states
- **UCB Selection**: Balances exploration of new moves with exploitation of promising ones
- **Pattern Recognition**: Identifies common Gomoku patterns for better evaluation
- **Smart Rollouts**: Uses pattern-based move selection during simulations

## Features

### Core MCTS Algorithm
- **Selection**: Uses Upper Confidence Bound (UCB) formula to traverse the tree
- **Expansion**: Adds new nodes to explore untried moves
- **Simulation**: Runs intelligent rollouts with pattern recognition
- **Backpropagation**: Updates win/visit statistics up the tree

### Advanced Capabilities
- **Pattern Detection**: Recognizes formations like open-three, blocked-four, etc.
- **Defensive Play**: Automatically blocks opponent's winning threats
- **Time Management**: Configurable time limit per move
- **Strategic Evaluation**: Considers board position and move quality

## Files

- `mcts_agent.py` - Pure MCTS implementation without external dependencies
- `mcts_tool_agent.py` - Extended version with API service integration
- `demo_mcts.py` - Interactive demonstration of the MCTS algorithm
- `agent.json` - Agent configuration file
- `my_example.py` - Original LLM-based agent for comparison

## How MCTS Works

The Monte Carlo Tree Search algorithm builds a search tree of possible game states:

1. **Selection Phase**: Starting from the root, select child nodes using UCB formula:
   ```
   UCB = win_rate + c * sqrt(ln(parent_visits) / node_visits)
   ```
   This balances exploitation (choosing good moves) with exploration (trying new moves).

2. **Expansion Phase**: When reaching a leaf node, expand by adding a new child for an untried move.

3. **Simulation Phase**: From the new node, play out a simulated game using intelligent random moves.

4. **Backpropagation Phase**: Update all nodes along the path with the simulation result.

## Usage

### Basic Usage

```python
from mcts_agent import MCTSAgent
from gomoku.core.models import GameState, Player

# Create agent
agent = MCTSAgent()
agent.player = Player.BLACK

# Get move for current game state
move = await agent.get_move(game_state)
```

### Running the Demo

```bash
python demo_mcts.py
```

This will show:
- How MCTS analyzes a game position
- Pattern recognition in action
- Move selection process with statistics

### Configuration Options

You can customize the MCTS behavior:

```python
agent = MCTSAgent()
agent.time_limit = 10.0      # Seconds per move (default: 5.0)
agent.c_param = 2.0          # Exploration constant (default: 1.414)
agent.simulation_depth = 100  # Max rollout depth (default: 50)
```

## Pattern Recognition

The agent recognizes these Gomoku patterns:

| Pattern | Description | Score |
|---------|-------------|-------|
| Five | Five in a row (winning) | 100,000 |
| Open Four | Four in a row, both ends open | 10,000 |
| Blocked Four | Four in a row, one end blocked | 1,000 |
| Open Three | Three in a row, both ends open | 500 |
| Blocked Three | Three in a row, one end blocked | 100 |
| Open Two | Two in a row, both ends open | 50 |
| Blocked Two | Two in a row, one end blocked | 10 |

## Performance

The MCTS agent performance depends on:
- **Time limit**: More time = more simulations = better moves
- **Board complexity**: Performance scales with number of legal moves
- **Pattern recognition**: Quickly identifies critical moves

Typical performance:
- 5 seconds: ~5,000-10,000 simulations
- Finds optimal moves in tactical positions
- Strong defensive play

## Comparison with AlphaGo Zero

This implementation is inspired by AlphaGo Zero but adapted for Gomoku:

**Similarities:**
- MCTS with UCB selection
- Tree reuse between moves
- Robust move selection (most visited)

**Differences:**
- No neural network (uses pattern recognition instead)
- Simpler evaluation function
- No self-play training
- Single-threaded (no parallel simulations)

## Requirements

- Python 3.7+
- Gomoku framework (see requirements.txt)
- No GPU required

## Installation

```bash
pip install -r requirements.txt
```

## Future Improvements

Potential enhancements:
- Neural network integration for position evaluation
- Parallel tree search for faster performance
- Opening book for common starting patterns
- Endgame tablebase for perfect play
- Self-play training system
