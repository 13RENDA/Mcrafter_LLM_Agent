# HumanAgent Integration

This document describes how to use HumanAgent in the multi-agent simulation system.

## Overview

HumanAgent allows human users to participate in the multi-agent simulation alongside AI agents. Human agents can manually select their actions and collaborate with AI agents to achieve the common goal of collecting a diamond.

## Features

- **Manual Action Selection**: Human agents can choose their operations, resources to collect/share, and target agents
- **Collaboration with AI**: Human agents can work together with AI agents through resource sharing
- **Flexible Configuration**: Support for multiple human agents and various simulation parameters
- **Real-time Interaction**: Human agents provide input during the simulation

## Quick Start

### Basic Usage

```bash
# Make agent 0 human-controlled with 3 total agents
python run.py --human_agents 0 --agent_num 3

# Make agents 0 and 2 human-controlled with 4 total agents
python run.py --human_agents 0,2 --agent_num 4

# Run with all AI agents (default)
python run.py --agent_num 6
```

### Advanced Usage

```bash
# Custom simulation parameters
python run.py --human_agents 1 --agent_num 3 --step_num 200 --num_of_turn 2

# Short simulation for testing
python run.py --human_agents 0 --agent_num 3 --step_num 50
```

## Available Operations

Human agents can choose from the following operations:

- **`Navigator`**: Navigate to collect resources
- **`share`**: Share resources with other agents
- **`noop`**: Do nothing
- **`move_left`**, **`move_right`**, **`move_up`**, **`move_down`**: Movement
- **`do`**: Perform action (like chopping wood, mining)
- **`sleep`**: Rest to recover energy

## Available Resources

- **`wood`**: Wood resource
- **`stone`**: Stone resource
- **`coal`**: Coal resource
- **`iron`**: Iron resource
- **`diamond`**: Diamond resource
- **`not_applicable`**: When no resource is needed

## Target Agent IDs

- Use **`-1`** for no target (when not sharing)
- Use **agent ID** (0, 1, 2, etc.) when sharing resources

## Game Objective

Work together with AI agents to collect a diamond as quickly as possible! Only one agent needs to collect the diamond for the team to win.

## Collaboration Strategy

1. **Resource Sharing**: Use the `share` operation to give resources to other agents
2. **Role Assignment**: AI agents will help the previous agent in the chain
3. **Communication**: Monitor the game state and adapt your strategy
4. **Teamwork**: Focus on the common goal rather than individual achievements

## Technical Details

### Integration Points

The HumanAgent integration modifies several key components:

1. **Agent Initialization** (`run.py`):
   - Parses `--human_agents` parameter
   - Creates HumanAgent instances for specified IDs

2. **Action Processing** (`utils.py`):
   - Handles HumanAgent actions differently from AI agents
   - Processes human input during simulation

3. **Thinking Process** (`memory_system/agent.py`):
   - Bypasses LLM processing for HumanAgent
   - Uses manual input instead of AI-generated responses

### Code Structure

```
memory_system/agent.py
├── Agent (base class)
└── HumanAgent (subclass)
    ├── choose_action() - Manual action selection
    ├── communicate() - Send messages to other agents
    └── receive_message() - Receive messages from other agents

run.py
├── initialize_agents() - Creates HumanAgent instances
└── main() - Parses command line arguments

utils.py
├── AgentThinkingProcessor - Handles human vs AI thinking
└── SimulationLogger - Reports human agent actions
```

## Testing

Run the test suite to verify HumanAgent integration:

```bash
# Run simplified tests (no dependencies)
python test_human_agent_simple.py

# Run full tests (requires environment setup)
python test_human_agent.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct environment
   ```bash
   conda activate mcrafter
   # or
   source mcrafter/bin/activate
   ```

2. **Simulation Hangs**: Check that all human agents have provided input

3. **Invalid Input**: Use the provided operation and resource names exactly

### Environment Setup

The project uses a conda environment defined in `environment.yml`:

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate mcrafter
```

## Examples

### Example 1: Single Human Agent

```bash
python run.py --human_agents 0 --agent_num 3 --step_num 100
```

This creates a simulation with:
- 3 total agents
- Agent 0 is human-controlled
- Agents 1 and 2 are AI-controlled
- 100 simulation steps

### Example 2: Multiple Human Agents

```bash
python run.py --human_agents 0,2 --agent_num 4 --step_num 150
```

This creates a simulation with:
- 4 total agents
- Agents 0 and 2 are human-controlled
- Agents 1 and 3 are AI-controlled
- 150 simulation steps

### Example 3: All AI Agents

```bash
python run.py --agent_num 6 --step_num 350
```

This creates a simulation with:
- 6 AI agents
- No human control
- 350 simulation steps

## Contributing

To extend HumanAgent functionality:

1. Modify `HumanAgent` class in `memory_system/agent.py`
2. Update action processing in `utils.py`
3. Add tests in `test_human_agent_simple.py`
4. Update documentation

## License

This HumanAgent integration is part of the multi-agent simulation project. 