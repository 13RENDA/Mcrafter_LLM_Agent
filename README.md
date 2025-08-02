# Mcrafter LLM Agent

A multi-agent simulation system built on the Crafter environment, featuring AI agents with memory systems, knowledge graphs, and advanced decision-making capabilities.

## 🎯 Overview

This project implements a sophisticated multi-agent system where AI agents collaborate to achieve goals in a Minecraft-inspired 2D environment. The system includes:

- **Multi-Agent Simulation**: Multiple AI agents working together
- **Memory System**: Agents with persistent memory and knowledge graphs
- **Advanced GUI**: Multi-view interface for observing agent behavior
- **Resource Management**: Complex crafting and resource sharing mechanics
- **LLM Integration**: AI-powered decision making and planning

## 🚀 Quick Start

### Environment Setup

1. **Install Dependencies**
   ```bash
   # Create conda environment
   conda env create -f environment.yml
   
   # Activate environment
   conda activate mcrafter
   ```

2. **Basic Simulation**
   ```bash
   # Run with 6 AI agents (default)
   python run.py --agent_num 6
   
   # Run with custom parameters
   python run.py --agent_num 3 --step_num 200
   ```

3. **GUI Interface**
   ```bash
   # Launch multi-view GUI
   python run_gui_multi_view.py --agent_num 4 --step_num 150
   ```

## 🎮 Multi-View GUI Usage

The `run_gui_multi_view.py` provides an advanced graphical interface for observing multi-agent simulations.

### Basic Commands

```bash
# Start simulation with 3 agents
python run_gui_multi_view.py --agent_num 3 --step_num 100

# Adjust simulation speed
python run_gui_multi_view.py --fps 10 --step_num 200

# Custom agent count
python run_gui_multi_view.py --agent_num 6 --step_num 350
```

### GUI Controls

#### Keyboard Controls
- **`v`**: Switch view mode (Single ↔ Multi ↔ Overview)
- **`l`**: Switch multi-view layout (Horizontal ↔ Vertical ↔ Grid)
- **`0-9`**: Follow specific agent in single view mode
- **`p`**: Pause/Resume simulation
- **`ESC`**: Exit GUI

#### View Modes
1. **Single View**: Follow one agent's perspective
2. **Multi View**: Display all agents simultaneously
3. **Overview**: Show overall environment state

#### Layout Options (Multi View)
- **Horizontal**: Views arranged side by side
- **Vertical**: Views stacked vertically
- **Grid**: Views in a grid arrangement

### Interface Features

- **Real-time Agent Status**: Monitor all agents' health, inventory, and current actions
- **Progress Tracking**: Visual progress bar showing simulation completion
- **Agent Information**: Detailed status for each agent including:
  - Agent type and ID
  - Current operation
  - Resource inventory
  - Health and energy levels
- **Interactive Controls**: Click buttons to switch views and control simulation

## 🏗️ System Architecture

### Core Components

```
memory_system/
├── agent.py              # Agent base class and AI logic
├── memory_system.py      # Memory management system
├── knowledge_graph.py    # Knowledge graph implementation
├── llm_api.py           # LLM integration
└── backbone_models/     # Vision and processing models

crafter/
├── env.py               # Crafter environment wrapper
├── engine.py            # Game engine
└── objects.py           # Game objects and mechanics

utils.py                 # Core simulation utilities
run.py                   # Main simulation runner
run_gui_multi_view.py    # Multi-view GUI interface
```

### Agent Capabilities

- **Resource Collection**: Wood, stone, coal, iron, diamond
- **Crafting**: Tools, weapons, and structures
- **Collaboration**: Resource sharing between agents
- **Planning**: Strategic decision making using LLM
- **Memory**: Persistent knowledge and experience

## 🎯 Game Objectives

The primary goal is for agents to collaborate to collect a diamond. This requires:

1. **Resource Gathering**: Collect basic materials (wood, stone)
2. **Tool Crafting**: Create pickaxes of increasing quality
3. **Advanced Mining**: Mine iron and coal for diamond tools
4. **Collaboration**: Share resources to optimize team performance

## 📊 Analysis and Visualization

### Results Processing
```bash
# Generate plots from simulation results
python plot.py

# Create videos from simulation data
python make_video_from_results.py
```

### Data Output
- **CSV Files**: Detailed agent statistics and performance metrics
- **PNG Images**: Screenshots of simulation states
- **Video Files**: Animated simulation recordings

## 🔧 Configuration

### Simulation Parameters

- **`--agent_num`**: Number of agents (default: 6)
- **`--step_num`**: Maximum simulation steps (default: 350)
- **`--fps`**: GUI refresh rate (default: 5)

### Environment Settings

- **World Size**: Configurable environment dimensions
- **Resource Distribution**: Procedurally generated terrain
- **Agent Starting Positions**: Distributed across the world

## 📁 Project Structure

```
Mcrafter_LLM_Agent/
├── crafter/              # Game environment
├── memory_system/        # AI agent systems
├── GUI/                  # GUI implementations
├── results/              # Simulation outputs
├── descriptions/         # System descriptions
├── icons/               # UI assets
├── run.py               # Main simulation
├── run_gui_multi_view.py # Multi-view GUI
├── utils.py             # Core utilities
└── environment.yml      # Dependencies
```

## 🎮 Game Mechanics

### Available Actions
- **Movement**: Up, down, left, right
- **Interaction**: Collect resources, use tools
- **Crafting**: Create tools and structures
- **Sleep**: Restore energy

### Resources and Tools
- **Basic Resources**: Wood, stone, coal, iron, diamond
- **Tools**: Wood/stone/iron pickaxes and swords
- **Structures**: Workbench, furnace, plants

### Survival Mechanics
- **Health**: Affected by hunger, thirst, and damage
- **Energy**: Consumed by actions, restored by sleeping
- **Hunger/Thirst**: Must be managed for survival

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the multi-agent simulation research. Please refer to the individual component licenses for specific terms.

## 📚 Documentation

- **Game Guide**: See `CRAFTER_GAME_GUIDE.md` for detailed game mechanics
- **GUI Guide**: See `GUI/MULTI_VIEW_GUI_README.md` for GUI usage
- **Examples**: Check `example_human_agent_usage.py` for usage examples

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure conda environment is activated
   ```bash
   conda activate mcrafter
   ```

2. **GUI Not Starting**: Check pygame installation
   ```bash
   pip install pygame
   ```

3. **Simulation Hangs**: Check agent configuration and step limits

### Performance Tips

- Reduce FPS for faster simulation: `--fps 1`
- Use fewer agents for testing: `--agent_num 2`
- Limit steps for quick runs: `--step_num 50`
