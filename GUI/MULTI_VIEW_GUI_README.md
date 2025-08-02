# Multi-View GUI User Guide

## 🎯 Overview

Multi-View GUI is an enhanced GUI that supports multiple perspectives and view switching, allowing you to:

- **Single View Mode**: Follow a specific agent's perspective
- **Multi View Mode**: Display all agents' perspectives simultaneously
- **Overview Mode**: Show the overall environment view
- **View Switching**: Switch between followed agents in real-time
- **Layout Switching**: Switch layouts in multi-view mode (horizontal/vertical/grid)

## 🚀 Quick Start

### Basic Usage
```bash
# Start simulation with 3 AI agents
python run_gui_multi_view.py --agent_num 3 --step_num 100

# Start simulation with HumanAgents
python run_gui_multi_view.py --human_agents 0,2 --agent_num 4

# Adjust FPS and steps
python run_gui_multi_view.py --fps 10 --step_num 200
```

### Test GUI Functionality
```bash
python test_multi_view_gui.py
```

## 🎮 Controls

### Keyboard Controls
- **`v`**: Switch view mode (Single ↔ Multi ↔ Overview)
- **`l`**: Switch multi-view layout (Horizontal ↔ Vertical ↔ Grid)
- **`0-9`**: Follow specific agent in single view mode
- **`h`**: Start HumanAgent action selection
- **`p`**: Pause/Resume simulation
- **`ESC`**: Exit GUI

### Mouse Controls
- **Left click buttons**: Execute corresponding actions
- **Button hover**: Show button highlight effects

## 📊 View Modes Explained

### 1. Single View Mode
- Follows a specific agent's perspective
- Displays the agent's local view and inventory
- Suitable for observing individual agent behavior
- Can quickly switch between agents using number keys 0-9

### 2. Multi View Mode
- Displays all agents' perspectives simultaneously
- Supports three layouts:
  - **Horizontal Layout**: All views arranged horizontally
  - **Vertical Layout**: All views arranged vertically
  - **Grid Layout**: Views arranged in a grid

### 3. Overview Mode
- Shows the overall environment view
- Uses Agent 0's perspective as overview
- Suitable for observing overall environment state

## 🎨 Interface Elements

### Main Game Area
- **Single View**: Shows the followed agent's view
- **Multi View**: Shows all agents' views
- **Overview**: Shows environment overview

### Sidebar
- **Title**: Displays "Multi-View GUI"
- **View Info**: Current view mode and layout
- **Progress Bar**: Simulation progress
- **Agent Status**: All agents' types, status, and current operations
- **Status Indicators**: Pause, input mode, and other states

### Control Buttons
- **View Controls**: Single View, Multi View, Overview, Layout
- **Agent Follow**: Agent 0-9 buttons
- **Simulation Controls**: Pause, Human Action, Help

## 🔧 Advanced Features

### HumanAgent Integration
- Supports mixed HumanAgent and AI agent simulation
- HumanAgents displayed in yellow, AI agents in white
- Currently followed agent displayed in cyan
- Supports HumanAgent action selection interface

### Real-time Status Updates
- Agent status updates in real-time
- Operation progress displayed in real-time
- View switching takes effect immediately

### Custom Configuration
- Adjustable FPS (default: 5)
- Configurable simulation steps
- Specifiable HumanAgent IDs

## 📋 Use Cases

### Research Scenarios
1. **Behavior Analysis**: Use single view mode to observe specific agent behavior
2. **Comparative Analysis**: Use multi-view mode to compare different agent behaviors
3. **Environment Observation**: Use overview mode to observe overall environment changes

### Debugging Scenarios
1. **Problem Localization**: Quickly switch views to locate problematic agents
2. **Status Checking**: Check all agent status through sidebar
3. **Interaction Testing**: Use HumanAgent to test interaction functionality

### Demonstration Scenarios
1. **Multi-agent Showcase**: Use multi-view mode to showcase multi-agent collaboration
2. **Behavior Comparison**: Use grid layout to compare different agent strategies
3. **Environment Display**: Use overview mode to display environment complexity

## 🛠️ Technical Features

### Performance Optimization
- Uses Pygame for efficient rendering
- Supports configurable FPS
- Optimized multi-view rendering

### Memory Management
- Intelligent view caching
- Efficient image conversion
- Optimized UI rendering

### Extensibility
- Modular design
- Easy to add new view modes
- Supports custom layouts

## 🔍 Troubleshooting

### Common Issues

1. **GUI Black Screen**
   - Check if pygame is properly installed
   - Ensure environment initialization is successful
   - Check agent number settings

2. **View Switching Not Working**
   - Ensure agent ID is within valid range
   - Check environment status
   - Restart GUI

3. **HumanAgent Input Issues**
   - Ensure HumanAgent ID is correctly set
   - Check input mode status
   - Use ESC to cancel input

### Debugging Tips

1. **Use Test Script**
   ```bash
   python test_multi_view_gui.py
   ```

2. **Check Console Output**
   - View simulation status information
   - Check error messages
   - Monitor agent behavior

3. **Adjust FPS**
   - Lower FPS to observe details
   - Increase FPS to speed up simulation

## 📈 Performance Recommendations

### Optimization Settings
- **Low-end devices**: FPS=3, agent_num≤3
- **Mid-range devices**: FPS=5, agent_num≤5
- **High-end devices**: FPS=10, agent_num≤8

### Memory Usage
- Multi-view mode uses more memory
- Recommend agent count not exceeding 8
- For long-running sessions, restart periodically

## 🎯 Summary

Multi-View GUI provides powerful multi-perspective observation capabilities, supporting:

- ✅ Flexible view switching
- ✅ Multiple layout options
- ✅ HumanAgent integration
- ✅ Real-time status monitoring
- ✅ High-performance rendering
- ✅ User-friendly interface

By properly using different view modes and layouts, you can better observe and analyze multi-agent system behavior. 