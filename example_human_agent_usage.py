#!/usr/bin/env python3
"""
Example script showing how to use HumanAgent in the multi-agent simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_usage_examples():
    """Print usage examples for HumanAgent"""
    print("=" * 60)
    print("HumanAgent Integration Examples")
    print("=" * 60)
    
    print("\n1. Basic Usage:")
    print("   python run.py --human_agents 0 --agent_num 3")
    print("   # This makes agent 0 human-controlled, with 3 total agents")
    
    print("\n2. Multiple Human Agents:")
    print("   python run.py --human_agents 0,2 --agent_num 4")
    print("   # This makes agents 0 and 2 human-controlled, with 4 total agents")
    
    print("\n3. All AI Agents (default):")
    print("   python run.py --agent_num 6")
    print("   # This runs with 6 AI agents, no human control")
    
    print("\n4. Custom Steps and Rounds:")
    print("   python run.py --human_agents 1 --agent_num 3 --step_num 200 --num_of_turn 2")
    print("   # This runs 2 rounds of 200 steps each, with agent 1 human-controlled")
    
    print("\n5. Available Operations for Human Agents:")
    print("   - 'Navigator': Navigate to collect resources")
    print("   - 'share': Share resources with other agents")
    print("   - 'noop': Do nothing")
    print("   - 'move_left', 'move_right', 'move_up', 'move_down': Movement")
    print("   - 'do': Perform action (like chopping wood, mining)")
    print("   - 'sleep': Rest to recover energy")
    
    print("\n6. Available Resources:")
    print("   - 'wood', 'stone', 'coal', 'iron', 'diamond'")
    print("   - 'not_applicable': When no resource is needed")
    
    print("\n7. Target Agent IDs:")
    print("   - Use -1 for no target (when not sharing)")
    print("   - Use agent ID (0, 1, 2, etc.) when sharing resources")
    
    print("\n" + "=" * 60)
    print("Game Objective:")
    print("Work together with AI agents to collect a diamond as quickly as possible!")
    print("Only one agent needs to collect the diamond for the team to win.")
    print("=" * 60)

def print_collaboration_tips():
    """Print tips for human-AI collaboration"""
    print("\n🤝 Collaboration Tips:")
    print("1. Communicate with AI agents through resource sharing")
    print("2. AI agents will help the previous agent in the chain")
    print("3. Focus on your assigned role in the collaboration")
    print("4. Use 'share' operation to give resources to other agents")
    print("5. Monitor the game state and adapt your strategy")
    print("6. Work towards the common goal of collecting diamond")

def print_troubleshooting():
    """Print troubleshooting information"""
    print("\n🔧 Troubleshooting:")
    print("1. If you get import errors, make sure you're in the correct environment:")
    print("   conda activate mcrafter")
    print("   # or")
    print("   source mcrafter/bin/activate")
    
    print("\n2. If the simulation hangs, check that all human agents have provided input")
    print("3. Use Ctrl+C to interrupt the simulation if needed")
    print("4. Check the console output for any error messages")

if __name__ == "__main__":
    print_usage_examples()
    print_collaboration_tips()
    print_troubleshooting()
    
    print("\n🚀 Ready to try it out?")
    print("Start with a simple example:")
    print("python run.py --human_agents 0 --agent_num 3 --step_num 50") 