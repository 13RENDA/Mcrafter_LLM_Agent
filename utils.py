import matplotlib.pyplot as plt
from memory_system.agent import ActionStatus, Agent
import pandas as pd
from crafter import constants as const
from memory_system.utils import go_and_find
from typing import List
from memory_system.utils import print_color
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import argparse
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

class AgentActionProcessor:
    """Handles agent action processing"""
    
    @staticmethod
    def process_all_agent_actions(agents, env, n_players):
        """Process actions for all agents with different operation types"""
        for agent in agents:
            AgentActionProcessor._process_single_agent_action(agent, agents, env, n_players)
    
    @staticmethod
    def _process_single_agent_action(agent, agents, env, n_players):
        """Process action for a single agent based on operation type"""
        if agent.op == 'Navigator':
            AgentActionProcessor._process_navigator_action(agent)
        elif agent.op == 'share':
            AgentActionProcessor._process_share_action(agent, agents, env, n_players)
        else:
            AgentActionProcessor._process_general_action(agent)
    
    @staticmethod
    def _process_navigator_action(agent):
        """Process navigator operation for an agent"""
        agent.create_experience()  # Update working memory
        
        if agent.rss_to_collect != 'not_applicable':
            action, is_finished = go_and_find(agent.info, agent.rss_to_collect)
            agent.update_action(const.actions[action])
            
            if agent.action_status == ActionStatus.IN_PROGRESS and is_finished:
                agent.update_action_status(ActionStatus.ALMOST_DONE)
            elif agent.action_status == ActionStatus.ALMOST_DONE and is_finished:
                agent.update_action_status(ActionStatus.DONE)
        else:
            agent.update_action('noop')
            agent.update_action_status(ActionStatus.DONE)
    
    @staticmethod
    def _process_share_action(agent, agents, env, n_players):
        """Process share operation for an agent"""
        if agent.rss_to_share != 'not_applicable':
            env.switch_player(agent.id)
            env.exchange_item(agent.target_agent_id, agent.rss_to_share)
        
        agent.update_action('noop')
        agent.update_action_status(ActionStatus.DONE)
        
        # Interrupt all agents after the target agent
        AgentActionProcessor._interrupt_agents_after_target(agents, agent.target_agent_id, n_players)
    
    @staticmethod
    def _process_general_action(agent):
        """Process general operation for an agent"""
        agent.update_action(agent.op)
        agent.update_action_status(ActionStatus.DONE)
    
    @staticmethod
    def _interrupt_agents_after_target(agents, target_agent_id, n_players):
        """Interrupt all agents after the target agent"""
        for i in range(target_agent_id, n_players):
            agents[i].update_action_status(ActionStatus.INTERRUPTED)
    
    @staticmethod
    def collect_agent_actions(agents, n_players):
        """Collect all agents' actions for environment stepping"""
        agents_actions = [0] * n_players      
        for agent in agents:
            agents_actions[agent.id] = const.actions.index(agent.action)
        return agents_actions


class EnvironmentManager:
    """Manages environment state and updates"""
    
    @staticmethod
    def step_environment(env, agents_actions):
        """Step the environment with collected actions"""
        return env.step(agents_actions)
    
    @staticmethod
    def update_crafting_stations(agents, env):
        """Update crafting station status for all agents"""
        for agent in agents:
            if env._unlocked and "place_table" in env._unlocked:
                agent.update_crafting_station_status("table")
            if env._unlocked and 'place_furnace' in env._unlocked:
                agent.update_crafting_station_status("furnace")
    
    @staticmethod
    def get_tool_availability_info(env, n_players):
        """Get information about tool availability across all agents"""
        tool_availability_info = ""
        for tool in ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe']:
            for agent_id in range(n_players):
                if env._players[agent_id].inventory[tool] > 0:
                    tool_availability_info += f"Agent {agent_id} has {env._players[agent_id].inventory[tool]} {tool}\n"
        return tool_availability_info

class AgentStateManager:
    """Manages agent state updates and reporting"""
    
    @staticmethod
    def update_all_agent_states(agents, obs, step, env, info, episode_number):
        """Update states for all agents"""
        for agent in agents:
            agent.update_state(obs[agent.id], step, env, info[agent.id], episode_number, episode_timestep=step)
            print_color(f"Player {agent.id} is doing {agent.op, agent.rss_to_collect, agent.rss_to_share, agent.target_agent_id} with action {agent.action}; current status: {agent.action_status}", color='green')
    
    @staticmethod
    def identify_agents_needing_thought(agents, info):
        """Identify which agents need new thoughts"""
        agents_with_new_thought = [False] * len(agents)
        for agent in agents:
            if agent.action_status != ActionStatus.IN_PROGRESS and info[agent.id]['sleeping'] == False:
                agents_with_new_thought[agent.id] = True
        return agents_with_new_thought


class WorkingMemoryParser:
    """Handles parsing of agent working memory"""
    
    @staticmethod
    def parse_agent_working_memory(agent):
        """Parse agent's working memory content"""
        if agent.wm_content == "":
            return None, None
        
        try:
            wm_parts = agent.wm_content.split("### ")[1].split("- For ")
            
            if len(wm_parts) >= 3:
                long_term_goal_req = WorkingMemoryParser._extract_goal_requirement(wm_parts[1])
                current_goal_req = WorkingMemoryParser._extract_goal_requirement(wm_parts[2])
                return long_term_goal_req, current_goal_req
        except (IndexError, AttributeError):
            return None, None
        
        return None, None
    
    @staticmethod
    def _extract_goal_requirement(goal_text):
        """Extract and clean goal requirement text"""
        goal_req = goal_text.split(": ")[1:]
        goal_req = ', '.join(goal_req)
        
        if "* " in goal_req:
            goal_req = goal_req.split("* ")[1]
        
        return goal_req

class GoalGenerator:
    """Generates goals based on agent status and collaboration needs"""
    
    @staticmethod
    def generate_goal_from_agent_status(previous_agent, long_term_goal, current_goal):
        """Generate goal text based on agent's status"""
        if not long_term_goal or not current_goal:
            return "Agent status unclear.\n"
        
        if "help_agent" in long_term_goal:
            return GoalGenerator._generate_help_based_goal(previous_agent, current_goal)
        else:
            return GoalGenerator._generate_work_based_goal(previous_agent, long_term_goal, current_goal)
    
    @staticmethod
    def _generate_help_based_goal(previous_agent, current_goal):
        """Generate goal when previous agent is in help mode"""
        if 'ready' in current_goal or 'Ready' in current_goal:
            return f"Agent {previous_agent.id} is completing the task.\n"
        else:
            return (
                f"Agent {previous_agent.id} is working on {current_goal}"
                f"If he needs any tool, work on the tool and share with him.\n"
                f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                f"Else, work on advancing your tools.\n"
            )
    
    @staticmethod
    def _generate_work_based_goal(previous_agent, long_term_goal, current_goal):
        """Generate goal when previous agent is in work mode"""
        if 'ready' in long_term_goal or 'Ready' in long_term_goal:
            return f"Agent {previous_agent.id} is completing its task. You should focus on advancing your tools.\n"
        else:
            if "work on long-term goal" in current_goal:
                return (
                    f"Agent {previous_agent.id} is completing {long_term_goal}.\n"
                    f"If he needs any tool, work on the tool and share with him.\n"
                    f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                    f"Else, work on advancing your tools.\n"
                )
            else:
                return (
                    f"Agent {previous_agent.id} is working on {long_term_goal} He is focusing on {current_goal} "
                    f"If he needs any tool, work on the tool and share with him.\n"
                    f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                    f"Else, work on advancing your tools.\n"
                )

class CollaborationContextGenerator:
    """Generates collaboration context for different agent types"""
    
    def __init__(self, n_players):
        self.n_players = n_players
        self.starting_furnace = False
    
    def generate_context_for_agent(self, agent, agents):
        """Generate collaboration context for a specific agent"""
        if agent.id == 0:
            return self._generate_agent_0_context()
        elif agent.id == self.n_players - 1:
            return self._generate_last_agent_context(agent)
        else:
            return self._generate_middle_agent_context(agent, agents)
    
    def _generate_agent_0_context(self):
        """Generate context for agent 0 (main agent)"""
        return (
            "\n### Collaboration: Need Your Help \n"
            f"You are agent 0. Focus on your tasks while other agents share resources with you. "
            f"Once you craft an iron_pickaxe, you should share it to agent {self.n_players-1}\n"
        )
    
    def _generate_last_agent_context(self, agent):
        """Generate context for the last agent (diamond collector)"""
        goal = "While other agents are working on make_iron_pickaxe, your long-term goal and current goal is to !!collect diamond!! You should navigate to a diamond, and collect the diamond. Other agent will share the tool with you.\n"
        
        return (
            "\n### Collaboration: Need Your Help!!!\n"
            f"{goal}"
            "\n### Collaboration Policy \n"
            f"You are agent {agent.id}. Your goal is to collect diamond. You should navigate to a diamond and collect the diamond. Other agents will share the tool with you.\n"
        )
    
    def _generate_middle_agent_context(self, agent, agents):
        """Generate context for middle agents"""
        previous_agent = agents[agent.id-1]
        goal = self._parse_previous_agent_goal(previous_agent)
        
        if agent.id == 1:
            return self._generate_agent_1_context(agent, goal)
        else:
            return self._generate_other_middle_agent_context(agent, goal)
    
    def _parse_previous_agent_goal(self, previous_agent):
        """Parse the previous agent's working memory to determine current goal"""
        long_term_goal, current_goal = WorkingMemoryParser.parse_agent_working_memory(previous_agent)
        return GoalGenerator.generate_goal_from_agent_status(previous_agent, long_term_goal, current_goal)
    
    def _generate_agent_1_context(self, agent, goal):
        """Generate context specifically for agent 1"""
        if self.starting_furnace or ('make_stone_pickaxe' in goal):
            self.starting_furnace = True
            return (
                "\n### Collaboration: Need Your Help!!!\n"
                f"While other agents working towards make_iron_pickaxe, your long-term goal and current goal should be !!share stone to agent 0!! You should focus on !!collect stone!! and share stone immediately.\n"
                "\n### Collaboration Policy \n"
                f"You are agent {agent.id}. Your ultimate goal is help_agent 0.\n"
                f"You long-term goal should be help_agent 0. Your current goal should be collect and share stone.\n"
            )
        else:
            return (
                "\n### Collaboration: Need Your Help!!!\n"
                f"{goal}"
                "\n### Collaboration Policy \n"
                f"You are agent {agent.id}. Your ultimate goal is help_agent 0. "
                f"You long-term goal should be 'help_agent'. You can do so by sharing resources/tools as soon as you gain access to the required items.\n"
                f"!!!Determine your current goal based on agents who need help. If they don't need help, you should focus on advancing your tools.\n"
            )
    
    def _generate_other_middle_agent_context(self, agent, goal):
        """Generate context for other middle agents"""
        return (
            "\n### Collaboration: Need Your Help!!!\n"
            f"{goal}"
            "\n### Collaboration Policy \n"
            f"You are agent {agent.id}. Your ultimate goal is help_agent 0 and help_agent {agent.id-1}. You should prioritize helping agent 0.\n"
            f"You long-term goal should be 'help_agent'. You can do so by sharing resources/tools as soon as you gain access to the required items.\n"
            f"Do not share resources/tools that are not needed by the agents.\n"
            f"Try to help the agents. If they don't need help, you should focus on advancing your tools.\n"
            f"If you are not sure what to do, share stone to agent 0.\n"
            f"Reconsider how you can help the previous agent.\n"
            f"!!!Determine your current goal based on agents who need help. If they don't need help, you should focus on advancing your tools.\n"
        )

class AgentThinkingProcessor:
    """Handles parallel processing of agent thinking"""
    
    @staticmethod
    def process_agent_thinking_parallel(agents, agents_contexts, info):
        """Process agent thinking in parallel using ThreadPoolExecutor, but skip HumanAgents"""
        agents_responses = []
        

        ai_agents = [agent for agent in agents if not (hasattr(agent, 'is_human') and agent.is_human)]
        
        if ai_agents:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(Agent.process_agent, agent, agents_contexts, info) 
                    for agent in ai_agents
                ]
                ai_responses = [future.result() for future in futures]
                agents_responses.extend(ai_responses)
        
        return agents_responses
    
    @staticmethod
    def update_agents_from_responses(agents, agents_responses):
        """Update agents based on their thinking responses"""
        for response in agents_responses:
            agent = agents[response['id']]
            if response['thoughts'] is not None:
                op, rss_to_collect, rss_to_share, target_agent_id = agent.consolidate_experience(response['thoughts'])
                agent.update_current_skill(op, rss_to_collect, rss_to_share, target_agent_id)
                agent.update_action_status(ActionStatus.IN_PROGRESS)


class SimulationLogger:
    """Handles simulation reporting and logging"""
    def __init__(self):
        # Task mapping: easy to modify and extend
        self.tasks = {
            "Collect wood": "collect_wood",
            "Place table": "place_table",
            "Make wood pickaxe": "make_wood_pickaxe",
            "Collect stone": "collect_stone",
            "Make stone pickaxe": "make_stone_pickaxe",
            "Collect iron": "collect_iron",
            "Collect coal": "collect_coal",
            "Place furnace": "place_furnace",
            "Make iron pickaxe": "make_iron_pickaxe",
            "Collect diamond": "collect_diamond",
        }

    @staticmethod
    def print_step_header(total_step, current_step):
        """Print step header information"""
        print_color("="*50, "total step: ", total_step, "current step: ", current_step, "="*50, color='green')
    
    @staticmethod
    def print_simulation_complete():
        """Print simulation completion message"""
        print_color("Simulation completed!", color='green')
    
    @staticmethod
    def show_step_report(agents, agents_with_new_thought):
        report = []
        # fig, ax = plt.subplots(1, len(agents), figsize=(2 * len(agents), 2))
        for agent_id, has_new_thought in enumerate(agents_with_new_thought):
            agent = agents[agent_id]
            # figures
            # ax[agent.id].imshow(agent.obs)
            # ax[agent.id].set_axis_off()
            # ax[agent.id].set_title(f"Player {agent.id}")
            # info
            if has_new_thought:
                # Handle HumanAgent differently
                if hasattr(agent, 'is_human') and agent.is_human:
                    context = "Human controlled agent"
                    info = (
                            f"long_term_goal: Human controlled\n\n"
                            f"current_goal: Human controlled\n\n"
                            f"op: {agent.op}\n\n"
                            f"navigate to: {agent.rss_to_collect}\n\n"
                            f"share: {agent.rss_to_share}\n\n"
                            f"target_agent_id: {agent.target_agent_id}\n"
                    )
                    summary = f"Human agent {agent.id} chose: {agent.op}"
                else:
                    context = agent.think_context
                    final_response = agent.experience.episodic_memory.final_response
                    reflection, goal, action = final_response.reflection, final_response.goal, final_response.action
                    info = (
                            f"long_term_goal: {goal.long_term_goal.value}\n\n"
                            f"current_goal: {goal.current_goal.value}\n\n"
                            f"op: {agent.op.value}\n\n"
                            f"navigate to: {agent.rss_to_collect.value}\n\n"
                            f"share: {agent.rss_to_share.value}\n\n"
                            f"target_agent_id: {agent.target_agent_id}\n"
                    )
                    summary = final_response.summary
            
                report.append({"agent_id": agent.id, 
                            "context": context,
                            "info": info,
                            "summary": summary})
            else:
                report.append({"agent_id": agent.id,
                            "context": "in progress",
                            "info": "in progress",
                            "summary": "in progress"})
                            
        for entry in report:
            agent_type = "HumanAgent" if hasattr(agents[entry['agent_id']], 'is_human') and agents[entry['agent_id']].is_human else "Agent"
            print(f"\n[{agent_type} {entry['agent_id']}]")
            print("Context (Thinking):\n", entry['context'])
            # print("Info:\n", entry['info'])
            # print("Summary:\n", entry['summary'])
    
    def log_stats(self, agents):
        min_completion_steps = []
        for agent in agents:
            df = pd.DataFrame(agent.replay)
            # Find the first step for each task completion
            completion_steps = {task_name: df[df[task_col] == 1]['step'].min() for task_name, task_col in self.tasks.items()}
            min_completion_steps.append(completion_steps)
        df = pd.DataFrame(min_completion_steps)
        print("Minimum step for each agent to complete each task:")
        print(df)
        print("\nEarliest step (across all agents) for each task:")
        earliest_steps = df.min()
        print(earliest_steps)
        earliest_df = pd.DataFrame([earliest_steps])
        return earliest_df
    

class SimulationContextManager:
    """Manages the creation of simulation contexts for agents"""
    
    def __init__(self, n_players):
        self.context_generator = CollaborationContextGenerator(n_players)
    
    def create_agent_contexts(self, agents, info):
        """Create contexts for all agents that need to think"""
        agents_contexts = {}
        # agents_with_new_thought = [False] * len(agents)
        for agent in agents:
            context = agent.create_experience()
            print(f"agent_{agent.id}", agent.wm_content)
            
            if agent.action_status != ActionStatus.IN_PROGRESS and info[agent.id]['sleeping'] == False:
                # agents_with_new_thought[agent.id] = True
                collaboration_context = self.context_generator.generate_context_for_agent(agent, agents)
                context += collaboration_context
                agents_contexts[agent.id] = context
        
        return agents_contexts

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Run multi-agent simulation with optional human agents.')
    parser.add_argument('--human_agents', type=str, default='',
                        help='Comma-separated list of agent ids to be human-controlled, e.g. "0,2". Default: all AI agents.')
    parser.add_argument('--memory', action='store_true', default=False,
                        help='Whether agents have memory (default: False).')
    parser.add_argument('--communication', action='store_true', default=False,
                        help='Whether agents have communication (default: False).')
    parser.add_argument('--agent_num', type=int, default=6,
                        help='Number of agents (default: 6).')
    parser.add_argument('--step_num', type=int, default=350,
                        help='Number of steps (default: 350).')
    parser.add_argument('--num_of_turn', type=int, default=1,
                        help='Number of simulation turns (default: 1).')
    # Add more custom arguments here as needed
    return parser




