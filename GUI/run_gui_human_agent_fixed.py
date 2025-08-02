#!/usr/bin/env python3
"""
Fixed GUI for HumanAgent integration - resolves black screen issue
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise
from PIL import Image

import crafter
from memory_system.agent import HumanAgent, Agent
from run import initialize_agents
from utils import AgentActionProcessor, EnvironmentManager, SimulationLogger, AgentStateManager, SimulationContextManager, AgentThinkingProcessor, ActionStatus


class FixedHumanAgentGUI:
    """Fixed GUI class for HumanAgent integration - no black screen"""
    
    def __init__(self, human_agent_ids=None, n_players=3, max_steps=350):
        self.human_agent_ids = human_agent_ids or []
        self.n_players = n_players
        self.max_steps = max_steps
        
        # Initialize agents
        self.agents = initialize_agents(human_agent_ids=human_agent_ids, n_players=n_players)
        
        # Initialize environment
        self.env = crafter.Env(length=max_steps, n_players=n_players, seed=4)
        self.env.reset()
        
        # Initialize processors
        self.action_processor = AgentActionProcessor()
        self.env_manager = EnvironmentManager()
        self.reporter = SimulationLogger()
        self.agent_state_manager = AgentStateManager()
        self.simulation_context_manager = SimulationContextManager(n_players)
        self.agent_thinking_processor = AgentThinkingProcessor()
        
        # GUI state
        self.current_step = 0
        self.running = True
        self.paused = False
        self.current_human_agent = None
        self.waiting_for_human_input = False
        self.input_mode = False
        self.input_buffer = ""
        self.input_prompt = ""
        self.input_field = ""
        self.input_type = ""  # 'op', 'collect', 'share', 'target'
        
        # Pygame setup
        self.window_size = (800, 600)
        self.fps = 5
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        
        # Key mappings
        self.keymap = {
            pygame.K_a: 'move_left',
            pygame.K_d: 'move_right',
            pygame.K_w: 'move_up',
            pygame.K_s: 'move_down',
            pygame.K_SPACE: 'do',
            pygame.K_TAB: 'sleep',
            pygame.K_r: 'place_stone',
            pygame.K_t: 'place_table',
            pygame.K_f: 'place_furnace',
            pygame.K_p: 'place_plant',
            pygame.K_1: 'make_wood_pickaxe',
            pygame.K_2: 'make_stone_pickaxe',
            pygame.K_3: 'make_iron_pickaxe',
            pygame.K_4: 'make_wood_sword',
            pygame.K_5: 'make_stone_sword',
            pygame.K_6: 'make_iron_sword',
            pygame.K_LSHIFT: 'switch_player',
            pygame.K_p: 'pause',
            pygame.K_h: 'human_action',
        }
        
        print('Fixed GUI Actions:')
        for key, action in self.keymap.items():
            print(f'  {pygame.key.name(key)}: {action}')
        print('  h: Human agent action selection')
        print('  p: Pause/Resume simulation')
        print('  ESC: Quit')
        print('  ENTER: Confirm input (when in input mode)')
    
    def render(self):
        """Render the game state"""
        # Render environment
        image = self.env.render(self.window_size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render UI overlay
        self.render_ui_overlay()
        
        # Render input interface if in input mode
        if self.input_mode:
            self.render_input_interface()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def render_ui_overlay(self):
        """Render UI overlay with agent information"""
        font = pygame.font.Font(None, 24)
        
        # Step counter
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
        self.screen.blit(step_text, (10, 10))
        
        # Agent status
        y_offset = 40
        for i, agent in enumerate(self.agents):
            agent_type = "Human" if hasattr(agent, 'is_human') and agent.is_human else "AI"
            status = "Waiting" if agent.action_status == ActionStatus.DONE else "Working"
            color = (255, 255, 0) if hasattr(agent, 'is_human') and agent.is_human else (255, 255, 255)
            
            agent_text = font.render(f'Agent {i} ({agent_type}): {status}', True, color)
            self.screen.blit(agent_text, (10, y_offset))
            
            if hasattr(agent, 'op') and agent.op:
                op_text = font.render(f'  Op: {agent.op}', True, (200, 200, 200))
                self.screen.blit(op_text, (20, y_offset + 20))
            
            y_offset += 50
        
        # Pause indicator
        if self.paused:
            pause_text = font.render('PAUSED - Press P to resume', True, (255, 0, 0))
            self.screen.blit(pause_text, (10, self.window_size[1] - 30))
        
        # Input mode indicator
        if self.input_mode:
            input_text = font.render('INPUT MODE - Type your response', True, (0, 255, 0))
            self.screen.blit(input_text, (10, self.window_size[1] - 60))
    
    def render_input_interface(self):
        """Render input interface overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        font = pygame.font.Font(None, 24)
        
        # Input prompt
        prompt_text = font.render(self.input_prompt, True, (255, 255, 255))
        self.screen.blit(prompt_text, (50, 200))
        
        # Input field
        input_text = font.render(f'Input: {self.input_field}', True, (255, 255, 0))
        self.screen.blit(input_text, (50, 230))
        
        # Instructions
        instruction_text = font.render('Press ENTER to confirm, ESC to cancel', True, (200, 200, 200))
        self.screen.blit(instruction_text, (50, 260))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.input_mode:
                    self.handle_input_event(event)
                else:
                    self.handle_normal_event(event)
    
    def handle_normal_event(self, event):
        """Handle events when not in input mode"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_h:
            self.start_human_action_selection()
        elif event.key in self.keymap:
            action = self.keymap[event.key]
            self.handle_action(action)
    
    def handle_input_event(self, event):
        """Handle events when in input mode"""
        if event.key == pygame.K_ESCAPE:
            # Cancel input
            self.input_mode = False
            self.input_field = ""
            self.input_type = ""
        elif event.key == pygame.K_RETURN:
            # Confirm input
            self.process_input()
        elif event.key == pygame.K_BACKSPACE:
            # Delete character
            self.input_field = self.input_field[:-1]
        else:
            # Add character
            if event.unicode.isprintable():
                self.input_field += event.unicode
    
    def handle_action(self, action):
        """Handle game actions"""
        if action == 'switch_player':
            # Switch between human agents
            human_agents = [i for i, agent in enumerate(self.agents) 
                          if hasattr(agent, 'is_human') and agent.is_human]
            if human_agents:
                if self.current_human_agent is None:
                    self.current_human_agent = human_agents[0]
                else:
                    current_idx = human_agents.index(self.current_human_agent)
                    self.current_human_agent = human_agents[(current_idx + 1) % len(human_agents)]
                print(f"Switched to Human Agent {self.current_human_agent}")
        elif action == 'human_action':
            self.start_human_action_selection()
    
    def start_human_action_selection(self):
        """Start human agent action selection process"""
        human_agents = [i for i, agent in enumerate(self.agents) 
                       if hasattr(agent, 'is_human') and agent.is_human]
        
        if not human_agents:
            print("No human agents available")
            return
        
        if self.current_human_agent is None:
            self.current_human_agent = human_agents[0]
        
        # Start input sequence
        self.input_mode = True
        self.input_type = 'op'
        self.input_prompt = f"Human Agent {self.current_human_agent} - Enter operation type (Navigator/share/noop/etc):"
        self.input_field = ""
    
    def process_input(self):
        """Process the current input"""
        if self.input_type == 'op':
            op = self.input_field.strip()
            if not op:
                op = 'noop'
            
            # Store operation and move to next input
            self.agents[self.current_human_agent].op = op
            self.input_type = 'collect'
            self.input_prompt = f"Enter resource to collect (or 'not_applicable'):"
            self.input_field = ""
            
        elif self.input_type == 'collect':
            rss_to_collect = self.input_field.strip()
            if not rss_to_collect:
                rss_to_collect = 'not_applicable'
            
            # Store collect resource and move to next input
            self.agents[self.current_human_agent].rss_to_collect = rss_to_collect
            self.input_type = 'share'
            self.input_prompt = f"Enter resource to share (or 'not_applicable'):"
            self.input_field = ""
            
        elif self.input_type == 'share':
            rss_to_share = self.input_field.strip()
            if not rss_to_share:
                rss_to_share = 'not_applicable'
            
            # Store share resource and move to next input
            self.agents[self.current_human_agent].rss_to_share = rss_to_share
            self.input_type = 'target'
            self.input_prompt = f"Enter target agent id (or -1 if not applicable):"
            self.input_field = ""
            
        elif self.input_type == 'target':
            try:
                target_agent_id = int(self.input_field.strip())
            except ValueError:
                target_agent_id = -1
            
            # Store target agent and complete input
            self.agents[self.current_human_agent].target_agent_id = target_agent_id
            
            # Update agent skills
            agent = self.agents[self.current_human_agent]
            agent.update_current_skill(agent.op, agent.rss_to_collect, agent.rss_to_share, target_agent_id)
            
            print(f"[Human Agent {self.current_human_agent}] Action set: op={agent.op}, collect={agent.rss_to_collect}, share={agent.rss_to_share}, target_agent_id={target_agent_id}")
            
            # Exit input mode
            self.input_mode = False
            self.input_field = ""
            self.input_type = ""
    
    def run_simulation_step(self):
        """Run one step of the simulation"""
        if self.paused or self.input_mode:
            return
        
        # Process agent actions
        self.action_processor.process_all_agent_actions(self.agents, self.env, self.n_players)
        
        # Collect actions
        agents_actions = self.action_processor.collect_agent_actions(self.agents, self.n_players)
        
        # Step environment
        obs, rewards, done, info = self.env_manager.step_environment(self.env, agents_actions)
        self.env_manager.update_crafting_stations(self.agents, self.env)
        
        # Update agent states
        self.agent_state_manager.update_all_agent_states(
            self.agents, obs, self.current_step, self.env, info, episode_number=0
        )
        
        # Process agent thinking
        agents_with_new_thought = self.agent_state_manager.identify_agents_needing_thought(self.agents, info)
        agents_contexts = self.simulation_context_manager.create_agent_contexts(self.agents, info)
        agents_responses = self.agent_thinking_processor.process_agent_thinking_parallel(
            self.agents, agents_contexts, info
        )
        self.agent_thinking_processor.update_agents_from_responses(self.agents, agents_responses)
        
        # Show step report
        self.reporter.show_step_report(self.agents, agents_with_new_thought)
        
        self.current_step += 1
        
        # Check if simulation is done
        if done or self.current_step >= self.max_steps:
            print("Simulation completed!")
            self.running = False
    
    def run(self):
        """Main simulation loop"""
        print(f"Starting Fixed HumanAgent GUI with {self.n_players} agents")
        print(f"Human agents: {self.human_agent_ids}")
        print("Press 'h' to select human agent actions, 'p' to pause/resume, ESC to quit")
        
        while self.running:
            self.handle_events()
            self.run_simulation_step()
            self.render()
        
        pygame.quit()
        print("Fixed GUI closed")


def main():
    parser = argparse.ArgumentParser(description='Fixed HumanAgent GUI for multi-agent simulation')
    parser.add_argument('--human_agents', type=str, default='',
                        help='Comma-separated list of agent ids to be human-controlled, e.g. "0,2"')
    parser.add_argument('--agent_num', type=int, default=3,
                        help='Number of agents (default: 3)')
    parser.add_argument('--step_num', type=int, default=350,
                        help='Number of steps (default: 350)')
    parser.add_argument('--fps', type=int, default=5,
                        help='FPS for GUI (default: 5)')
    
    args = parser.parse_args()
    
    # Parse human agent IDs
    human_agent_ids = []
    if args.human_agents.strip():
        human_agent_ids = [int(x.strip()) for x in args.human_agents.split(',') if x.strip().isdigit()]
    
    # Create and run GUI
    gui = FixedHumanAgentGUI(
        human_agent_ids=human_agent_ids,
        n_players=args.agent_num,
        max_steps=args.step_num
    )
    gui.fps = args.fps
    gui.run()


if __name__ == '__main__':
    main() 