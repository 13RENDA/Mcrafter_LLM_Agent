#!/usr/bin/env python3
"""
Fixed Advanced GUI for HumanAgent integration - resolves black screen issue
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


class FixedAdvancedHumanAgentGUI:
    """Fixed Advanced GUI class for HumanAgent integration - no black screen"""
    
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
        self.show_help = False
        self.show_inventory = False
        self.input_mode = False
        self.input_prompt = ""
        self.input_field = ""
        self.input_type = ""  # 'op', 'collect', 'share', 'target'
        
        # Pygame setup
        self.window_size = (1200, 800)
        self.game_area_size = (600, 600)
        self.sidebar_width = 600
        self.fps = 5
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'gray': (128, 128, 128),
            'dark_gray': (64, 64, 64),
            'light_gray': (192, 192, 192),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128)
        }
        
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
        }
        
        # UI buttons
        self.buttons = self.create_buttons()
        
        print('Fixed Advanced GUI Controls:')
        print('  h: Human agent action selection')
        print('  p: Pause/Resume simulation')
        print('  i: Show/hide inventory')
        print('  ?: Show/hide help')
        print('  ESC: Quit')
        print('  Mouse: Click buttons for actions')
        print('  ENTER: Confirm input (when in input mode)')
    
    def create_buttons(self):
        """Create UI buttons"""
        buttons = {}
        
        # Action buttons
        button_y = 650
        button_width = 80
        button_height = 30
        button_spacing = 10
        
        actions = [
            ('Navigator', 'Navigator'),
            ('Share', 'share'),
            ('Noop', 'noop'),
            ('Move Left', 'move_left'),
            ('Move Right', 'move_right'),
            ('Move Up', 'move_up'),
            ('Move Down', 'move_down'),
            ('Do', 'do'),
            ('Sleep', 'sleep')
        ]
        
        for i, (text, action) in enumerate(actions):
            x = 610 + (i % 3) * (button_width + button_spacing)
            y = button_y + (i // 3) * (button_height + 5)
            buttons[action] = {
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': text,
                'action': action,
                'color': self.colors['light_gray'],
                'hover_color': self.colors['yellow']
            }
        
        # Control buttons
        control_buttons = [
            ('Pause', 'pause', 610, 750),
            ('Human Action', 'human_action', 700, 750),
            ('Help', 'help', 790, 750),
            ('Inventory', 'inventory', 880, 750)
        ]
        
        for text, action, x, y in control_buttons:
            buttons[action] = {
                'rect': pygame.Rect(x, y, 80, 30),
                'text': text,
                'action': action,
                'color': self.colors['blue'],
                'hover_color': self.colors['orange']
            }
        
        return buttons
    
    def render(self):
        """Render the game state"""
        # Clear screen
        self.screen.fill(self.colors['black'])
        
        # Render game area
        self.render_game_area()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render buttons
        self.render_buttons()
        
        # Render overlays
        if self.show_help:
            self.render_help_overlay()
        if self.show_inventory:
            self.render_inventory_overlay()
        if self.input_mode:
            self.render_input_overlay()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def render_game_area(self):
        """Render the main game area"""
        # Render environment
        image = self.env.render(self.game_area_size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render game area border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (0, 0, self.game_area_size[0], self.game_area_size[1]), 2)
    
    def render_sidebar(self):
        """Render the sidebar with agent information"""
        sidebar_x = self.game_area_size[0] + 10
        
        # Title
        font_large = pygame.font.Font(None, 36)
        title = font_large.render('Fixed HumanAgent Simulation', True, self.colors['white'])
        self.screen.blit(title, (sidebar_x, 10))
        
        # Step counter
        font = pygame.font.Font(None, 24)
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, self.colors['white'])
        self.screen.blit(step_text, (sidebar_x, 50))
        
        # Progress bar
        progress = self.current_step / self.max_steps
        bar_width = 400
        bar_height = 20
        bar_x = sidebar_x
        bar_y = 80
        
        # Background
        pygame.draw.rect(self.screen, self.colors['dark_gray'], 
                        (bar_x, bar_y, bar_width, bar_height))
        # Progress
        pygame.draw.rect(self.screen, self.colors['green'], 
                        (bar_x, bar_y, int(bar_width * progress), bar_height))
        # Border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Agent status
        self.render_agent_status(sidebar_x, bar_y + 40)
        
        # Status indicators
        self.render_status_indicators(sidebar_x, 400)
    
    def render_agent_status(self, x, y):
        """Render agent status information"""
        font = pygame.font.Font(None, 20)
        title = font.render('Agent Status:', True, self.colors['yellow'])
        self.screen.blit(title, (x, y))
        
        y_offset = y + 30
        for i, agent in enumerate(self.agents):
            # Agent type and status
            agent_type = "Human" if hasattr(agent, 'is_human') and agent.is_human else "AI"
            status = "Waiting" if agent.action_status == ActionStatus.DONE else "Working"
            color = self.colors['yellow'] if hasattr(agent, 'is_human') and agent.is_human else self.colors['white']
            
            agent_text = font.render(f'Agent {i} ({agent_type}): {status}', True, color)
            self.screen.blit(agent_text, (x, y_offset))
            
            # Current operation
            if hasattr(agent, 'op') and agent.op:
                op_text = font.render(f'  Op: {agent.op}', True, self.colors['light_gray'])
                self.screen.blit(op_text, (x + 10, y_offset + 20))
            
            # Resources
            if hasattr(agent, 'rss_to_collect') and agent.rss_to_collect:
                collect_text = font.render(f'  Collect: {agent.rss_to_collect}', True, self.colors['light_gray'])
                self.screen.blit(collect_text, (x + 10, y_offset + 40))
            
            if hasattr(agent, 'rss_to_share') and agent.rss_to_share:
                share_text = font.render(f'  Share: {agent.rss_to_share}', True, self.colors['light_gray'])
                self.screen.blit(share_text, (x + 10, y_offset + 60))
            
            y_offset += 80
    
    def render_status_indicators(self, x, y):
        """Render status indicators"""
        font = pygame.font.Font(None, 20)
        
        # Pause indicator
        if self.paused:
            pause_text = font.render('PAUSED - Press P to resume', True, self.colors['red'])
            self.screen.blit(pause_text, (x, y))
        
        # Input mode indicator
        if self.input_mode:
            input_text = font.render('INPUT MODE - Type your response', True, self.colors['green'])
            self.screen.blit(input_text, (x, y + 25))
        
        # Current human agent
        if self.current_human_agent is not None:
            current_text = font.render(f'Current Human Agent: {self.current_human_agent}', True, self.colors['orange'])
            self.screen.blit(current_text, (x, y + 50))
    
    def render_buttons(self):
        """Render UI buttons"""
        font = pygame.font.Font(None, 18)
        mouse_pos = pygame.mouse.get_pos()
        
        for button_id, button in self.buttons.items():
            # Check hover
            color = button['hover_color'] if button['rect'].collidepoint(mouse_pos) else button['color']
            
            # Draw button
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['white'], button['rect'], 2)
            
            # Draw text
            text = font.render(button['text'], True, self.colors['black'])
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
    
    def render_input_overlay(self):
        """Render input interface overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill(self.colors['black'])
        self.screen.blit(overlay, (0, 0))
        
        font = pygame.font.Font(None, 24)
        
        # Input prompt
        prompt_text = font.render(self.input_prompt, True, self.colors['white'])
        self.screen.blit(prompt_text, (50, 200))
        
        # Input field
        input_text = font.render(f'Input: {self.input_field}', True, self.colors['yellow'])
        self.screen.blit(input_text, (50, 230))
        
        # Instructions
        instruction_text = font.render('Press ENTER to confirm, ESC to cancel', True, self.colors['light_gray'])
        self.screen.blit(instruction_text, (50, 260))
    
    def render_help_overlay(self):
        """Render help overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill(self.colors['black'])
        self.screen.blit(overlay, (0, 0))
        
        # Help content
        font = pygame.font.Font(None, 20)
        help_texts = [
            'Fixed Advanced GUI - Controls:',
            '',
            'Keyboard Controls:',
            '  WASD: Move',
            '  SPACE: Do action',
            '  TAB: Sleep',
            '  H: Human action selection',
            '  P: Pause/Resume',
            '  I: Show inventory',
            '  ?: Show this help',
            '  ESC: Quit',
            '',
            'Mouse Controls:',
            '  Click buttons for actions',
            '',
            'Input Mode:',
            '  Type your response and press ENTER',
            '  Press ESC to cancel input',
            '',
            'Game Objective:',
            '  Work with AI agents to collect diamond',
            '  Use resource sharing for collaboration'
        ]
        
        y = 100
        for text in help_texts:
            text_surface = font.render(text, True, self.colors['white'])
            self.screen.blit(text_surface, (50, y))
            y += 25
    
    def render_inventory_overlay(self):
        """Render inventory overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill(self.colors['black'])
        self.screen.blit(overlay, (0, 0))
        
        # Inventory content
        font = pygame.font.Font(None, 20)
        title = font.render('Agent Inventories:', True, self.colors['yellow'])
        self.screen.blit(title, (50, 50))
        
        y = 100
        for i, agent in enumerate(self.agents):
            if hasattr(self.env, '_players') and i < len(self.env._players):
                player = self.env._players[i]
                agent_text = font.render(f'Agent {i} Inventory:', True, self.colors['white'])
                self.screen.blit(agent_text, (50, y))
                
                y += 25
                for item, count in player.inventory.items():
                    if count > 0:
                        item_text = font.render(f'  {item}: {count}', True, self.colors['light_gray'])
                        self.screen.blit(item_text, (70, y))
                        y += 20
                y += 20
    
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_mouse_click(event.pos)
    
    def handle_normal_event(self, event):
        """Handle events when not in input mode"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_h:
            self.start_human_action_selection()
        elif event.key == pygame.K_i:
            self.show_inventory = not self.show_inventory
        elif event.key == pygame.K_QUESTION:
            self.show_help = not self.show_help
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
    
    def handle_mouse_click(self, pos):
        """Handle mouse clicks on buttons"""
        for button_id, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                self.handle_button_action(button_id)
                break
    
    def handle_button_action(self, button_id):
        """Handle button actions"""
        if button_id == 'pause':
            self.paused = not self.paused
        elif button_id == 'human_action':
            self.start_human_action_selection()
        elif button_id == 'help':
            self.show_help = not self.show_help
        elif button_id == 'inventory':
            self.show_inventory = not self.show_inventory
        else:
            # Handle action buttons
            self.handle_action(button_id)
    
    def handle_action(self, action):
        """Handle game actions"""
        # For now, just print the action
        print(f"Action selected: {action}")
    
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
        print(f"Starting Fixed Advanced HumanAgent GUI with {self.n_players} agents")
        print(f"Human agents: {self.human_agent_ids}")
        print("Press 'h' for human actions, 'p' to pause/resume, '?' for help, ESC to quit")
        
        while self.running:
            self.handle_events()
            self.run_simulation_step()
            self.render()
        
        pygame.quit()
        print("Fixed Advanced GUI closed")


def main():
    parser = argparse.ArgumentParser(description='Fixed Advanced HumanAgent GUI for multi-agent simulation')
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
    gui = FixedAdvancedHumanAgentGUI(
        human_agent_ids=human_agent_ids,
        n_players=args.agent_num,
        max_steps=args.step_num
    )
    gui.fps = args.fps
    gui.run()


if __name__ == '__main__':
    main() 