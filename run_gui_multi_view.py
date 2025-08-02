#!/usr/bin/env python3
"""
Multi-view GUI for HumanAgent integration with perspective switching
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


class MultiViewGUI:
    """Multi-view GUI class with perspective switching"""
    
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
        self.input_prompt = ""
        self.input_field = ""
        self.input_type = ""
        
        # View state
        self.current_view_mode = "single"  # "single", "multi", "overview"
        self.current_followed_agent = 0  # Which agent to follow in single view
        self.view_layout = "horizontal"  # "horizontal", "vertical", "grid"
        
        # Pygame setup
        self.window_size = (1400, 900)
        self.game_area_size = (600, 600)
        self.sidebar_width = 400
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
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255)
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
        
        print('Multi-View GUI Controls:')
        print('  h: Human agent action selection')
        print('  p: Pause/Resume simulation')
        print('  v: Switch view mode (single/multi/overview)')
        print('  l: Switch layout (horizontal/vertical/grid)')
        print('  0-9: Follow specific agent (in single view)')
        print('  ESC: Quit')
        print('  Mouse: Click buttons for actions')
    
    def create_buttons(self):
        """Create UI buttons"""
        buttons = {}
        
        # View control buttons
        view_buttons = [
            ('Single View', 'single_view', 610, 650),
            ('Multi View', 'multi_view', 710, 650),
            ('Overview', 'overview', 810, 650),
            ('Layout', 'switch_layout', 910, 650)
        ]
        
        for text, action, x, y in view_buttons:
            buttons[action] = {
                'rect': pygame.Rect(x, y, 80, 30),
                'text': text,
                'action': action,
                'color': self.colors['blue'],
                'hover_color': self.colors['orange']
            }
        
        # Agent follow buttons
        for i in range(min(self.n_players, 10)):
            x = 610 + (i % 5) * 60
            y = 700 + (i // 5) * 35
            buttons[f'follow_agent_{i}'] = {
                'rect': pygame.Rect(x, y, 50, 25),
                'text': f'Agent {i}',
                'action': f'follow_agent_{i}',
                'color': self.colors['light_gray'],
                'hover_color': self.colors['yellow']
            }
        
        # Control buttons
        control_buttons = [
            ('Pause', 'pause', 610, 750),
            ('Human Action', 'human_action', 700, 750),
            ('Help', 'help', 790, 750)
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
        
        # Render based on view mode
        if self.current_view_mode == "single":
            self.render_single_view()
        elif self.current_view_mode == "multi":
            self.render_multi_view()
        elif self.current_view_mode == "overview":
            self.render_overview()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render buttons
        self.render_buttons()
        
        # Render input interface if in input mode
        if self.input_mode:
            self.render_input_interface()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def render_single_view(self):
        """Render single agent view"""
        # Switch to followed agent's perspective
        self.env.switch_player(self.current_followed_agent)
        
        # Render environment from that agent's perspective
        image = self.env.render(self.game_area_size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render game area border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (0, 0, self.game_area_size[0], self.game_area_size[1]), 2)
        
        # Render current agent indicator
        font = pygame.font.Font(None, 36)
        agent_text = font.render(f'Following Agent {self.current_followed_agent}', True, self.colors['yellow'])
        self.screen.blit(agent_text, (10, self.game_area_size[1] + 10))
    
    def render_multi_view(self):
        """Render multiple agent views"""
        # Get all agent views
        all_views = self.env.render_all(self.game_area_size)
        
        if self.view_layout == "horizontal":
            self.render_horizontal_layout(all_views)
        elif self.view_layout == "vertical":
            self.render_vertical_layout(all_views)
        elif self.view_layout == "grid":
            self.render_grid_layout(all_views)
    
    def render_horizontal_layout(self, all_views):
        """Render views in horizontal layout"""
        view_width = self.window_size[0] // len(all_views)
        view_height = self.game_area_size[1]
        
        for i, view in enumerate(all_views):
            x = i * view_width
            y = 0
            
            # Convert view to surface
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            font = pygame.font.Font(None, 24)
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_vertical_layout(self, all_views):
        """Render views in vertical layout"""
        view_width = self.game_area_size[0]
        view_height = self.window_size[1] // len(all_views)
        
        for i, view in enumerate(all_views):
            x = 0
            y = i * view_height
            
            # Convert view to surface
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            font = pygame.font.Font(None, 24)
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_grid_layout(self, all_views):
        """Render views in grid layout"""
        cols = int(np.ceil(np.sqrt(len(all_views))))
        rows = int(np.ceil(len(all_views) / cols))
        
        view_width = self.window_size[0] // cols
        view_height = self.window_size[1] // rows
        
        for i, view in enumerate(all_views):
            col = i % cols
            row = i // cols
            x = col * view_width
            y = row * view_height
            
            # Convert view to surface
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            font = pygame.font.Font(None, 20)
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_overview(self):
        """Render overview of all agents"""
        # Use the first agent's view as overview
        self.env.switch_player(0)
        image = self.env.render(self.game_area_size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render game area border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (0, 0, self.game_area_size[0], self.game_area_size[1]), 2)
        
        # Render overview label
        font = pygame.font.Font(None, 36)
        overview_text = font.render('Overview (Agent 0 Perspective)', True, self.colors['cyan'])
        self.screen.blit(overview_text, (10, self.game_area_size[1] + 10))
    
    def render_sidebar(self):
        """Render the sidebar with agent information"""
        sidebar_x = self.window_size[0] - self.sidebar_width + 10
        
        # Title
        font_large = pygame.font.Font(None, 36)
        title = font_large.render('Multi-View GUI', True, self.colors['white'])
        self.screen.blit(title, (sidebar_x, 10))
        
        # View mode info
        font = pygame.font.Font(None, 24)
        view_text = font.render(f'View Mode: {self.current_view_mode}', True, self.colors['cyan'])
        self.screen.blit(view_text, (sidebar_x, 50))
        
        if self.current_view_mode == "single":
            follow_text = font.render(f'Following: Agent {self.current_followed_agent}', True, self.colors['yellow'])
            self.screen.blit(follow_text, (sidebar_x, 80))
        
        layout_text = font.render(f'Layout: {self.view_layout}', True, self.colors['cyan'])
        self.screen.blit(layout_text, (sidebar_x, 110))
        
        # Step counter
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, self.colors['white'])
        self.screen.blit(step_text, (sidebar_x, 140))
        
        # Progress bar
        progress = self.current_step / self.max_steps
        bar_width = 300
        bar_height = 20
        bar_x = sidebar_x
        bar_y = 170
        
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
        self.render_status_indicators(sidebar_x, 500)
    
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
            
            # Highlight current followed agent
            if self.current_view_mode == "single" and i == self.current_followed_agent:
                color = self.colors['cyan']
            
            agent_text = font.render(f'Agent {i} ({agent_type}): {status}', True, color)
            self.screen.blit(agent_text, (x, y_offset))
            
            # Current operation
            if hasattr(agent, 'op') and agent.op:
                op_text = font.render(f'  Op: {agent.op}', True, self.colors['light_gray'])
                self.screen.blit(op_text, (x + 10, y_offset + 20))
            
            y_offset += 50
    
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
            
            # Highlight current view mode
            if button_id == f'{self.current_view_mode}_view':
                color = self.colors['green']
            
            # Highlight current followed agent
            if button_id == f'follow_agent_{self.current_followed_agent}':
                color = self.colors['green']
            
            # Draw button
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['white'], button['rect'], 2)
            
            # Draw text
            text = font.render(button['text'], True, self.colors['black'])
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
    
    def render_input_interface(self):
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
        elif event.key == pygame.K_v:
            self.switch_view_mode()
        elif event.key == pygame.K_l:
            self.switch_layout()
        elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                          pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
            # Follow specific agent
            agent_id = event.key - pygame.K_0
            if agent_id < self.n_players:
                self.current_followed_agent = agent_id
                print(f"Now following Agent {agent_id}")
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
        elif button_id == 'single_view':
            self.current_view_mode = "single"
        elif button_id == 'multi_view':
            self.current_view_mode = "multi"
        elif button_id == 'overview':
            self.current_view_mode = "overview"
        elif button_id == 'switch_layout':
            self.switch_layout()
        elif button_id.startswith('follow_agent_'):
            agent_id = int(button_id.split('_')[-1])
            if agent_id < self.n_players:
                self.current_followed_agent = agent_id
                self.current_view_mode = "single"
                print(f"Now following Agent {agent_id}")
    
    def switch_view_mode(self):
        """Switch between view modes"""
        modes = ["single", "multi", "overview"]
        current_index = modes.index(self.current_view_mode)
        self.current_view_mode = modes[(current_index + 1) % len(modes)]
        print(f"Switched to {self.current_view_mode} view mode")
    
    def switch_layout(self):
        """Switch between multi-view layouts"""
        layouts = ["horizontal", "vertical", "grid"]
        current_index = layouts.index(self.view_layout)
        self.view_layout = layouts[(current_index + 1) % len(layouts)]
        print(f"Switched to {self.view_layout} layout")
    
    def handle_action(self, action):
        """Handle game actions"""
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
        print(f"Starting Multi-View GUI with {self.n_players} agents")
        print(f"Human agents: {self.human_agent_ids}")
        print("Press 'h' for human actions, 'p' to pause/resume, 'v' to switch view, 'l' to switch layout")
        print("Press 0-9 to follow specific agent, ESC to quit")
        
        while self.running:
            self.handle_events()
            self.run_simulation_step()
            self.render()
        
        pygame.quit()
        print("Multi-View GUI closed")


def main():
    parser = argparse.ArgumentParser(description='Multi-View GUI for HumanAgent simulation')
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
    gui = MultiViewGUI(
        human_agent_ids=human_agent_ids,
        n_players=args.agent_num,
        max_steps=args.step_num
    )
    gui.fps = args.fps
    gui.run()


if __name__ == '__main__':
    main() 