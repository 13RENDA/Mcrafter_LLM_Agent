from memory_system.llm_api import get_completion
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyvis.network import Network
import os
import memory_system.utils as utils
from concurrent.futures import ThreadPoolExecutor
from memory_system.llm_api import get_completion
import memory_system.utils as utils
import networkx as nx
from typing import List, Dict, Any, Tuple
from memory_system.utils import print_color

class SemanticMemory:
    def __init__(self):
        self.crafting_recipes = {
            "collect_cow": {"facing": "cow"},
            "collect_drink": {"facing": "water"},
            "collect_wood": {"facing": "tree"},
            "collect_stone": {"facing": "stone", "wood_pickaxe": 1},
            "collect_coal": {"facing": "coal", "wood_pickaxe": 1},
            "collect_iron": {"facing": "iron", "stone_pickaxe": 1},
            "collect_diamond": {"facing": "diamond", "iron_pickaxe": 1},
            "obtain_diamond": {"facing": "diamond", "iron_pickaxe": 1},
            "place_table": {"facing": "grass", "wood": 2},
            "place_furnace": {"facing": "grass", "stone": 4},
            "make_wood_pickaxe": {"facing": "table", "wood": 1},
            "make_stone_pickaxe": {"facing": "table", "stone": 1, "wood": 1},
            "make_iron_pickaxe": {"facing": "furnace", "iron": 1, "coal": 1, "wood": 1},
        }
        self.placed_a_table = False
        self.placed_a_furnace = False
        
    def update_crafting_station_status(self, crafting_station_name):
        if crafting_station_name == "table":
            self.placed_a_table = True
        elif crafting_station_name == "furnace":
            self.placed_a_furnace = True
        
    def _check_goal(self, goal, inventory):
        # Get the required materials for the given goal
        required_materials = self.crafting_recipes.get(goal, None)
        
        if goal == 'help_agent':
            return f"Goal 'help_agent' requires you to help the agent immediately if you do have the material in your inventory."
        if goal == "share":
            return f"Goal 'share' will work if and only if you have the material in your inventory. Double check!"
        
        if goal == "make_wood_pickaxe":
            if inventory.get('wood_pickaxe', 0) > 0:
                return f"Wood pickaxe is already made."
        if goal == "make_stone_pickaxe":
            if inventory.get('stone_pickaxe', 0) > 0:
                return f"Stone pickaxe is already made."
        if goal == "make_iron_pickaxe":
            if inventory.get('iron_pickaxe', 0) > 0:
                return f"Iron pickaxe is already made."
        
        if required_materials is None:
            return f"Goal '{goal}' is not recognized."
        
        missing_items = {}
        is_facing_the_only_missing_item = True
        for item, requirement in required_materials.items():
            if item == "facing" and requirement != inventory['facing']:
                missing_items[item] = f"everything is ready, should navigate to {requirement}; "
                if requirement == 'table' and not self.placed_a_table:
                    missing_items[item] += f" need to place a {requirement} first, and Missing {2 - inventory.get('wood', 0)} wood, and navigate to grass to place it;"
                elif requirement == 'furnace' and not self.placed_a_furnace:
                    missing_items[item] += f" need to place a {requirement} first, and Missing {4 - inventory.get('stone', 0)} stone, and navigate to grass to place it;"
                
            # Check if the item exists in the inventory and if there is enough of it
            
            elif inventory.get(item, 0) < requirement:
                is_facing_the_only_missing_item = False
                missing_items[item] = f"Missing {requirement - inventory.get(item, 0)} {item}; "
        
        if not missing_items:
            return f"*Ready* '{goal}'"
        else:
            if is_facing_the_only_missing_item:
                missing = " ".join(missing_items.values())
            else:
                if 'facing' in missing_items:
                    del missing_items['facing']
                missing = " ".join(missing_items.values())
            return f"*Cannot complete* '{goal}': {missing}"
    
    def check_goal(self, LTG, CG, inventory, facing):
        if type(inventory) != dict:
            inventory_items = (inventory.split(', '))[1:]
            inventory = {
                item.split(': ')[0]: int(item.split(': ')[1])
                for item in inventory_items if int(item.split(': ')[1]) > 0
            }
        inventory |= facing # merge dict
        return self._check_goal(LTG, inventory), self._check_goal(CG, inventory)

class LongTermMemory:
    def __init__(self, path="", recent_experience_num=10):
        self.semantic_memory = SemanticMemory()
        self.recent_episoidic_buffer = []
        self.recent_experience_num = recent_experience_num
        self.G = None
        self.current_goal = None
        self.current_task = None
        self.event_nodes = []
        self.event_counter = 1
        self.kg_count = 0 # knowledge graph counter
        self.path = path
        # check if save path exists and create if not
        if self.path != "":
            if not os.path.exists(self.path):
                os.makedirs(self.path)
    
    def update_memory(self, experiences):
        self.experiences = experiences
    
    def update_knowledge_graph(self):
        self.kg_count += 1
        # get tqdm progress for each step
        self.recent_episoidic_buffer = []
        # exececute functions in a list in order
        functions = [
            self.extract_goal_and_relationships, 
            self.generate_knowledge_graph, 
            # self.generate_event_level_summaries, 
            # self.generate_community_level_summaries, 
            # self.plot_fancy_knowledge_graph
            ]
        
        for func in functions:
            func()
    
    def extract_goal_and_relationships(self):
        self.processed_summaries = []
        for experience in self.experiences[-self.recent_experience_num:]:
            final_response = experience.episodic_memory.final_response
            reflection, goal, action = final_response.reflection, final_response.goal, final_response.action
            # Extract necessary information
            step_data = {
                'episode': final_response.epsiode_number,
                'timestep': final_response.timestep,
                'inventory': experience.episodic_memory.inventory,
                'past_events': final_response.past_events,
                'summary': final_response.summary,
                'current_facing_direction': final_response.current_facing_direction,
                # 'vision_matrix': experience.sensory_memory.vision,
                # 'vision': reflection.vision,
                'last_action': reflection.last_action,
                'last_action_result': reflection.last_action_result,
                'long_term_goal': goal.long_term_goal,
                'long_term_goal_progress': goal.long_term_goal_progress,
                'current_goal_reason': goal.current_goal_reason,
                'current_goal': goal.current_goal.value,
                'current_goal_status': goal.current_goal_status,
                'action': action.final_next_action.value,
                'action_destination': action.final_target_material_to_collect.value,
                'action_share_destination': action.final_target_material_to_share.value,
                'action_target_agent_id': action.final_target_agent_id,
                'action_reason': action.next_action_reason,
            }
            self.processed_summaries.append(step_data.copy())
            self.event_counter += 1
        return self.processed_summaries
    
    def generate_knowledge_graph(self):
        if self.G is None:
            self.G = nx.DiGraph()
            self.event_counter = 1
            # Initialize previous variables
            self.previous_step_node = None
            self.previous_goal_node = None
            self.previous_goal = None
            self.previous_long_term_goal_node = None
            self.previous_long_term_goal = None
            # Counters for unique identification
            self.goal_instance_counter = 0
            self.long_term_goal_instance_counter = 0
            self.previous_inventory = None
            self.previous_episode = None
            # Initialize current long-term goal node
            self.current_long_term_goal_node = None

        for entry in self.processed_summaries:
            episode = entry['episode']
            if episode != self.previous_episode:
                self.previous_step_node = None
                self.previous_goal_node = None
                self.previous_goal = None
                self.previous_long_term_goal_node = None
                self.previous_long_term_goal = None
                self.previous_inventory = None
                # Update the previous_episode
                self.previous_episode = episode
                # Reset goal instance counters for the new episode
                self.goal_instance_counter = 0
                self.long_term_goal_instance_counter = 0
                # Reset current long-term goal node
                self.current_long_term_goal_node = None

            step_num = entry['timestep']
            step_node = f"Episode {episode} Step {step_num}"
            # Normalize strings
            long_term_goal = entry['long_term_goal'].strip().lower().rstrip('.')
            current_goal = entry['current_goal'].strip().lower().rstrip('.')

            # Process the current inventory
            inventory_items = (entry['inventory'].split(', '))[1:]
            current_inventory = {
                item.split(': ')[0]: int(item.split(': ')[1])
                for item in inventory_items if int(item.split(': ')[1]) > 0
            }
            # Perform the semantic memory check
            # LTG and CG
            semantic_memory_check_results = self.semantic_memory.check_goal(
                long_term_goal,
                current_goal,
                current_inventory,
                {'facing': entry['current_facing_direction']}
            )
            # Add the semantic memory check to the entry data
            entry['semantic_memory_check_LTG'], entry['semantic_memory_check_CG'] = semantic_memory_check_results

            # Check if the long-term goal has changed
            if long_term_goal != self.previous_long_term_goal:
                # Create a new long-term goal node
                self.long_term_goal_instance_counter += 1
                long_term_goal_node = f"Episode {episode} Long Term Goal {self.long_term_goal_instance_counter}: {long_term_goal}"
                self.G.add_node(long_term_goal_node, type='long_term_goal')
                # Connect previous long-term goal node to current long-term goal node
                if self.previous_long_term_goal_node:
                    self.G.add_edge(self.previous_long_term_goal_node, long_term_goal_node)
                # Update current long-term goal node
                self.current_long_term_goal_node = long_term_goal_node
            else:
                # Use the same long-term goal node
                long_term_goal_node = self.current_long_term_goal_node

            # Check if the current goal has changed
            if self.previous_goal != current_goal or current_inventory != self.previous_inventory:
                # Increment goal instance counter
                self.goal_instance_counter += 1

                # Create a unique goal node label
                current_goal_node = f"Episode {episode} Goal {self.goal_instance_counter}: {current_goal}"
                self.G.add_node(current_goal_node, type='current_goal')
                # Connect current goal node to current long-term goal node
                self.G.add_edge(long_term_goal_node, current_goal_node)
                # Connect previous goal node to current goal node
                if self.previous_goal_node:
                    self.G.add_edge(self.previous_goal_node, current_goal_node)
            else:
                # If the goal hasn't changed, use the same goal node
                current_goal_node = self.previous_goal_node

            # Create step node with semantic memory check included
            self.G.add_node(step_node, type='step', **entry)
            # Connect step node to current goal node
            self.G.add_edge(current_goal_node, step_node)

            # Connect steps sequentially if under the same current goal
            if self.previous_step_node and self.previous_goal == current_goal and current_inventory == self.previous_inventory:
                self.G.add_edge(self.previous_step_node, step_node)

            # Update previous variables
            self.previous_step_node = step_node
            self.previous_goal_node = current_goal_node
            self.previous_goal = current_goal
            self.previous_long_term_goal_node = long_term_goal_node
            self.previous_long_term_goal = long_term_goal
            self.previous_inventory = current_inventory

        return self.G

    #
    def plot_fancy_knowledge_graph(self, summaries=None):
        for goal_node, summary in summaries.items():
            # Update the node data with the working memory summary
            if self.G.has_node(goal_node):
                self.G.nodes[goal_node]['working_memory'] = summary

        G_loaded = self.G
        if G_loaded is None or len(G_loaded) == 0:
            print("Knowledge graph is empty. Cannot plot.")
            return


        # Initialize pyvis network for visualization
        net = Network(notebook=True, cdn_resources='remote', height='800px', width='100%')

        # Add all nodes with appropriate styles and tooltips
        for node, data in G_loaded.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type == 'step':
                # Existing code for step nodes remains the same
                long_term_goal = data.get('long_term_goal', '')
                long_term_goal_progress = data.get('long_term_goal_progress', '')
                current_goal = data.get('current_goal', '')
                action = data.get('action', '')
                action_destination = data.get('action_destination', '')
                action_share_destination = data.get('action_share_destination', '')
                inventory = data.get('inventory', '')
                current_facing_direction = data.get('current_facing_direction', '')
                # Process inventory for display
                inventory_items = (inventory.split(', '))[1:]
                inventory_dict = {
                    item.split(': ')[0]: int(item.split(': ')[1])
                    for item in inventory_items if int(item.split(': ')[1]) > 0
                }
                action_reason = data.get('action_reason', '')
                # Retrieve semantic memory check
                semantic_memory_check_LTG = data.get('semantic_memory_check_LTG', '')
                semantic_memory_check_CG = data.get('semantic_memory_check_CG', '')
                # Create tooltip
                title = (
                    f"\nLong-Term Goal: {long_term_goal}"
                    f"\nLTG Progress: {long_term_goal_progress}"
                    f"\nCurrent Goal: {current_goal}"
                    f"\nAction: {action}"
                    f"\nDestination: {action_destination}"
                    f"\nShare Destination: {action_share_destination}"
                    f"\nReason: {action_reason}"
                    f"\nInventory: {inventory_dict}"
                    f"\nSemantic Memory Check LTG: {semantic_memory_check_LTG}"
                    f"\nSemantic Memory Check CG: {semantic_memory_check_CG}"
                )
                net.add_node(
                    node,
                    label=node,
                    color='skyblue',
                    title=title,
                    shape='box',
                    size=15
                )
            elif node_type == 'current_goal':
                # Extract data from the current goal node
                current_goal_description = node.split(': ', 1)[-1]
                # Retrieve the working memory summary
                working_memory = data.get('working_memory', {})
                # Build the tooltip using the working memory summary
                title = f"Current Goal: {current_goal_description}"
                if working_memory:
                    # Include working memory details in the tooltip
                    title += (
                        f"\nReason: {working_memory.get('current_goal_reason', '')}"
                        f"\n### Long-Term Goal: {working_memory.get('long_term_goal', '')}"
                        f"\n### LTG Progress: {working_memory.get('long_term_goal_progress', '')}"
                        f"\n### Current Progress and Efforts So Far:"
                    )
                    # Include efforts
                    efforts = working_memory.get('efforts_so_far', [])
                    for effort in efforts:
                        title += f"\n- {effort}"
                    # Include past events
                    past_events = working_memory.get('past_events', [])
                    if past_events:
                        title += f"\n### Completed Events:"
                        for event in past_events:
                            title += f"\n- {event}"
                    # Include the semantic memory check
                    title += f"\n### Semantic Memory Check:\n{working_memory.get('semantic_memory_check', '')}"
                else:
                    # Fallback if no working memory is available
                    current_goal_reason = data.get('current_goal_reason', '')
                    title += f"Reason: {current_goal_reason}"
                net.add_node(
                    node,
                    label=node,
                    color='lightgreen',
                    title=title,
                    shape='ellipse',
                    size=15
                )
            elif node_type == 'long_term_goal':
                # Existing code for long-term goal nodes remains the same
                long_term_goal_description = node.split(': ', 1)[-1]
                long_term_goal_progress = data.get('long_term_goal_progress', '')
                long_term_goal_status = data.get('long_term_goal_status', '')
                # Create tooltip
                title = (
                    f"<b>Long-Term Goal:</b> {long_term_goal_description}<br>"
                    f"<b>Progress:</b> {long_term_goal_progress}<br>"
                    f"<b>Status:</b> {long_term_goal_status}"
                )
                net.add_node(
                    node,
                    label=node,
                    color='lightcoral',
                    title=title,
                    shape='ellipse',
                    size=20
                )
            else:
                # For any other node types
                net.add_node(
                    node,
                    label=node,
                    title='',
                    shape='ellipse',
                    size=15
                )

        # Add all edges directly from the graph
        for u, v in G_loaded.edges():
            net.add_edge(u, v)

        # Configure physics for a smoother layout
        net.set_options("""
        var options = {
        "nodes": {
            "borderWidth": 2,
            "shadow": {
            "enabled": true
            },
            "font": {
            "size": 12,
            "face": "Arial"
            }
        },
        "edges": {
            "color": {
            "inherit": true
            },
            "smooth": {
            "type": "continuous"
            },
            "arrows": {
            "to": {
                "enabled": true,
                "scaleFactor": 0.5
            }
            }
        },
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -20000,
            "springLength": 150
            },
            "minVelocity": 0.75
        }
        }
        """)

        # Show or save the interactive network
        if self.path != "":
            net.save_graph(f"{self.path}/knowledge_graph_{self.kg_count}.html")
            print_color(f"Knowledge graph saved to {self.path}/knowledge_graph_{self.kg_count}.html", color="yellow")
        else:
            net.show("fancy_knowledge_graph.html")


# Define the WorkingMemory class
class WorkingMemory:
    def __init__(self, knowledge_graph: nx.DiGraph):
        self.G = knowledge_graph

    def retrieve_working_memory(self):
        # get most recent goal node
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'current_goal':
                current_goal_node = node
                break
        # get working memory summary
        working_memory_summary = self.generate_working_memory_summaries()
        return working_memory_summary
    
    def generate_working_memory_summaries(self) -> Dict[str, Any]:
        """
        Generates a working memory summary for each current goal node in the knowledge graph.
        Returns a dictionary where keys are current goal node identifiers and values are the working memory summaries.
        """
        working_memory_summaries = {}
        # Iterate over all current goal nodes in the knowledge graph
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'current_goal':
                # The current goal node
                current_goal_node = node

                # Extract the episode number from the current goal node
                episode = self._extract_episode_number(current_goal_node)

                # Extract the long-term goal and related information
                long_term_goal_info = self._get_long_term_goal_info(current_goal_node)

                # Get efforts (summary of steps) done so far under this current goal
                efforts = self._get_efforts_under_goal(current_goal_node)

                # Get past events that have occurred before this current goal
                past_events = self._get_past_events(current_goal_node, episode)

                # Extract current goal description from node label if not in data
                current_goal_description = data.get('current_goal_description', '')
                if not current_goal_description:
                    # Assume node label is like "Episode X Goal Y: goal_description"
                    if ': ' in current_goal_node:
                        current_goal_description = current_goal_node.split(': ', 1)[-1]
                    else:
                        current_goal_description = current_goal_node

                # Get semantic memory check from the most recent step
                steps = self._get_steps_under_goal(current_goal_node)
                if steps:
                    most_recent_step = steps[-1]
                    most_recent_step_data = self.G.nodes[most_recent_step]
                    semantic_memory_check_LTG = most_recent_step_data.get('semantic_memory_check_LTG', '')
                    semantic_memory_check_CG = most_recent_step_data.get('semantic_memory_check_CG', '')
                else:
                    semantic_memory_check_LTG = ''
                    semantic_memory_check_CG = ''

                # Construct the working memory summary
                working_memory_summary = {
                    'long_term_goal': long_term_goal_info['long_term_goal'],
                    'long_term_goal_progress': long_term_goal_info['long_term_goal_progress'],
                    'long_term_goal_reason': long_term_goal_info['long_term_goal_reason'],
                    'current_goal': current_goal_description,
                    'current_goal_reason': long_term_goal_info['current_goal_reason'],
                    'efforts_so_far': efforts,
                    'past_events': past_events,
                    'semantic_memory_check_LTG': semantic_memory_check_LTG,  # Include semantic memory check
                    'semantic_memory_check_CG': semantic_memory_check_CG,  # Include semantic memory check
                }

                # Add the summary to the dictionary
                working_memory_summaries[current_goal_node] = working_memory_summary

        return working_memory_summaries

    def _extract_episode_number(self, node_label: str) -> int:
        """
        Extracts the episode number from a node label.
        Assumes node labels are in the format "Episode X ..."
        """
        try:
            if node_label.startswith('Episode '):
                parts = node_label.split(' ', 2)
                episode_number = int(parts[1])
                return episode_number
        except (IndexError, ValueError):
            pass
        # Default to episode 1 if unable to extract
        return 1

    def _get_long_term_goal_info(self, current_goal_node: str) -> Dict[str, str]:
        """
        Retrieves the long-term goal and related information associated with the current goal node.
        """
        long_term_goal_info = {
            'long_term_goal': '',
            'long_term_goal_progress': '',
            'long_term_goal_reason': '',
            'current_goal_reason': ''
        }

        # Find the long-term goal node connected to this current goal node
        for pred in self.G.predecessors(current_goal_node):
            pred_data = self.G.nodes[pred]
            if pred_data.get('type') == 'long_term_goal':
                # Extract long-term goal description from node label
                if ': ' in pred:
                    long_term_goal_description = pred.split(': ', 1)[-1]
                else:
                    long_term_goal_description = pred
                long_term_goal_info['long_term_goal'] = long_term_goal_description
                # Since we might not have 'long_term_goal_reason' directly, we'll extract it from the steps
                break

        # Get the first step under this current goal to extract additional information
        steps = self._get_steps_under_goal(current_goal_node)
        if steps:
            first_step_data = self.G.nodes[steps[0]]
            long_term_goal_info['long_term_goal_progress'] = first_step_data.get('long_term_goal_progress', '')
            long_term_goal_info['current_goal_reason'] = first_step_data.get('current_goal_reason', '')
            # Extract 'long_term_goal_reason' from the step data if available
            long_term_goal_info['long_term_goal_reason'] = first_step_data.get('long_term_goal_reason', '')

        return long_term_goal_info

    def _get_efforts_under_goal(self, current_goal_node: str) -> List[str]:
        """
        Collects a summary of steps (efforts) done so far under the given current goal node.
        """
        efforts = []
        steps = self._get_steps_under_goal(current_goal_node)
        for step in steps:
            step_data = self.G.nodes[step]
            action = step_data.get('action', '')
            action_destination = step_data.get('action_destination', '')
            action_share_destination = step_data.get('action_share_destination', '')
            last_action_result = step_data.get('last_action_result', '')
            timestep = step_data.get('timestep', '')
            # Construct a summary of the step
            if action == 'share':
                effort_summary = (
                    f"In step {timestep}: {action} {action_share_destination} to target agent. "
                )
            elif action == 'Navigator':
                effort_summary = (
                    f"In step {timestep}: {action} to {action_destination}. "
                )
            else:
                effort_summary = (
                    f"In step {timestep}: {action}. "
                )
            efforts.append(effort_summary)
        return efforts

    def _get_steps_under_goal(self, current_goal_node: str) -> List[str]:
        """
        Retrieves all step nodes under the given current goal node, sorted by timestep.
        """
        steps = []
        for succ in self.G.successors(current_goal_node):
            succ_data = self.G.nodes[succ]
            if succ_data.get('type') == 'step':
                steps.append(succ)
        # Sort the steps by timestep
        steps = sorted(steps, key=lambda x: self.G.nodes[x]['timestep'])
        return steps

    def _get_past_events(self, current_goal_node: str, episode: int) -> List[str]:
        """
        Retrieves past events (past goal descriptions) that occurred before the current goal,
        only if they belong to the same episode.
        """
        past_events = []
        # Collect past goal descriptions
        visited_goals = set()
        self._collect_past_goal_descriptions(current_goal_node, past_events, visited_goals, episode)
        # Reverse to get the correct chronological order
        past_events = past_events[::-1]
        return past_events

    def _collect_past_goal_descriptions(self, goal_node: str, past_events: List[str], visited_goals: set, episode: int):
        """
        Recursively collects past goal descriptions, only if they belong to the same episode.
        """
        if goal_node in visited_goals:
            return
        visited_goals.add(goal_node)
        for pred in self.G.predecessors(goal_node):
            pred_data = self.G.nodes[pred]
            if pred_data.get('type') == 'current_goal':
                # Extract the episode number of the predecessor goal node
                pred_episode = self._extract_episode_number(pred)
                if pred_episode == episode:
                    # Extract the goal description
                    if ': ' in pred:
                        goal_description = pred.split(': ', 1)[-1]
                    else:
                        goal_description = pred
                    past_events.append(goal_description)
                    # Recursively collect past goals
                    self._collect_past_goal_descriptions(pred, past_events, visited_goals, episode)
            elif pred_data.get('type') == 'long_term_goal':
                # Continue traversing if needed
                continue