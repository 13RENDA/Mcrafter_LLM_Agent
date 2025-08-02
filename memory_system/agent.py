from .memory_system import MemorySystem, Experience
# import enum
from enum import Enum
from .knowledge_graph import LongTermMemory, SemanticMemory, WorkingMemory
from .backbone_models.vision_model import VisionModel
from .llm_api import TextModel, get_completion
import torch
from .utils import generate_inquiry, safe_extract_op
from concurrent.futures import ThreadPoolExecutor
# vision_model = VisionModel()
# vision_model.load_state_dict(torch.load("memory_system/backbone_models/vision_model.pth"))
# text_model = TextModel()

class ActionStatus(Enum):
    IN_PROGRESS = 0
    ALMOST_DONE = 1
    DONE = 2
    INTERRUPTED = 3
    
class Agent:
    def __init__(self, id, kg_update_freq=10):
        self.memory_system = MemorySystem(num_history_actions=7)
        self.id = id
        self.action_status = ActionStatus.DONE
        self.op = 'noop'
        self.action = 'noop'
        self.rss_to_collect = ''
        self.rss_to_share = ''
        self.summary = ""
        self.target_agent_id = -1
        self.kg_update_freq = kg_update_freq
        self.long_term_mem = LongTermMemory(path="results", recent_experience_num=kg_update_freq)
        self.replay = []
        self.text_model = TextModel()
        
        self.wm_content = ""
        self.think_context = ""
        # global communication
        self.table_avaliable = False
        self.furnace_avaliable = False
    
    def update_crafting_station_status(self, station_name):
        if station_name == 'table':
            self.table_avaliable = True
        elif station_name == 'furnace':
            self.furnace_avaliable = True
            
    def __len__(self):
        return len(self.memory_system)
    
    def update_action(self, action):
        self.action = action
    
    def update_action_status(self, status):
        self.action_status = status
    
    def update_current_skill(self, op, rss_to_collect, rss_to_share, target_agent_id):
        self.op = op
        self.rss_to_collect = rss_to_collect
        self.rss_to_share = rss_to_share
        self.target_agent_id = target_agent_id
    
    def update_state(self, obs, step, env, info, episode_number, episode_timestep):
        self.obs = obs
        self.step = step
        self.env = env
        self.info = info
        self.replay.append({'step': step} | info['achievements'] | info['inventory'])
        self.episode_number = episode_number
        self.episode_timestep = episode_timestep
    
    def create_experience(self):
        # pre experience
        obs, step, env, info = self.obs, self.step, self.env, self.info
        self.experience = Experience(None, self.text_model, path=f"results/agent_{self.id}")
        # self.experience.sensory_memory.set_memory(vision=obs, step=step, show=False)
        self.experience.procedural_memory.set_memory(self.memory_system.get_history_actions())
        self.experience.episodic_memory.set_memory(player=env._players[self.id],
                                            temporal_info=step,
                                            scene_semantic=info['semantic'],
                                            episode_number=self.episode_number,
                                            episode_timestep=self.episode_timestep)
        context = self.experience.generate_context_from_episodic_and_procedural_memory()
        self.experience.generate_embedding()
        #relevant_experiences = self.memory_system.find_relevant_experiences(self.experience, k=0)
        wm_content = ""
        self.wm_content = ""
        if len(self.memory_system) > 0:
            long_term_mem = LongTermMemory(path=f"results/agent_{self.id}", recent_experience_num=len(self.memory_system))
            if self.table_avaliable:
                long_term_mem.semantic_memory.update_crafting_station_status("table")
            if self.furnace_avaliable:
                long_term_mem.semantic_memory.update_crafting_station_status("furnace")
                
            long_term_mem.update_memory(self.memory_system.experiences)
            long_term_mem.update_knowledge_graph()
            working_memory = WorkingMemory(long_term_mem.G)
            summaries = working_memory.generate_working_memory_summaries()
            long_term_mem.plot_fancy_knowledge_graph(summaries)
            
            wm = summaries[list(summaries)[-1]]
            
            craft_station_summary = (
                f"\n### Crafting Station Availability:\n"
                f"  - Table: {'placed alerady' if self.table_avaliable else 'still needs to be placed'}.\n"
                f"  - Furnace: {'placed alerady' if self.furnace_avaliable else 'still needs to be placed'}.\n"
            )
            
            LTG_preq, CG_preq = long_term_mem.semantic_memory.check_goal(
                                            wm['long_term_goal'], 
                                            wm['current_goal'], 
                                            self.experience.episodic_memory.inventory, 
                                            {'facing': self.experience.episodic_memory.describe_facing_object()})
            prerequisites_summary = (
                f"\n### Prerequisites Check:\n"
                f"  - For long-term goal: {LTG_preq}\n"
                f"  - For immediate goal: {'should work on long-term goal now.' if ('Ready' in LTG_preq or 'everything is ready' in LTG_preq) else CG_preq}\n"
            )
            
            efforts_summary = (
                f"- To satisify the prerequiste, you made the following efforts recently towards {wm['current_goal']}: \n" 
                f"  *{'; in '.join(wm['efforts_so_far'])}*.\n" 
                f"- Past Accomplishments: You have completed the following tasks: {', '.join(wm['past_events'])}.\n"
            )
            
            # might be confusing. quick fix: self.wm_content only wants prerequisites_summary
            wm_content = craft_station_summary + prerequisites_summary + efforts_summary # past_summary + 
            self.wm_content = prerequisites_summary
            
        previous_timestamps_summary = self.memory_system.generate_previous_timestamps_summary(k=1)
        self.context = previous_timestamps_summary + context + wm_content
        self.context = self.context.replace("next", "").replace("Next", "")
        return self.context

    def think(self, context):
        self.think_context = context
        msg = generate_inquiry(vision=f'results/agent_{self.id}/{self.step}.png', 
                                     description=context)
        self.thoughts = get_completion(msg)
        return self.thoughts
    

    def consolidate_experience(self, final_response):
        self.experience.episodic_memory.final_response = final_response#.summary
        self.summary = self.experience.episodic_memory.generate_summary()
        self.memory_system.add_experience(self.experience)
        op, rss_to_collect, rss_to_share, target_agent_id, extract_history = safe_extract_op(final_response)
        if extract_history:
            self.memory_system.history_actions.append(extract_history)
        return op, rss_to_collect, rss_to_share, target_agent_id

    # @staticmethod
    def process_agent(agent, agents_contexts, info):
        if agent.action_status != ActionStatus.IN_PROGRESS and not info[agent.id]['sleeping']:
            # Skip HumanAgent processing - they are removed from pipeline
            if hasattr(agent, 'is_human') and agent.is_human:
                # HumanAgent is disabled - return None thoughts
                return {"id": agent.id, "thoughts": None}
            else:
                # Regular AI agent processing
                return {"id": agent.id, "thoughts": agent.think(agents_contexts[agent.id])}
        return {"id": agent.id, "thoughts": None} # In case the agent does not think    

class HumanAgent(Agent):
    """
    A subclass of Agent that allows a human user to manually select operations and participate in communication.
    """
    def __init__(self, id, kg_update_freq=10):
        super().__init__(id, kg_update_freq)
        self.is_human = True

    def choose_action(self):
        print(f"\n[HumanAgent {self.id}] Please select your operation:")
        print("Available operations: 'Navigator', 'share', 'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep'")
        print("Available resources: 'wood', 'stone', 'coal', 'iron', 'diamond', 'not_applicable'")
        
        op = input("Enter operation type: ").strip()
        if not op:
            op = 'noop'
            
        rss_to_collect = input("Enter resource to collect (or 'not_applicable'): ").strip()
        if not rss_to_collect:
            rss_to_collect = 'not_applicable'
            
        rss_to_share = input("Enter resource to share (or 'not_applicable'): ").strip()
        if not rss_to_share:
            rss_to_share = 'not_applicable'
            
        try:
            target_agent_id = int(input("Enter target agent id (or -1 if not applicable): "))
        except ValueError:
            target_agent_id = -1
            
        self.update_current_skill(op, rss_to_collect, rss_to_share, target_agent_id)
        print(f"[HumanAgent {self.id}] Action set: op={op}, collect={rss_to_collect}, share={rss_to_share}, target_agent_id={target_agent_id}")

    def communicate(self, message, agents):
        """
        Send a message to other agents (simple broadcast for demonstration).
        """
        print(f"[HumanAgent {self.id}] Sending message: {message}")
        for agent in agents:
            if agent.id != self.id:
                if hasattr(agent, 'receive_message'):
                    agent.receive_message(message, self.id)

    def receive_message(self, message, from_id):
        print(f"[HumanAgent {self.id}] Received message from Agent {from_id}: {message}")    