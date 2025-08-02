from .memory_modules import SensoryMemory, EpisodicMemory, ProceduralMemory
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from .utils import *

class MemorySystem():
    def __init__(self, num_history_actions=7):
        self.experiences = []
        self.goal = ""
        # self.vision_embeddings = []
        self.history_actions = []
        self.num_history_actions = num_history_actions
        self.current_experience = None
    
    def add_experience(self, experience):
        # embeddings = experience.generate_embedding()
        # self.vision_embeddings.append(embeddings['vision_embedding'].squeeze(0).detach().cpu().numpy())
        self.experiences.append(experience)
        
    def generate_previous_timestamps_summary(self, k=1):
        previous_timestamps_summary = ""#f"\nShowing the recent {k} experiences in the memory system:\n"
        for i, experience in enumerate(self.experiences[-k:]):
            previous_timestamps_summary += f"\n### Last Experience Summary \n"
            previous_timestamps_summary += experience.episodic_memory.summary#['summary']
            previous_timestamps_summary += f"\n"
        previous_timestamps_summary += f"\n"
        return previous_timestamps_summary

    # Not using may implement in the future
    # def find_relevant_experiences(self, experience, k=3):
    #     target_experience = experience.generate_embedding()
    #     if k==0 or len(self.experiences) == 0:
    #         return "No experiences found in the memory system."

    #     exp_encoding = target_experience['vision_embedding'].squeeze(0).detach().cpu().numpy()
    #     vision_embeddings = np.array(self.vision_embeddings)
    #     cosine_sim = cosine_similarity([exp_encoding], vision_embeddings)
    #     most_similar_indices = np.argsort(cosine_sim)[0][-k:]
    #     content = f"I have found the following {k} experiences that are similar to the current situation:"
    #     for i in most_similar_indices:
    #         content += f"Experience {i}:\n" + self.experiences[i].episodic_memory.summary + "\n"
    #     return content

    def get_history_actions(self):
        return self.history_actions[-self.num_history_actions:]
    
    def __len__(self):
        return len(self.experiences)
    
class Experience:
    def __init__(self, vision_model=None, text_model=None, show=False, path=None):
        self.path = path
        # self.sensory_memory = SensoryMemory(vision_model, text_model, self.path)
        self.episodic_memory = EpisodicMemory(text_model)
        self.procedural_memory = ProceduralMemory(text_model)
        # self.sensory_embedding = None
        self.episodic_embedding = None
        self.procedural_embedding = None
        self.context_description = None
    
    def generate_embedding(self):
        # self.sensory_embedding = self.sensory_memory.generate_embedding()
        self.episodic_embedding = self.episodic_memory.generate_embedding()
        self.procedural_embedding = self.procedural_memory.generate_embedding()
        # return {**self.sensory_embedding, **self.episodic_embedding, **self.procedural_embedding}
        return {**self.episodic_embedding, **self.procedural_embedding}
    
    def generate_context_from_episodic_and_procedural_memory(self):
        episodic_description = self.episodic_memory.generate_description()
        procedural_description = self.procedural_memory.generate_description()
        self.context_description = episodic_description + procedural_description
        return self.context_description
    
    # Not using
    # def save_experience(self, msg, relevant_experiences):
    #     step = self.episodic_memory.temporal_info
    #     response_content = self.episodic_memory.final_response
    #     save_txt(f'results/images3/{step}.txt', msg, self.context_description, relevant_experiences, response_content)