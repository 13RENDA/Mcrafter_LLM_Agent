import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_player_vitals_and_inventory
from .constants import WORLD_SIZE, world_ids_item
# from .llm_api import get_completion
import os

class SensoryMemory():
    def __init__(self, vision_model=None, text_model=None, path=None):
        self.vision_model = vision_model
        self.text_model = text_model
        self.obs = None
        self.vision = None
        self.stats_bar = None
        self.vision_embedding = None
        self.vision_reconstruction = None
        self.conversation = None # WAIT FOR MULTIAGENT
        self.conversation_embedding = None # WAIT FOR MULTIAGENT
        self.path = path
        # if path does not exist, create it
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
        
    def set_memory(self, vision, step, show=False):
        self.obs = vision
        self.set_vision(vision, step, show)
        
    def set_vision(self, obs, step, show=False):
        self.vision = obs[:48, :, :].astype('float32') / 255.0
        self.stats_bar = obs[49:, :, :].astype('float32') / 255.0
        
        plt.imsave(f'{self.path}/{step}.png', self.vision)
        plt.imsave(f'{self.path}/{step}_stats.png', self.stats_bar)
        if show:
            #fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            #ax[0].imshow(self.vision)
            #ax[1].imshow(self.stats_bar)
            #ax[0].set_axis_off()
            #ax[1].set_axis_off()
            plt.figure(figsize=(5, 3))
            plt.imshow(obs.astype('float32') / 255.0)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    """Didn't use, may implement in the future"""    
    def get_vision_embedding(self, reconstruct=False, show_reconstruction=False):
        
        # if self.vision_embedding is None:
        #     self.vision_embedding = torch.tensor(self.vision, dtype=torch.float32).unsqueeze(0)
        #     self.vision_embedding = self.vision_embedding.permute(0, 3, 1, 2)
        #     self.vision_embedding = self.vision_model.encoder(self.vision_embedding)
        # if reconstruct:
        #     self.render_reconstruction(show_reconstruction)
        # return self.vision_embedding
        return torch.tensor([0])
    
    """Didn't use, may implement in the future
    def render_reconstruction(self, show_reconstruction):
        if self.vision_reconstruction is None:
            self.vision_reconstruction = self.vision_model.decoder(self.vision_embedding)
            self.vision_reconstruction = self.vision_reconstruction.squeeze(0).permute(1, 2, 0).detach().numpy()
        # draw original and reconstructed images
        if show_reconstruction:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.vision)
            ax[1].imshow(self.vision_reconstruction)
            ax[2].imshow(self.stats_bar)
            plt.show()
    """ 
    def get_conversation_embedding(self):
        return None
        #if self.conversation_embedding is None:
        #    self.conversation_embedding = self.text_model(self.conversation)
        #return self.conversation_embedding
    
    def generate_embedding(self):
        return {"vision_embedding": self.get_vision_embedding(), "conversation_embedding": self.get_conversation_embedding()}
"""
json_output_example = \

Use double quotation marks for string values. Do not use single quotation marks.
example 1:
{
    "step": 1, 
    "goal": "Collect Wood", 
    "current_progress": "No progress",
    "partially_done": False, 
    "partially_done_task": "null", 
    "goal_completion": False, 
    "final_action": "Navigate to tree",
    "summary": "Initial step to collect wood, navigating to tree. Partial goal is not done yet. [AND EPISODE SUMMARY: what happened, what failed, why failed, what learned]"
}
example 2:
{
    "step": 8, 
    "goal": "Collect Wood", 
    "current_progress": "Collected  1 wood, facing grass",
    "partially_done": True, 
    "partially_done_task": "Collect 1 wood.", 
    "goal_completion": False, 
    "final_action": "do",
    "summary": "performed do to collect tree. Partial goal is done. [AND EPISODE SUMMARY: what happened, what failed, why failed, what learned]"
}

"""
# with open('descriptions/env_description.txt', 'r') as file:
#     env_description = file.read()
class EpisodicMemory():
    def __init__(self, text_model=None):
        self.text_model = text_model
        
        self.vitals = None
        self.inventory = None
        self.temporal_info = None
        self.spatial_info = None
        self.facing_direction = None
        self.facing_object = None
        
        self.scene_semantic = None
        self.final_response = None
        
        self.description = None
        self.description_embedding = None
        self.summary = None
        self.summary_embedding = None
        
        self.episode_ct = 1
    
    def set_memory(self, player, temporal_info, scene_semantic, episode_number, episode_timestep):
        self.summary = None
        self.vitals, self.inventory = get_player_vitals_and_inventory(player)
        self.temporal_info = temporal_info
        self.spatial_info = player.pos
        self.scene_semantic = scene_semantic
        self.scene_semantic_embedding = None
        self.facing_direction = player.facing
        self.episode_number = episode_number
        self.episode_timestep = episode_timestep
        
    def describe_facing_object(self):
        if self.facing_object is None:
            x, y = self.spatial_info[0] + self.facing_direction[0], self.spatial_info[1] + self.facing_direction[1]
            obj = self.scene_semantic[x, y]
            self.facing_object = world_ids_item[obj]
        return self.facing_object
    
    def generate_description(self):
        self.description = f"### Current Experience\n" + \
                           f"- This is episoide {self.episode_number}. It is currently {self.episode_timestep} steps into the episoide. You are at loaction {self.spatial_info}." + \
                           f"\n- Facing Direction: " + \
                           f"You are facing a {self.describe_facing_object()}." + \
                           f"\n- Current Health Stats: " + \
                           f"{self.vitals}." + \
                           f"\n- Current Inventory: " + \
                           f"{self.inventory}.\n"
        return self.description
    
    def generate_description_embedding(self):
        if self.description_embedding is None:
            self.description_embedding = self.text_model(self.description)
        return self.description_embedding
    
    def generate_scene_semantic_embedding(self):
        unique, counts = np.unique(self.scene_semantic, return_counts=True)
        self.scene_semantic_embedding = np.zeros(19)
        for i, u in enumerate(unique):
            self.scene_semantic_embedding[u] = counts[i]
        return self.scene_semantic_embedding
    
    # to conclude what happened
    def generate_summary(self):
        # assert self.final_response is not None, "Error genereate summary: Final response is not obtained."
        if self.final_response is None:
            self.summary = None
            return self.summary
        self.summary = self.final_response.summary
        return self.summary
        # if self.summary is None:
            # input = [   
            #             {
            #                 "role": "system", 
            #                 "content": "You are a helpful assistant." 
            #             },
            #             {
            #                 "role": "user",
            #                 "content": "Description of the environment: \n" + env_description
            #             },
            #             {
            #                 "role": "user", 
            #                 "content": "Please summarize the following content in a paragraph, clearly, comprehensively, and concisely. " + \
            #                             "1. What is the time in the episode? What happened? " + \
            #                             "2. What is the goal? Find the closest description from enviornment task list. Use the exact task name. " + \
            #                             "3. What is the current progress? "+ \
            #                             "4. Is the goal partially done? If so, what have been done? " + \
            #                             "5. If the goal is completed, did you success? Why or why not? " + \
            #                             "6. Did you learn anything new?"
            #                             "7. What is the final action taken? Why?"+ \
            #                             "Please answer in JSON format clearly with keys: step, goal, current_progress, partially_done, partially_done_task, goal_completion, final_action, and summary. The summary part should be less than 200 words." + \
            #                             "Do not add unncessary formating characters. The output should be directly convertable to json. Do not add '''json to the begining." + \
            #                             json_output_example + \
            #                             f"Content: \n{self.description + self.final_response}"
            #             }
            #         ]
            # self.summary = get_completion(input)
            # self.summary = self.summary.replace('\t', '').replace('\n', '')
            # self.summary = self.summary.replace('false', 'False').replace('true', 'True').replace('```json', '').replace('```', '').replace('json', '')
            # self.summary = self.summary.replace('null', 'None')
            # self.summary = eval(self.summary)
        return self.summary
    
    def generate_summary_embedding(self):
        if self.summary is None:
            if self.generate_summary() is None:
                self.summary_embedding = None
                return self.summary_embedding
        self.summary_embedding = self.text_model(self.summary)
        return self.summary_embedding
        # return None
    
    
    def generate_embedding(self):
        return {"description_embedding": self.generate_description_embedding(),
                "temporal_info_embedding": self.temporal_info,
                "spatial_info_embedding": self.spatial_info,
                "scene_semantic_embedding": self.generate_scene_semantic_embedding(),
                "summary_embedding": self.generate_summary_embedding(),
                } # can be None when set as target for searching as the llm have not yet generate action for current scene

# add current action
class ProceduralMemory():
    def __init__(self, text_model=None, num_past_actions=None):
        self.actions = None
        self.description = None
        self.description_embedding = None
        self.text_model = text_model
        self.num_past_actions = num_past_actions or 7
        
    def set_memory(self, actions):
        self.actions = actions
        
    def generate_description(self):
        self.description = f"\n### Past Actions\n" + \
                           f"The last {self.num_past_actions} actions are: "
        for act in self.actions:
            self.description += act['content']
        self.description += "\n"
        return self.description
        
    def generate_embedding(self):
        if self.description is None:
            self.generate_description()
        if self.description_embedding is None:
            self.description_embedding = self.text_model(self.description)
        return {"actions_description_embedding": self.description_embedding}