from typing import Any
from openai import AzureOpenAI
import re
import tiktoken
from .utils import print_color
from pydantic import BaseModel, Field
from typing import Union, Literal
import time
from openai import APITimeoutError

client = AzureOpenAI(
  azure_endpoint = " ",
  api_key=" ",  
  api_version=" "
)

from enum import Enum

class ResultType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    
class ActionType(str, Enum):
    noop = "noop"
    move_left = "move_left"
    move_right = "move_right"
    move_up = "move_up"
    move_down = "move_down"
    do = "do"
    sleep = "sleep"
    place_stone = "place_stone"
    place_table = "place_table"
    place_furnace = "place_furnace"
    place_plant = "place_plant"
    make_wood_pickaxe = "make_wood_pickaxe"
    make_stone_pickaxe = "make_stone_pickaxe"
    make_iron_pickaxe = "make_iron_pickaxe"
    Navigator = "Navigator"
    share = "share"
    
class GoalType(str, Enum):
    COLLECT_WOOD = "collect_wood"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    COLLECT_STONE = "collect_stone"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    COLLECT_IRON = "collect_iron"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    COLLECT_DIAMOND = "collect_diamond"
    
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    COLLECT_COAL = "collect_coal"
    SHARE = "share"

class LongTermGoalType(str, Enum):
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    COLLECT_DIAMOND = "collect_diamond"
    HELP_AGENT = "help_agent"
    
class MaterialType(str, Enum):
    TABLE = "table"
    FURNACE = "furnace"
    GRASS = "grass"
    SAND = "sand"
    LAVA = "lava"
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    
class Reflection(BaseModel):
    vision: list[MaterialType] = Field(description="List of materials you see around you.")
    last_action: ActionType #= Field(description="Your last action.")
    last_action_result: ResultType
    last_action_result_reflection: str #= Field(description="Reflection on the last action.")
    last_action_repeated_reflection: str = Field(description="Did you repeat the last action? If so, why?")

class Goal(BaseModel):
    ultimate_goal: LongTermGoalType = Field(description="What is your ultimate goal?")
    
    long_term_goal: LongTermGoalType = Field(description="Working towards the ultimate goal, what should be your next goal?")
    long_term_goal_subgoals: str = Field(Description="What are the subgoals to complete the long term goal?")
    long_term_goal_progress: GoalType = Field(Description="What is the progress of the long term goal?")
    long_term_goal_status: ResultType
    
    current_goal: GoalType = Field(description="The current goal that you are working on.")
    current_goal_reason: str #= Field(description="Why do you choose the current goal?")
    current_goal_status: ResultType
    
class NavigationDestinationItems(str, Enum):
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    IRON = "iron"
    DIAMOND = "diamond"
    COAL = "coal"
    GRASS = "grass"
    # COW = 'cow'
    TABLE = "table"
    FURNACE = "furnace"
    NOT_APPICABLE = "not_applicable"
    
class ShareableItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
    # food
    # water
    NOT_APPLICABLE = "not_applicable"

class InventoryItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
 
class InventoryItemsCount(BaseModel):
    item: InventoryItems
    count: int

# Model for planning the next action
class NextAction(BaseModel):
    next_action: ActionType = Field(description="What is the next action you plan to take?")
    next_action_reason: str # = Field(description="Why do you think this shoud be the next action?")
    next_action_prerequisites_status: ResultType = Field(description="Are the prerequisites met?")
    next_action_prerequisites: str = Field(description="What prerequisites are not met?")
    final_next_action: ActionType = Field(description="What is your final decision on next action.")
    final_next_action_reason: str # = Field(description="Reason for choosing this action.")
    final_next_action_status: ResultType # = Field(description="What is the status of the final decision?")
    final_target_material_to_collect: NavigationDestinationItems = Field(description="Navigate to where?")
    final_target_material_to_share: ShareableItems = Field(description="Share what?")
    final_target_agent_id: int = Field(description="Which agent to share with, if applicable, or return -1.")
    
class Collaboration(BaseModel):
    target_agent_to_help: int = Field(description="Which agent should you help, if applicable?")
    target_agent_need: ShareableItems = Field(description="What does the target agent need, if applicable?")
    help_method: str = Field(description="What can you do to help the agent, if applicable?")
    can_help_now: ResultType = Field(description="Can you help the agent now? Do you have the resources in inventory?")
    being_helped_by_agent: int = Field(description="Which agent is helping you, if applicable?")
    help_method_by_agent: str = Field(description="What is the agent doing to help you, if applicable?")
    change_in_plan: str = Field(description="How does the help from the agent change your plan, if applicable?")
    
class ResponseEvent(BaseModel):
    epsiode_number: int = Field(Description="What is the current episode?")
    timestep: int = Field(Description="What is the current timestep in the episode?")
    past_events: str = Field(Description="Briefly describe the past events in the episode.")
    current_facing_direction: MaterialType
    current_inventory: list[InventoryItemsCount] = Field(Description="What is in your current inventory? Only list items with item count greater than 0.")
    collaboration: Collaboration
    reflection: Reflection
    goal: Goal
    action: NextAction
    summary: str = Field(Description=(
                                "Summarize the episode, including the timestep, long-term goal, progress, significant events, and plan. "
                                "Explain your actions, the rationale behind your decisions. Treat as if you have done the next actions aleardy. Explain your intended support for other agents (if applicable). What should come next?"
                                "Keep the summary concise and focused on key information, using *past tense* for everything as it serves as a note for future reference. Use clear and plain language. "
                                "Use PAST TENSE!!!\n"
"""
Template:
This is agent [...]. In Episode: [...] ; Timestep: [...]. My inventory contained [...]. In the past, I successfully [...]; I failed to [...]. On collaboration, [...].
I [current action: made/placed/navigated/ate/shared] a [current object] because [...]. This action succeeded/failed/was in progress, becuase it does not exist in my inventory. I planned to work towards [goal] because [...].
"""
                           )
                        )
    #                         "Ssummrize the episode,"
    #                         "including timestep, long term goal, your progress, what happened in the past, and plan. "
    #                         "Include how you plan to help the other agent, if applicable."
    #                         "What action do you decide to do, why (I decided to xxx in this step because xxx)."
    #                         "What should be the next goal and why?"
    #                         "Be clear and concise and include the most important information."
    #                         "The whole summary act as a note for the next episode, so use past tense for everything; Avoid using 'next' to describe action because it would be done by next time step.")
    # )
    
def get_completion(messages, model="gpt-4o"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model == "gpt-4o":
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=ResponseEvent,
                )
                return response.choices[0].message.parsed
            else:
                raise ValueError("Model not found")
        except (APITimeoutError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print_color(f"Timeout occurred in get_completion, retrying in 10 seconds... (attempt {attempt+1})", color="yellow")
                time.sleep(10)
            else:
                print_color("Timeout occurred in get_completion, max retries reached.", color="red")
                raise

class TextModel:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # self.tokenizer = None
    
    def __call__(self, text) -> Any:
        normalized_text = self.nomralize(text)
        token_len = self.check_token_len(text)
        if token_len > 8192:
            print_color(f"TextEmbbeding Warning: The token length of the text is {token_len}, it may cause the model to fail to generate the response.", color="red")
        return self.ada_embedding(normalized_text)
    
    def ada_embedding(self, text):
        response = client.embeddings.create(input=[text], model="text-embedding")
        return response.data[0].embedding
        # return ''
    def check_token_len(self, text):
        return len(self.tokenizer.encode(text))
        # return 0
        
    @staticmethod
    def nomralize(s) -> Any:
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s