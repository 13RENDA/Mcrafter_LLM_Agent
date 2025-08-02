import crafter
import crafter.constants as const

env = crafter.Env()
world_item_ids = env._world._mat_ids
world_item_ids.update({'player': len(world_item_ids),
                        'cow': len(world_item_ids) + 1, 
                        'zombie': len(world_item_ids) + 2,
                        'skeleton': len(world_item_ids) + 3,
                        'arrow': len(world_item_ids) + 4,
                        'plant': len(world_item_ids) + 5, 
                    })
# reverse the dictionary
world_ids_item = {v: k for k, v in world_item_ids.items()}
walkables = {world_item_ids[w] for w in const.walkable}
del env

ACTIONS = ["noop", "move_left", "move_right",
                "move_up","move_down","do",
                "sleep","place_stone", "place_table",
                "place_furnace","place_plant","make_wood_pickaxe",
                "make_stone_pickaxe","make_iron_pickaxe","make_wood_sword",
                "make_stone_sword","make_iron_sword", "Navigator"]
MATERIALS = ["water", "grass", "stone", 
                "path", "sand", "tree", 
                "lava", "coal", "iron", 
                "diamond", "table", "furnace", 'cow']
VITALS = ["health", "food", "drink", "energy"]
with open("descriptions/general_description.txt", "r") as file:
    GENERAL_DESCRIPTION = file.read()
with open("descriptions/env_description.txt", "r") as file:
    ENV_DESCRIPTION = file.read()
with open("descriptions/response_description.txt", "r") as file:
    RESPONSE_DESCRIPTION = file.read()
WORLD_SIZE = [64, 64]