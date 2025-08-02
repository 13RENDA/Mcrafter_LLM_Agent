import base64
from mimetypes import guess_type
from PIL import Image
import io
from .constants import world_item_ids, walkables, MATERIALS, ACTIONS, GENERAL_DESCRIPTION, ENV_DESCRIPTION, RESPONSE_DESCRIPTION, VITALS
from .pathfinding import PathFinding
import crafter.constants as const
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pathfinding_model = PathFinding(walkables)

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


# read a image 
textures = local_image_to_data_url("/Users/brenda/Desktop/AI_Agents/1012_6_agent/memory_system/textures.png")

#load the text from the file
def generate_inquiry(vision, description=""):
    data_url = local_image_to_data_url(vision)
    msg = [ 
            {
                "role": "system", 
                "content": "You are a helpful assistant." 
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": GENERAL_DESCRIPTION + ENV_DESCRIPTION + "The image of all materials in the game and your current view of the envionrment."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": textures,
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    },
                    {
                        "type": "text",
                        "text": description + """\n
### Instructions for response
- Please answer each question accurately and faithfully relying on current inventory, view, and situation (health stats, inventory items, and recent actions) and what you see.
- consider the past experiences, goals, and relevant expriences. What happened, what did you do, did you succeed, and can you do better?
- consider action prerequisites when make plans.\n
""",
                    },
                ]
            }
        ]
    return msg

# Not Using
# def find_most_similar_term(term, references):
#     vectorizer = TfidfVectorizer()
#     max = 0
#     most_similar = ''
#     for ref in references:
#         vectors = vectorizer.fit_transform([term, ref])
#         similarity = cosine_similarity(vectors)
#         similarity_score = similarity[0][1]
#         if similarity_score > max:
#             max = similarity_score
#             most_similar = ref
#     return most_similar, max

def print_color(*args, **kwargs):
    # get color from kwargs
    color = kwargs['color']
    # get color code
    color_code = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[30m',
    }.get(color, '\033[30m')
    print(color_code, end='')
    print(*args, end='')
    print('\033[0m')

# def safe_extract_op(response_content):
#     history = ({
#             "role": "user", 
#             "content": f"\"Could not recognize action\". " 
#     })
    
#     try:
#         op = response_content.split('#***#')
#         if len(op) >= 2:
#             op = op[-2]
#             op = op[-50:] # only take the last 50 characters
#         else:
#             print_color(f"ExtractError0: could not extract from op. {response_content} is splitted into {len(op)} pieces, setting op to noop.", color='red')
#             return 'noop', None, history
#     except:
#         print_color(f"ExtractError1: could not extract from {response_content}, setting op to noop.", color='red')
#         return 'noop', None, history
    
#     most_likely_op, _ = find_most_similar_term(op, references=ACTIONS)
#     if most_likely_op == '':
#         print_color(f"ExtractError2: could not find most similar term for {op}, setting op to noop.", color='red')
#         return 'noop', None, history
#     else:
#         print_color(f"Skill: {most_likely_op}", color='cyan')
#         if most_likely_op == 'Navigator':
#             try:
#                 rss = op.split(';')[1]
#             except:
#                 print_color(f"ExtractError3: could not extract rss from {op}, setting rss to grass.", color='red')
#                 return 'noop', None, history
            
#             most_likely_rss, _ = find_most_similar_term(rss, references=MATERIALS)
#             if most_likely_rss == '':
#                 print_color(f"ExtractError4: could not find most similar term for {rss}, setting rss to grass.", color='red')
#                 return 'noop', None, history
#             else:
#                 print_color(f"Resource: {most_likely_rss}", color='cyan')
#                 history = {
#                     "role": "user", 
#                     "content": f"Attempted \"navigate to a {most_likely_rss}\". " 
#                 }
#                 return most_likely_op, most_likely_rss, history # no history added as the action is recognized
#         else:
#             history = {
#                     "role": "user", 
#                     "content": f"Attempted \"{most_likely_op}\". " 
#                 }
#             return most_likely_op, None, history # no history added as the action is recognized

def safe_extract_op(response_content):
    op = response_content.action.final_next_action
    rss_to_collect = response_content.action.final_target_material_to_collect
    rss_to_share = response_content.action.final_target_material_to_share 
    target_id = response_content.action.final_target_agent_id
    if op == "Navigator":
        history = ({
                "role": "user", 
                "content": f"Attempted \"navigate to a {rss_to_collect}\". " 
        })
    elif op == "share":
        history = ({
                "role": "user", 
                "content": f"Attempted \"share {rss_to_share} with agent {target_id}\". " 
        })
    else:
        history = ({
                "role": "user", 
                "content": f"Attempted \"{op}\". " 
        })
    return op, rss_to_collect, rss_to_share, target_id, history


def go_and_find(info, target_rss):
    rss = world_item_ids[target_rss]
    grid, _pos = info['semantic'].T, info['player_pos']
    pos = (_pos[1], _pos[0])  # _pos is in (x, y), but grid is in (r, c)
    actions = pathfinding_model.find(grid, pos, rss)
    if actions:
        is_finished = True if len(actions) == 1 else False # 2 means one more step; 0 means not finished
        return actions[0], is_finished
    else:
        return 0, True # 0 is action for noop; 1 means finished


def go_to_something(env, rss):
    is_finished = 0 # 0 means not finished; 1 means finished; 2 means one more step
    action = 0
    num_step = 0
    done = False
    history = ({
        "role": "user", 
        "content": f"\"Failed navigate to a {rss} \". " 
    })
    while not done:
        obs, _, done, info = env.step(action)
        action, is_finished = go_and_find(info, rss)
        num_step += 1
        if is_finished:
            if action == -1:
                return obs, num_step, done, info, history
            history = ({
                "role": "user", 
                "content": f"\"Navigated to a {rss}\". "
            })
            if action != 0:
                obs, _, done, info = env.step(action) # step one more time to face the target rss
                num_step += 1
            return obs, num_step, done, info, history
    
    return obs, num_step, done, info, history
    

def safe_execute_skill(env, op, rss, step):
    if op == 'Navigator':
        obs, step_, done, info, history = go_to_something(env, rss) # return history if the navigator worked
        step += step_
    else:
        action = const.actions.index(op)
        obs, _, done, info = env.step(action)
        step += 1
        if action == 0:
            history = None
        else:
            history = {
                "role": "user", 
                "content": f"Attempted \"{op}\". " 
            }
    return obs, step, done, info, history

# def generate_image(obs, path):
#     img = Image.fromarray(obs[:48,:, :], 'RGB') # only image part
#     img.save(f'{path}.jpg')
#     stats_image = Image.fromarray(obs[48:,:, :], 'RGB') # only image part
#     stats_image.save(f'{path}_stats.jpg')
#     return img, stats_image

def get_player_vitals_and_inventory(player):
    vitals_str = ""
    inventory_str = ""
    for key, value in player.inventory.items():
        if key in VITALS:
            vitals_str += f"{key}: {value}, "
        else:
            inventory_str += f"{key}: {value}, "
    return vitals_str[:-2], inventory_str[:-2]

# save jsonl file
# def save_jsonl(path, msg):
#     with open(path, 'w') as f:
#         for entry in msg:
#             f.write(json.dumps(entry) + '\n')
#         f.write(episodic_description+procedural_summary+relevant_experiences)
#         f.write(json.dumps({"Response": f"{response_content}"}) + '\n')

# Not Using
# def save_txt(path, msg, context, relevant_experiences, response_content):
#     with open(path, 'w') as f:
#         for entry in msg:
#             if type(entry['content']) == list:
#                 for c in entry['content']:
#                     if 'text' in c.keys():
#                         f.write(c['text'])
#                         f.write("\n")
#                     else:
#                         f.write(c['image_url']['url'])
#                         f.write("\n")
#             else:
#                 f.write(entry['content'])
#                 f.write("\n")
#         f.write("\n===================== context ===================\n")
#         f.write(context)
#         f.write("\n===================== relevant_experiences ===================\n")
#         f.write(relevant_experiences)
#         f.write("\n===================== response_content ===================\n")
#         f.write(response_content)

if __name__ == "__main__":
    data_url = local_image_to_data_url("images/0.jpg")
    print(data_url)