import collections

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen

from . import memory

NONE_EXCHANGE_ITEMS = ['health', 'energy']
# Gym is an optional dependency.
try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object

class Env(BaseClass):
  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, n_players=1, seed=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    self._area = area
    self._view = view
    self._size = size
    self._reward = reward
    self._length = length
    self._seed = seed
    self._episode = 0
    self._world = engine.World(area, constants.materials, (12, 12))
    self._textures = engine.Textures(constants.root / 'assets')
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self._world, self._textures, [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    self._sem_view = engine.SemanticView(self._world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.Skeleton, objects.Arrow, objects.Plant])
    self._step = None
    self._unlocked = {}
    
    #self._player = None
    #self._last_health = None
    
    # multi player
    self.player_id = 0 # set the curr viewing player
    self.n_players = n_players
    self._alive_players_id = set() # player ids
    self.canvases = []
    self.memories = [memory.ChatMemory(id_) for id_ in range(self.n_players)]
    self._players = [None] * self.n_players
    self._last_healths = [None] * self.n_players
    self._last_inventory = [None] * self.n_players
    
    # Some libraries expect these attributes to be set.
    self.reward_range = None
    self.metadata = None

  # multiagent gui
  def switch_player(self, player_id):
    self.player_id = player_id
    
  def exchange_item(self, target_player_id, item):
    if item in NONE_EXCHANGE_ITEMS:
      print("Cannot exchange item: ", item)
      return
    if self.player_id == target_player_id:
      print("Cannot exchange item with self.")
      return
    curr_player = self._players[self.player_id]
    # curr_player_item_before = curr_player.inventory[item]
    # target_player_item_before = self._players[target_player_id].inventory[item]
    if curr_player.inventory[item] > 0:
      curr_player.inventory[item] -= 1
      self._players[target_player_id].inventory[item] += 1

    # assert curr_player_item_before == curr_player.inventory[item] + 1
    # assert target_player_item_before == self._players[target_player_id].inventory[item] - 1

  def chat(self, msg, player_ids):
    for player_id in player_ids:
      self.memories[player_id].store(msg)
  
  def show_history(self):
    print(self.memories[self.player_id])
      
  @property
  def observation_space(self):
    return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self):
    self._episode += 1
    
    self._step = 0
    # fixed world
    #self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._world.reset(seed=hash((self._seed, self._seed)) % (2 ** 31 - 1))
    self._update_time()
    
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    x, y = center
    
    for i in range(self.n_players):
      self._alive_players_id.add(i)
      self._players[i] = objects.Player(self._world, (x+i - self.n_players // 2, y))
      self._last_healths[i] = self._players[i].health
      self._last_inventory[i] = self._players[i].inventory.copy()
      self._world.add(self._players[i])

    self._unlocked = set()
    worldgen.generate_world(self._world, self._players[self.n_players // 2])
    self.canvases = []
    self.render_all()
    if self.n_players == 1:
      return self._obs()[0]
    return self._obs()
  
  def step(self, actions):
    self._step += 1
    self._update_time()
    
    player_rewards = {}
    unlocked = set()
    players_to_remove = []
    for player_id in self._alive_players_id:
      if self.n_players == 1:
        if type(actions) is list:
          action = actions[0]
        else:
          action = actions
      else:
        action = actions[player_id]
      curr_reward, is_alive = self.step_one_player(action, player_id)
      player_rewards[player_id] = curr_reward
      
      # unlocked.update(curr_unlocked)
      if is_alive == False:
        players_to_remove.append(player_id)
    
    for player_id in players_to_remove:
      self._alive_players_id.remove(player_id)
    
    for obj in self._world.objects:
      obj.update()
    if self._step % 10 == 0:
      for chunk, objs in self._world.chunks.items():
        self._balance_chunk(chunk, objs)
        
    self.render_all()
    obs = self._obs()
    
    all_dead = (len(self._alive_players_id) <= 0)
    over = self._length and self._step >= self._length
    
    # added for single task
    is_collected_diamond = [player.achievements['collect_diamond'] > 0 for player in self._players]
    done = all_dead or over or any(is_collected_diamond)
    if done:
      print("!!Done!!")
    #reward = sum(reward)
    #if unlocked:
    #  before = len(self._unlocked)
    #  self._unlocked |= unlocked
    #  after = len(self._unlocked)
    #  reward += max(1.0, after - before)
      
    info = []
    reward = []
    for id_, p in enumerate(self._players):
      reward.append(player_rewards[id_] if id_ in player_rewards.keys() else 0)
      info_p = {
        'id': id_,
        'inventory': p.inventory.copy(),
        'achievements': p.achievements.copy(),
        'sleeping': p.sleeping,
        'discount': 1 - float(p.health <= 0),
        'semantic': self._sem_view(),
        'player_pos': p.pos,
        'player_facing': p.facing,
        'reward': reward[-1],
        'dead': p.health <= 0,
        'unlocked': unlocked,
        'action': p.action,
        'view': self._view,
      }
      info.append(info_p)
    #if not self._reward:
    #  reward = 0.0
      
    if self.n_players == 1:
      return obs[0], reward[0], done, info[0]
    return obs, reward, done, info
    
  def step_one_player(self, action, player_id):
    curr_player = self._players[player_id]
    
    curr_player.action = constants.actions[action]
    #for obj in self._world.objects:
    #  if curr_player.distance(obj) < 2 * max(self._view):
    #    obj.update()
    
    #if self._step % 10 == 0:
    #  for chunk, objs in self._world.chunks.items():
    #    self._balance_chunk(chunk, objs)
    # updated
    curr_player_reward = (curr_player.health - self._last_healths[player_id]) # / 10
    self._last_healths[player_id] = curr_player.health
    
    # task_difficulties = {
    #     # first time complete the task
    #     'collect_wood': 2, 'collect_sapling': 2, 'place_plant': 5, 'eat_plant': 10, 'collect_stone': 5, 'make_wood_pickaxe': 2, 'make_wood_sword': 2,
    #     'collect_coal': 10, 'collect_iron': 10, 'make_stone_pickaxe': 10, 'make_stone_sword': 10, 'place_table': 20, 'place_furnace': 20, 'collect_drink': 2, 
    #     'wake_up': 5, 'make_iron_pickaxe': 7, 'make_iron_sword': 7, 'eat_cow': 5, 'collect_diamond': 10, 'place_stone': 1, 'defeat_zombie': 8, 'defeat_skeleton': 8,
    #     # subsequent tasks
    #     'wood': 2, 'stone': 4, 'coal': 6, 'iron': 6, 'diamond': 30,
    # }
    # task_difficulties = {
    #     # first time complete the task
    #     'collect_wood': 1, 'collect_sapling': 1, 'place_plant': 1, 'eat_plant': 1, 'collect_stone': 1, 'make_wood_pickaxe': 3, 'make_wood_sword': 3,
    #     'collect_coal': 3, 'collect_iron': 3, 'make_stone_pickaxe': 4, 'make_stone_sword': 4, 'place_table': 3, 'place_furnace': 3, 'collect_drink': 1, 
    #     'wake_up': 1, 'make_iron_pickaxe': 3, 'make_iron_sword': 3, 'eat_cow': 2, 'collect_diamond': 1, 'place_stone': 1, 'defeat_zombie': 5, 'defeat_skeleton': 5,
    #     # subsequent tasks
    #     'wood': 1, 'stone': 2, 'coal': 3, 'iron': 3, 'diamond': 10,
    # }
    
    task_difficulties = {
        # first time complete the task
        'collect_wood': 1, 'collect_sapling': 0, 'place_plant': 0, 'eat_plant': 0, 'collect_stone': 4, 'make_wood_pickaxe': 3, 'make_wood_sword': 0,
        'collect_coal': 4, 'collect_iron': 6, 'make_stone_pickaxe': 5, 'make_stone_sword': 1, 'place_table': 2, 'place_furnace': 5, 'collect_drink': 1, 
        'wake_up': 1, 'make_iron_pickaxe': 7, 'make_iron_sword': 0, 'eat_cow': 1, 'collect_diamond': 10, 'place_stone': 1, 'defeat_zombie': 0, 'defeat_skeleton': 0,
        # subsequent tasks
        'wood': 1, 'stone': 1, 'coal': 1, 'iron': 1, 'diamond': 1,
    }

    unlocked = {
      name for name, count in curr_player.achievements.items()
      if count > 0 and name not in self._unlocked}

    if action == 0: # if noop
     curr_player_reward -= 1
    elif action < 5:
     curr_player_reward += 0.1
      
    if unlocked:
      self._unlocked |= unlocked
      # print("Unlocked: ", unlocked)
      for t in unlocked:
        curr_player_reward += task_difficulties[t] * 2
     
    # find the difference between the two dictionaries
    # diff_inventory = {k: curr_player.inventory[k] - self._last_inventory[player_id][k] for k in curr_player.inventory if curr_player.inventory[k] != self._last_inventory[player_id][k]}
    # for k, v in diff_inventory.items():
    #   if k in task_difficulties.keys() and v > 0:
    #     curr_player_reward += task_difficulties[k]
    self._last_inventory[player_id] = curr_player.inventory.copy()
    
    # remove if health is 0
    is_alive = True
    if curr_player.health <= 0:
      self._world.remove(curr_player)
      is_alive = False
    return curr_player_reward, is_alive
    #return curr_player_reward, curr_unlocked, is_alive

  def render_one_player(self, player_id, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    
    curr_player = self._players[player_id]
    local_view = self._local_view(curr_player, unit)
    item_view = self._item_view(curr_player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))
  
  def render_all(self, size=None):
    self.canvases = []
    for player_id in range(len(self._players)):
      self.canvases.append(self.render_one_player(player_id, size))    
    return self.canvases
  
  def render(self, size=None):
    self.render_all(size)
    return self.canvases[self.player_id]

  def _obs(self):
    return self.canvases

  """def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    
    curr_player = self._players[self.player_id]
    local_view = self._local_view(curr_player, unit)
    item_view = self._item_view(curr_player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    return self.render()"""

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    progress = (self._step / 300) % 1 + 0.3
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self._world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
    light = self._world.daylight
    # for player_id in self._alive_players_id:
    #   curr_player = self._players[player_id]
    #   self._balance_object(
    #       chunk, objs, objects.Zombie, 'grass', 6, 0, 0.0, 0.4,
    #       lambda pos: objects.Zombie(self._world, pos, curr_player),
    #       lambda num, space: (
    #           0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
    #   self._balance_object(
    #       chunk, objs, objects.Skeleton, 'path', 7, 7, 0.0, 0.1,
    #       lambda pos: objects.Skeleton(self._world, pos, curr_player),
    #       lambda num, space: (0 if space < 6 else 1, 2))
    
    # no need go in forloop
    self._balance_object(
       chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
       lambda pos: objects.Cow(self._world, pos),
       lambda num, space: (0 if space < 30 else 1, 1.5 + light))

  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    
    xmin, xmax, ymin, ymax = chunk
    random = self._world.random
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    mask = self._world.mask(*chunk, material)
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      i = random.randint(0, len(xs))
      pos = np.array((xs[i], ys[i]))
      empty = self._world[pos][1] is None
      # modified
      away = True
      for p in self._players:
        _away = p.distance(pos) >= span_dist
        away = away and _away
      if empty and away:
        #print(ctor(pos))
        self._world.add(ctor(pos))
          
      #away = self._player.distance(pos) >= span_dist or self._player2.distance(pos) >= span_dist
      #if empty and away:
      #  self._world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      #away = self._player.distance(obj.pos) >= despan_dist and self._player2.distance(obj.pos) >= despan_dist
      away = True
      for p in self._players:
        _away = p.distance(obj.pos) >= despan_dist
        away = away and _away
      if away:
        self._world.remove(obj)