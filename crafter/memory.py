class ChatMemory:
    def __init__(self, player_id):
        self.player_id = player_id
        self.memory = []

    def store(self, msg):
        self.memory.append(msg)
        
    def __str__(self):
        return f"This is the memory of agent {self.player_id}. \nThe latest 5 conversations are: ###{self.memory[-5:]}.###"