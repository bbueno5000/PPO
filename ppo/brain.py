class BrainInfo:

    def __init__(self, state, memory=None, reward=None, agents=None, local_done=None, action=None):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.agents = agents
        self.local_done = local_done
        self.memories = memory
        self.previous_actions = action
        self.rewards = reward
        self.states = state

class BrainParameters:

    def __init__(self, brain_name, brain_param):
        """
        Contains all brain-specific parameters.

        :param brain_name: Name of brain.
        :param brain_param: Dictionary of brain parameters.
        """
        self.action_space_size = brain_param['actionSize']
        self.action_space_type = brain_param['actionSpaceType']
        self.brain_name = brain_name
        self.number_observations = 0
        self.state_space_size = brain_param['stateSize']
        self.state_space_type = brain_param['stateSpaceType']

    def __str__(self):
        return '''Unity brain name: {0}
        State space type: {1}
        State space size (per agent): {2}
        Action space type: {3}
        Action space size (per agent): {4}'''.format(self.brain_name,
                                                     self.state_space_type,
                                                     str(self.state_space_size),
                                                     self.action_space_type,
                                                     str(self.action_space_size))
