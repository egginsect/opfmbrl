import abc

def universe(world, optimize_world_model, reward_function,
             optimize_policy, day_length, callback=None):
    def live_one_day(world_model, policy):
        experience = toil(world, world_model, policy)
        new_world_model = optimize_world_model(world_model, experience)
        new_policy = optimize_policy(dream_world(new_world_model), policy)
        return new_world_model, new_policy

    def toil(world, world_model, policy):
        def plod(belief, state):
            action = policy(belief)
            new_state, rendering = world.next_state(state, action)
            new_belief = world_model.update_belief(belief, action, rendering)
            return new_belief, new_state, (action, rendering)

        initial_state = world.new_day()
        initial_belief = world_model.initial_belief()
        final_belief, experience = \
            run(unit(initial_belief, initial_state), (plod for t in range(day_length)))

        if callback:
            actions, renderings = zip(*experience)
            callback(states=world.trajectory[1:], renderings=renderings, actions=actions)

        return experience

    def dream_world(world_model):
        def dream(test_policy):
            experience = toil(world_model, world_model, test_policy)
            return sum(map(reward_function, experience))

        return dream

    # the next three functions make up the 'memory monad'

    def unit(belief, state):
        return belief, state, []

    def bind(result, step):
        belief, state, experience = result
        new_belief, new_state, event = step(belief, state)
        return new_belief, new_state, experience + [event]

    def run(result, steps):
        for step in steps:
            result = bind(result, step)
        return result

    return live_one_day

class World(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def new_day(self):
        # Returns a state
        pass

    @abc.abstractmethod
    def next_state(self, state, action):
        # Returns new_state, rendering
        # In the case of a real world, new_state is a dummy just to keep things functional
        pass

class WorldModel(World):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initial_belief(self):
        # returns initial state distribution
        pass

    @abc.abstractmethod
    def update_belief(self, belief, action, rendering):
        # Returns a new belief
        pass
