from importlib import import_module
import yaml
from primal_env.envs.PrimalState import PrimalState
from primal_env.envs.Trainer import Trainer
from primal_env.envs.Visualizer import Visualizer
from primal_env.envs.Simulator import BasicSimulator


def create(config_file, create_state=True):

    # Read the yml config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

        # create and configure the working memory
        # memory_config = config['working-memory']
        # memory_size = memory_config['size']
        # memory = WorkingMemory(memory_size)

        # create and configure the state
        if create_state:
            state = PrimalState(config)
        else:
            state = None

        # create and configure the trainer
        trainer_config = config['trainer']
        # trainer = Trainer(trainer_config, simulator, state)
        trainer = Trainer(trainer_config, state)

        # create and configure the simulator
        # TODO: need to get rid of this black magic
        simulator_config = config['simulator']
        # simulator_type = simulator_config['type']
        # module_path, class_name = simulator_type.rsplit('.', 1)
        # module = import_module(module_path)
        # simulator = getattr(module, class_name)(state, **simulator_config)
        width = simulator_config['width']
        height =simulator_config['height']
        simulator = BasicSimulator(trainer, state, width, height)

        # create a model

        # model = ActorCriticModel(trainer_config)



        # create and configure the visualizer
        visualizer = Visualizer(config['visualization'])

        # tester = PrimalTester(trainer, simulator, state, model)

    return trainer, simulator, state, visualizer
