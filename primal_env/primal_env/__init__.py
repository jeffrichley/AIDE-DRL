import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Primal-v0',
    entry_point='primal_env.envs:EasyPrimalEnv'
)

register(
    id='Primal-Single-v0',
    entry_point='primal_env.envs:SingleAgentPrimalEnv'
)

register(
    id='Primal-FourEnemyFourAlly-v0',
    entry_point='primal_env.envs:FourEnemyFourAllyAgentPrimalEnv'
)

register(
    id='Primal-FourEnemyFourAllyFiftyWall-v0',
    entry_point='primal_env.envs:FourEnemyFourAllyFiftyWalllsAgentPrimalEnv'
)


