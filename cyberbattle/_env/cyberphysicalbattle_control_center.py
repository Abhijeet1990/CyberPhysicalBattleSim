# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on CyPSA network structure"""

from ..samples.control_center import control_center
from . import cyberphysicalbattle_env
from ..samples.control_center.control_center import GridEnv
import random

train_sets_random = []
for i in range(30000):
    data_set = [[5, '1', round(150*random.uniform(0.7,1.3),2)], [6, '1', round(185*random.uniform(0.7,1.3),2)], [8, '1', round(100*random.uniform(0.7,1.3),2)]]
    train_sets_random.append(['load', ['BusNum', 'LoadID', 'LoadMW'], data_set])

class CyberPhysicalBattleControlCenter(cyberphysicalbattle_env.CyPhyBattleEnv):
    """CyberPhysicalBattle environment based on CyPSA network structure"""

    def __init__(self, **kwargs):
        self.env = GridEnv(train_sets=train_sets_random)
        super().__init__(
            initial_environment=control_center.new_environment(),
            env = self.env,
            **kwargs)


    @ property
    def name(self) -> str:
        return f"CyberPhysicalBattleControlCenter"
