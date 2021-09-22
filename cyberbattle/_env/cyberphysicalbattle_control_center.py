# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on CyPSA network structure"""

from ..samples.control_center import control_center
from . import cyberphysicalbattle_env


class CyberPhysicalBattleControlCenter(cyberphysicalbattle_env.CyPhyBattleEnv):
    """CyberPhysicalBattle environment based on CyPSA network structure"""

    def __init__(self, **kwargs):
        super().__init__(
            initial_environment=control_center.new_environment(),
            **kwargs)

    @ property
    def name(self) -> str:
        return f"CyberPhysicalBattleControlCenter"
