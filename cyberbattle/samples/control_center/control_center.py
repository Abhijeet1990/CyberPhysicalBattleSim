

from cyberbattle.simulation import cp_model as m
from cyberbattle.simulation.cp_model import NodeID, NodeInfo, VulnerabilityID, VulnerabilityInfo,Identifiers
from typing import Dict, Iterator, cast, Tuple
import random
from esa import SAW

# file path for the power world case
FilePath = r"D:\CyberBattleSim-main\PWCase\WSCC_9_bus.pwb"

class GridEnv:
    def __init__(self, train_sets):
        self.phy_env = SAW(FilePath)
        self.raw_train_sets = random.sample(train_sets, len(train_sets))
        self.train_sets = self.raw_train_sets
        self.last_action = None

    def getState(self):
        # output = None
        # scriptcommand = "SolvePowerFlow(RECTNEWT)"
        # if self.phy_env.runScriptCommand(scriptcommand) == True:
        #     output = self.phy_env.getPowerFlowResult('bus').loc[:, 'BusPUVolt'].astype(float)
        #
        #
        self.phy_env.SolvePowerFlow()
        voltages = self.phy_env.get_power_flow_results('bus').loc[:, 'BusPUVolt'].astype(float)
        return voltages.values.tolist()

    def nextSet(self):
        output=None
        action = self.train_sets.pop(0)
        self.last_action = action
        self.phy_env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        try:
            output = self.getState()
        except Exception as e:
            pass
        return output

    def reset(self):
        action = self.last_action
        if action is None:
            action = self.train_sets.pop(0)
        output = self.phy_env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        res = self.nextSet()
        return res


# Network nodes involve in the control center and substation network.
# Assume there are 3 substation network (within each substation there are two PC nodes)
# There is one control center network with 2 servers and a DNP3 Master.

# default outgoing rule for the control center
default_scada_allow_rules = [
    m.FirewallRule("DNP", m.RulePermission.ALLOW),
]

default_generic_allow_rules =[
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("FTP", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)
]

DEFAULT_ALLOW_RULES = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)]

# Environment constants used for all instances of the chain network
ENV_IDENTIFIERS = Identifiers(
    properties=[
        'Windows',
        'Linux',
        'ApacheWebSite',
        'IIS_2019',
        'IIS_2020_patched',
        'MySql',
        'Ubuntu',
        'nginx/1.10.3',
        'SMB_vuln',
        'SMB_vuln_patched',
        'SQLServer',
        'Win10',
        'Win10Patched',
        'FLAG:Linux'
    ],
    ports=[
        'HTTPS',
        'GIT',
        'SSH',
        'RDP',
        'PING',
        'MySQL',
        'SSH-key',
        'su',
        'DNP'
    ],
    local_vulnerabilities=[
        'ScanBashHistory',
        'ScanExplorerRecentFiles',
        'SudoAttempt',
        'CrackKeepPassX',
        'CrackKeepPass'
    ],
    remote_vulnerabilities=[
        'ProbeLinux',
        'ProbeWindows',
        'SqlInject',
        'ArpCachePoisonA',
        'ArpCachePoisonB',
        'ArpCachePoisonC',
        'CredScanGitHistory'
    ],
    phy_vulnerabilities=[
        'VoltageSetPoint',
        'BreakerStatus'
    ]
)


def prefix(x: int, name: str):
    """Prefix node name with an instance"""
    return f"{x}_{name}"


def rdp_password(index):
    """Generate RDP password for the specified chain link"""
    return f"WindowsPassword!{index}"


def ssh_password(index):
    """Generate SSH password for the specified chain link"""
    return f"LinuxPassword!{index}"


# look into Firewall paper to get more details

# we will use the cypsa objects to create the services, firewall policies etc.
# in cyberbattlesim they have system based firewall policies

# There are 2 types of vulnerability incorporated in the library (Local and Remote)
# There are 5 privilege levels

# add the vulnerabilities from the cypsa case

# add the list of vulnerabilities and scores
nodes = {
    "start" : m.NodeInfo(
            services=[],
            value=0,
            properties=['Windows','Win10'],
            vulnerabilities=dict(
                ScanExplorerRecentFiles=m.VulnerabilityInfo(
                    description="Scan Windows Explorer recent files for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node="cc_web", port="SSH",
                                           credential=ssh_password(1))]),
                    reward_string="Found a reference to a remote Web Server in the control center network",
                    cost=1.0
                ),
                CrackKeepPassX=m.VulnerabilityInfo(
                    description="Attempt to crack KeepPassX and look for credentials",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node="sA_web", port="RDP",
                                           credential=rdp_password(1))]),
                    reward_string=f"Discovered password to Windows machine {1}",
                    cost=1.0
                ),
                ScanBashHistory=m.VulnerabilityInfo(
                    description="Scan bash history for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedNodesId(["sB_web"]),
                    reward_string="Found a reference to a remote Windows node in bash history",
                    cost=1.0
                ),
                CrackKeepPass=m.VulnerabilityInfo(
                    description="Scan bash history for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedNodesId(["sC_web"]),
                    reward_string="Found a reference to a remote Windows node in bash history",
                    cost=1.0
                ),
            ),
            agent_installed=True,
            reimagable=False),
    "sA_dnp" : m.NodeInfo(
        services=[m.ListeningService("DNP")],
        value=0,
        properties=['Windows','Win10'],
        firewall=m.FirewallConfiguration(incoming=default_scada_allow_rules,
                                         outgoing=default_scada_allow_rules + default_generic_allow_rules),
        vulnerabilities=
            dict(
                VoltageSetPoint=m.VulnerabilityInfo(
                    description='Modify the voltage set point in Substation A',
                    type=m.VulnerabilityType.EXECUTE,
                    outcome=m.PhysicalAttack(),
                    reward_string="Cause voltage instability",
                    cost = 1.0
                ),
            ),
        agent_installed=True,
        affect_phy=True

    ),
    "sB_dnp":m.NodeInfo(
        services=[m.ListeningService("DNP")],
        value=0,
        properties=['Windows','Win10'],
        firewall=m.FirewallConfiguration(incoming=default_scada_allow_rules,
                                         outgoing=default_scada_allow_rules + default_generic_allow_rules),
        vulnerabilities=
            dict(
                VoltageSetPoint=m.VulnerabilityInfo(
                    description='Modify the voltage set point in Substation B',
                    type=m.VulnerabilityType.EXECUTE,
                    outcome=m.PhysicalAttack(),
                    reward_string="Cause voltage instability",
                    cost = 1.0
                ),
            ),
        agent_installed=True,
        affect_phy=True
    ),
    "sC_dnp":m.NodeInfo(
        services=[m.ListeningService("DNP")],
        value=0,
        properties=['Windows','Win10'],
        firewall=m.FirewallConfiguration(incoming=default_scada_allow_rules,
                                         outgoing=default_scada_allow_rules + default_generic_allow_rules),
        vulnerabilities=
            dict(
                VoltageSetPoint=m.VulnerabilityInfo(
                    description='Modify the voltage set point in Substation C',
                    type=m.VulnerabilityType.EXECUTE,
                    outcome=m.PhysicalAttack(),
                    reward_string="Cause voltage instability",
                    cost = 1.0
                ),
            ),
        agent_installed=True,
        affect_phy=True
    ),
    "sA_web":m.NodeInfo(
        services=[m.ListeningService("HTTP"), m.ListeningService("HTTPS")],
        value=0,
        properties=['Windows','Win10','MySql','SQLServer'],
        firewall=m.FirewallConfiguration(incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW),m.FirewallRule("HTTP", m.RulePermission.ALLOW)],
                                         outgoing=default_generic_allow_rules),
        vulnerabilities = dict(
                SqlInject=m.VulnerabilityInfo(
                    description="SQL injection to the DNP server DB",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node="sA_dnp", port="MySQL",
                                           credential=rdp_password(1))]),
                    reward_string="Remote machine is not running Linux",
                    cost=3.0
                )),
        affect_phy=False
    ),
    "sB_web":m.NodeInfo(
        services=[m.ListeningService("HTTP"), m.ListeningService("HTTPS")],
        value=0,
        properties=['Windows','Win10','MySql','SQLServer','IIS_2019','IIS_2020_patched'],
        firewall=m.FirewallConfiguration(incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW),m.FirewallRule("HTTP", m.RulePermission.ALLOW)],
                                         outgoing=default_generic_allow_rules),
        vulnerabilities = dict(
                SqlInject=m.VulnerabilityInfo(
                    description="SQL injection to the DNP server DB",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node="sB_dnp", port="DNP",
                                           credential=rdp_password(1))]),
                    reward_string="Remote machine is not running Linux",
                    cost=4.0
                )),
        affect_phy=False
    ),
    "sC_web":m.NodeInfo(
        services=[m.ListeningService("HTTP"), m.ListeningService("HTTPS")],
        value=0,
        properties=['Windows','Win10','MySql','SQLServer','IIS_2019','IIS_2020_patched'],
        firewall=m.FirewallConfiguration(incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW),m.FirewallRule("HTTP", m.RulePermission.ALLOW)],
                                         outgoing=default_generic_allow_rules),
        vulnerabilities=dict(
            CredScanGitHistory=m.VulnerabilityInfo(
                description="Some secure access token (SAS) leaked in a "
                            "reverted git commit",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="sC_dnp",
                                       port="HTTPS",
                                       credential="SASTOKEN1")]),
                reward_string="CredScan success: Some secure access token (SAS) was leaked in a reverted git commit",
                cost=1.0
            )),
        affect_phy=False
    ),
    "cc_web":m.NodeInfo(
        services=[m.ListeningService("HTTP"), m.ListeningService("HTTPS")],
        value=0,
        properties=['Windows','Win10','MySql','SQLServer','IIS_2019','IIS_2020_patched'],
        firewall=m.FirewallConfiguration(incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW),m.FirewallRule("HTTP", m.RulePermission.ALLOW)],
                                         outgoing=default_generic_allow_rules),
        vulnerabilities = dict(
                SqlInject=m.VulnerabilityInfo(
                    description="SQL injection to the PI server DB",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node="cc_pi", port="MySQL",
                                           credential=rdp_password(1))]),
                    reward_string="Remote machine is not running Linux",
                    cost=1.0
                )),
        affect_phy=False
    ),
    "cc_dnp":m.NodeInfo(
        services=[m.ListeningService("DNP")],
        value=0,
        properties=['Windows','Win10'],
        firewall=m.FirewallConfiguration(incoming=default_scada_allow_rules,
                                         outgoing=default_scada_allow_rules + default_generic_allow_rules),
        vulnerabilities = dict(
                ArpCachePoisonA=m.VulnerabilityInfo(
                    description="This is DNP3 command modification due to poisoning substation A network with ARP",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedNodesId(["sA_dnp"]),
                    reward_string="Found a reference to a remote DNP3 master in the control center",
                    cost=1.0
                ),
                ArpCachePoisonB=m.VulnerabilityInfo(
                    description="This is DNP3 command modification due to poisoning substation B network with ARP",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedNodesId(["sB_dnp"]),
                    reward_string="Found a reference to a remote DNP3 master in the control center",
                    cost=1.0
                ),
                ArpCachePoisonC=m.VulnerabilityInfo(
                    description="This is DNP3 command modification due to poisoning substation C network with ARP",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.LeakedNodesId(["sC_dnp"]),
                    reward_string="Found a reference to a remote DNP3 master in the control center",
                    cost=1.0
                ),
        ),
        affect_phy=False
    ),
    "cc_pi":m.NodeInfo(
        services=[m.ListeningService("DNP")],
        value=0,
        properties=['Windows','Win10','MySql','SQLServer'],
        firewall=m.FirewallConfiguration(incoming=default_scada_allow_rules,
                                         outgoing=default_generic_allow_rules),
        vulnerabilities = dict(
                ScanBashHistory=m.VulnerabilityInfo(
                    description="Scan bash history for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedNodesId(["cc_dnp"]),
                    reward_string="Found a reference to a remote DNP3 master in the control center",
                    cost=1.0
                )),
        affect_phy=False
    ),
}

global_vulnerability_library: Dict[VulnerabilityID, VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.infer_constants_from_nodes(
    cast(Iterator[Tuple[NodeID, NodeInfo]], list(nodes.items())),
    global_vulnerability_library)


def new_environment() -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=ENV_IDENTIFIERS
    )

