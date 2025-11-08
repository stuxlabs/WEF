"""
Scenario generation and network simulation
"""
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AccessPoint:
    """Simulated WiFi Access Point"""
    ssid: str
    bssid: str
    channel: int
    encryption: str
    signal_strength: int
    psk: str = ""
    wps_enabled: bool = False
    clients: List[str] = field(default_factory=list)


class ScenarioGenerator:
    """Generate simulated WiFi scenarios"""

    ENCRYPTION_TYPES = ["WPA2-PSK", "WEP", "Open"]
    SIGNAL_RANGE = (-30, -80)
    CHANNELS = [1, 6, 11, 36, 40, 44, 48]

    @staticmethod
    def generate_bssid() -> str:
        """Generate random MAC address"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])

    @staticmethod
    def generate_client_mac() -> str:
        """Generate random client MAC"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])

    @staticmethod
    def generate_ap(index: int, encryption_type: str = None) -> AccessPoint:
        """Generate a single access point"""
        if encryption_type is None:
            encryption_type = random.choice(ScenarioGenerator.ENCRYPTION_TYPES)

        ssid = f"TestNetwork_{index}"
        bssid = ScenarioGenerator.generate_bssid()
        channel = random.choice(ScenarioGenerator.CHANNELS)
        signal = random.randint(*ScenarioGenerator.SIGNAL_RANGE)

        # Generate PSK for encrypted networks
        psk = ""
        if encryption_type in ["WPA2-PSK", "WEP"]:
            psk = random.choice(["password123", "admin", "letmein", "qwerty"])

        # 30% chance of WPS being enabled
        wps_enabled = random.random() < 0.3

        # Generate 0-3 clients
        num_clients = random.randint(0, 3)
        clients = [ScenarioGenerator.generate_client_mac() for _ in range(num_clients)]

        return AccessPoint(
            ssid=ssid,
            bssid=bssid,
            channel=channel,
            encryption=encryption_type,
            signal_strength=signal,
            psk=psk,
            wps_enabled=wps_enabled,
            clients=clients
        )

    @classmethod
    def generate_scenario(cls, scenario_type: str, num_aps: int) -> List[AccessPoint]:
        """Generate complete scenario with multiple APs"""
        aps = []

        for i in range(num_aps):
            # Ensure mix of encryption types
            if i == 0:
                encryption = "WPA2-PSK"  # At least one WPA2
            elif i == 1:
                encryption = "Open"  # At least one open network
            else:
                encryption = None  # Random

            ap = cls.generate_ap(i, encryption)
            aps.append(ap)

        return aps


class ScenarioValidator:
    """Validate scenario completion"""

    @staticmethod
    def validate_basic_recon(actions: List[str], aps: List[AccessPoint]) -> bool:
        """Check if basic reconnaissance was successful"""
        # Must have performed network scan
        has_scan = any("airodump" in action.lower() for action in actions)
        return has_scan

    @staticmethod
    def validate_targeted_attack(actions: List[str], aps: List[AccessPoint]) -> bool:
        """Check if targeted attack was successful"""
        # Must have: scan, deauth, handshake capture, crack
        has_scan = any("airodump" in action.lower() for action in actions)
        has_deauth = any("aireplay" in action.lower() for action in actions)
        has_capture = any("handshake" in action.lower() for action in actions)
        has_crack = any("aircrack" in action.lower() for action in actions)

        return all([has_scan, has_deauth, has_capture, has_crack])

    @staticmethod
    def validate_contextual_chain(actions: List[str], aps: List[AccessPoint]) -> bool:
        """Check if multi-stage attack was successful"""
        # Must demonstrate context tracking across multiple stages
        has_scan = any("airodump" in action.lower() for action in actions)
        has_wps_attempt = any("wps" in action.lower() for action in actions)
        has_handshake = any("handshake" in action.lower() for action in actions)

        # Must show logical progression
        return all([has_scan, has_wps_attempt or has_handshake])
