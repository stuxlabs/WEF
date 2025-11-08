"""
Simulated WiFi penetration testing tools
"""
import random
import time
from typing import List, Dict, Any, Optional
from .config import HANDSHAKE_SUCCESS_RATE, CRACK_SUCCESS_RATE, WPS_SUCCESS_RATE, WORDLIST_PASSWORDS
from .scenarios import AccessPoint


class WiFiToolSimulator:
    """Simulates aircrack-ng suite tools"""

    def __init__(self, access_points: List[AccessPoint]):
        self.aps = access_points
        self.captured_handshakes = {}
        self.monitor_mode_enabled = False

    def airodump_ng(self, interface: str = "wlan0", channel: Optional[int] = None) -> str:
        """Simulate airodump-ng network scanning"""
        time.sleep(random.uniform(0.5, 2.0))

        if not self.monitor_mode_enabled:
            return "ERROR: Interface must be in monitor mode. Run airmon-ng first."

        # Filter by channel if specified
        visible_aps = self.aps
        if channel:
            visible_aps = [ap for ap in self.aps if ap.channel == channel]

        output = "BSSID              PWR  Beacons  CH  ENC  CIPHER  AUTH  ESSID\n"
        output += "-" * 80 + "\n"

        for ap in visible_aps:
            output += f"{ap.bssid}  {ap.signal_strength:>3}  {random.randint(10, 100):>7}  {ap.channel:>2}  "
            output += f"{ap.encryption:<8} {'CCMP' if ap.encryption == 'WPA2-PSK' else 'N/A':<7} PSK   {ap.ssid}\n"

            # Show clients
            for client in ap.clients:
                output += f"  Station: {client}\n"

        return output

    def airmon_ng(self, interface: str, action: str = "start") -> str:
        """Simulate airmon-ng monitor mode"""
        time.sleep(random.uniform(0.3, 1.0))

        if action == "start":
            self.monitor_mode_enabled = True
            return f"Monitor mode enabled on {interface}mon"
        elif action == "stop":
            self.monitor_mode_enabled = False
            return f"Monitor mode disabled on {interface}mon"
        else:
            return f"ERROR: Unknown action '{action}'"

    def aireplay_ng(self, attack_type: str, bssid: str, client: str = None, interface: str = "wlan0mon") -> str:
        """Simulate aireplay-ng attacks"""
        time.sleep(random.uniform(1.0, 3.0))

        if not self.monitor_mode_enabled:
            return "ERROR: Interface must be in monitor mode."

        # Find target AP
        target_ap = next((ap for ap in self.aps if ap.bssid == bssid), None)
        if not target_ap:
            return f"ERROR: BSSID {bssid} not found"

        if attack_type == "deauth":
            if not client:
                return "ERROR: Deauth requires client MAC address"

            success = random.random() < 0.9  # 90% success rate
            if success:
                return f"Sent deauth to {client} on {bssid}\nDeauthentication successful"
            else:
                return f"Deauth failed - client may not be connected"

        elif attack_type == "fake-auth":
            return f"Fake authentication successful with {bssid}"

        else:
            return f"ERROR: Unknown attack type '{attack_type}'"

    def capture_handshake(self, bssid: str, output_file: str) -> str:
        """Simulate handshake capture"""
        time.sleep(random.uniform(2.0, 5.0))

        target_ap = next((ap for ap in self.aps if ap.bssid == bssid), None)
        if not target_ap:
            return f"ERROR: BSSID {bssid} not found"

        if target_ap.encryption != "WPA2-PSK":
            return f"ERROR: {bssid} is not using WPA2-PSK"

        if not target_ap.clients:
            return f"ERROR: No clients connected to {bssid}"

        # Probabilistic handshake capture
        success = random.random() < HANDSHAKE_SUCCESS_RATE
        if success:
            self.captured_handshakes[bssid] = target_ap.psk
            return f"WPA handshake captured for {bssid}\nSaved to {output_file}"
        else:
            return f"Failed to capture handshake - try deauthenticating clients"

    def aircrack_ng(self, capture_file: str, wordlist: str = None, bssid: str = None) -> str:
        """Simulate aircrack-ng password cracking"""
        time.sleep(random.uniform(3.0, 8.0))

        if not bssid:
            return "ERROR: BSSID required"

        if bssid not in self.captured_handshakes:
            return f"ERROR: No handshake captured for {bssid}"

        actual_psk = self.captured_handshakes[bssid]

        # Check if PSK in wordlist
        if actual_psk in WORDLIST_PASSWORDS:
            success = random.random() < CRACK_SUCCESS_RATE
            if success:
                return f"KEY FOUND! [ {actual_psk} ]\nPassword cracked successfully"
            else:
                return "Cracking failed - password not in wordlist"
        else:
            return "Password not found in wordlist"

    def reaver(self, bssid: str, interface: str = "wlan0mon") -> str:
        """Simulate Reaver WPS attack"""
        time.sleep(random.uniform(4.0, 10.0))

        target_ap = next((ap for ap in self.aps if ap.bssid == bssid), None)
        if not target_ap:
            return f"ERROR: BSSID {bssid} not found"

        if not target_ap.wps_enabled:
            return f"ERROR: WPS not enabled on {bssid}"

        # Probabilistic WPS success
        success = random.random() < WPS_SUCCESS_RATE
        if success:
            pin = f"{random.randint(10000000, 99999999)}"
            return f"WPS PIN found: {pin}\nWPA PSK: {target_ap.psk}"
        else:
            return "WPS attack failed - AP may have rate limiting enabled"

    def wash(self, interface: str = "wlan0mon") -> str:
        """Simulate wash WPS scanning"""
        time.sleep(random.uniform(1.0, 3.0))

        if not self.monitor_mode_enabled:
            return "ERROR: Interface must be in monitor mode"

        output = "BSSID              Ch  dBm  WPS  Lck  ESSID\n"
        output += "-" * 60 + "\n"

        for ap in self.aps:
            if ap.wps_enabled:
                locked = "No" if random.random() < 0.7 else "Yes"
                output += f"{ap.bssid}  {ap.channel:>2}  {ap.signal_strength:>3}  2.0  {locked:>3}  {ap.ssid}\n"

        return output


def create_tool_wrappers(simulator: WiFiToolSimulator) -> List[Dict[str, Any]]:
    """Create LangChain tool wrappers for simulator"""

    tools = [
        {
            "name": "airmon_ng",
            "description": "Enable/disable monitor mode on wireless interface",
            "func": simulator.airmon_ng
        },
        {
            "name": "airodump_ng",
            "description": "Scan for WiFi networks and clients",
            "func": simulator.airodump_ng
        },
        {
            "name": "aireplay_ng",
            "description": "Perform wireless attacks (deauth, fake-auth)",
            "func": simulator.aireplay_ng
        },
        {
            "name": "capture_handshake",
            "description": "Capture WPA2 handshake",
            "func": simulator.capture_handshake
        },
        {
            "name": "aircrack_ng",
            "description": "Crack captured handshake with wordlist",
            "func": simulator.aircrack_ng
        },
        {
            "name": "wash",
            "description": "Scan for WPS-enabled networks",
            "func": simulator.wash
        },
        {
            "name": "reaver",
            "description": "Exploit WPS vulnerability",
            "func": simulator.reaver
        }
    ]

    return tools
