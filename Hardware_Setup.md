# Hardware & Network Setup for RTSP Cameras

This guide provides detailed, step-by-step instructions for the one-time setup of the physical hardware and network configuration required for a CBAS recording rig. This is a technical guide intended for the person building the system.

For instructions on how to use the CBAS software to record from already-configured cameras, see the [**Recording Guide**](Recording.md).

---

## 1. Required Hardware & Software

### Hardware
(See the [master parts list](Recording_setup.md) for detailed recommendations and links.)

- PoE IP cameras  
- Network Switch (e.g., Aruba Instant On 1930 24G Class4 PoE)  
- 10Gb SFP+ to RJ-45 Transceiver (for connecting the computer to the switch's high-speed uplink port)  
- Cat5/6 Ethernet cables (for connecting cameras to the switch)  
- Cat6a/8 Ethernet cable (for connecting the computer to the switch)  
- A dedicated computer for recording and analysis  
- NAS (e.g., Synology) for network-based storage and backup  

### Software
- [Advanced IP Scanner](https://advanced-ip-scanner.com/download/): For easily discovering all devices on your network.  
- [VLC Media Player](https://videolan.org/vlc): For testing the camera's RTSP stream directly.  

---

## 2. Physical Connections

<p align="left">
  <img src=".//assets/switch.png" alt="Switch Connection Diagram" style="width: 300px; height: auto;">
</p>

1. Insert the RJ-45 transceiver into one of the SFP+ uplink ports on the switch (typically on the far right).  
2. Using a high-quality Ethernet cable (Cat6a/8), connect the computer’s Ethernet port to the transceiver in the switch.  
3. Using Cat5/6 cables, connect each PoE IP camera to one of the numbered PoE ports on the switch. The switch will provide power and data over this single cable.  
4. Connect the **NAS** to one of the remaining Ethernet ports on the switch (not Wi-Fi).  

---

## 3. Configure the Computer’s Static IP Address

To ensure reliable communication on this isolated network, manually assign an IP address to the computer:

1. In Windows Search, type **“Ethernet settings”** and open it.  
2. Find your Ethernet adapter and click **Edit** next to “IP assignment.”  
3. Change the dropdown from “Automatic (DHCP)” to **Manual.**  
4. Toggle the **IPv4** switch to **On.**  
5. Fill in the fields as follows:  
   - **IP address:** `192.168.1.30`  
   - **Subnet mask:** `255.255.255.0`  
   - **Gateway:** (leave blank)  
   - **Preferred DNS:** (leave blank)  
6. Click **Save.**

---

## 4. Configure the Network Switch

1. Open a web browser (Chrome, Edge, etc.).  
2. Navigate to the switch’s default IP address: `http://192.168.1.1`  
3. The switch’s login portal will appear. **Do not** connect it to the internet or an Aruba cloud account.  
4. Sign in with the default credentials (username: `admin`, password: *leave blank*).  
5. You will be prompted to create a new, secure password. Do so and log in again.  
6. On the Aruba dashboard, go to **Setup Network.**  
7. Under **IPv4 Configuration**, set:  
   - **Management Address Type:** Static  
   - **IP Address:** `192.168.1.2/24`  
   - **Subnet:** `255.255.255.0`  

> **Note:**  
> The NAS will use `192.168.1.1`, and the switch is now assigned `192.168.1.2` to avoid IP conflicts.  
> After applying changes, the switch will briefly disconnect; reconnect at `http://192.168.1.2`.  
> Record these IPs for future reference.  

---

## 5. Configure the NAS

1. Power on the NAS and connect it to the same switch.  
2. Using a browser, visit `http://192.168.1.1:5000` to access the NAS login page.  
   - If it does not appear, use the vendor’s discovery utility (e.g., Synology Assistant) or check the switch’s ARP table to locate its IP.  
3. Log in with the NAS admin credentials.  
4. Go to **Control Panel → Network → Network Interface.**  
5. Edit the LAN interface and set:  
   - **IP address:** `192.168.1.1`  
   - **Subnet mask:** `255.255.255.0`  
   - **Gateway / DNS:** (leave blank for isolated network)  
6. Under **File Services**, enable **SMB** sharing.  
   - Disable **SMB1** and ensure **SMB2/3** are enabled.  
7. Create a shared folder for CBAS projects (e.g., `CBAS_Projects`, or another descriptive name).  
8. Assign appropriate read/write permissions for the user account that CBAS will use (e.g., `labuser`, `cbas`, or your preferred account name).  
9. On the CBAS computer, mount the NAS share permanently:  
   ```powershell
   net use Z: \\192.168.1.1\<YourShareName> /user:<YourUsername> "YourPasswordHere" /persistent:yes
   ```  
10. Verify the share is accessible:  
    ```powershell
    dir Z:\
    ```  

---

## 6. Discover and Set Camera Static IPs

Your cameras will initially get a random IP from the switch. Assign each a permanent, static IP to ensure consistency.

### Step 6.1: Discover Initial IP Addresses
Use **Advanced IP Scanner**:
1. Set the scan range to `192.168.1.1–254` and click **Scan.**  
2. You should see:  
   - NAS (`192.168.1.1`)  
   - Switch (`192.168.1.2`)  
   - Computer (`192.168.1.30`)  
   - Cameras (`192.168.1.51–100`)  
3. Note the IP for each camera.  

### Step 6.2: Assign Static IPs
1. Open the camera’s IP in a browser (e.g., `192.168.1.53`).  
2. Log in (`admin`, *leave password blank*).  
3. Set a new password.  
4. Go to **Network → Basic → TCP/IP.**  
5. Configure:  
   - Uncheck **DHCP.**  
   - Assign a unique IP in `192.168.1.51–192.168.1.100`.  
   - Check **CloudSEE1.0 Compatibility Mode.**  
   - Uncheck **Auto online/offline** and **IP self-adaption.**  
   - Click **Lock IP** (if available).  
   - Click **Save.**

---

## 7. Configure Camera Video & Image Settings

Once IPs are stable, configure the video parameters for CBAS:

**Main Stream (analysis):**
- Codec: H265  
- FPS: 10  
- Quality: Best  
- Bitrate Control: VBR  
- Resolution: 2304×1296  
- Bitrate: 3072  

**Sub Stream (live preview):**
- Codec: H265  
- FPS: 10  
- Quality: Good  
- Bitrate Control: VBR  
- Resolution: 720×480  
- Bitrate: 256  

Disable nonessential features:
- Uncheck **Enable audio stream.**  
- Disable overlays in **OSD.**  
- Uncheck **Enable privacy mask.**  
- Uncheck **Enable motion detection.**

---

## 8. Final Verification

Before recording in CBAS, confirm that the network is stable and isolated.

1. **Scan the subnet:**  
   Run **Advanced IP Scanner** on `192.168.1.1–254` and verify all expected devices appear.  
2. **Test a camera stream in VLC:**  
   ```
   rtsp://admin:password@192.168.1.51:8554/profile0
   ```  
3. **Confirm NAS access:**  
   ```powershell
   dir Z:\
   ```  
   Ensure you can see your CBAS project directory.  
4. **Check routing priority:**  
   ```powershell
   Get-NetRoute -DestinationPrefix 192.168.1.0/24
   ```  
   Verify the Ethernet interface (not Wi-Fi) has the lowest metric.  
5. **Optional troubleshooting:**  
   If CBAS cannot detect cameras or NAS, temporarily disable Wi-Fi and ensure the switch (`192.168.1.2`) responds to:  
   ```powershell
   ping 192.168.1.2
   ```  

Your hardware and network are now fully configured and NAS-integrated for CBAS recording and analysis.
