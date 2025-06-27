# Hardware & Network Setup for RTSP Cameras

This guide provides detailed, step-by-step instructions for the one-time setup of the physical hardware and network configuration required for a CBAS recording rig. This is a technical guide intended for the person building the system.

For instructions on how to use the CBAS software to record from already-configured cameras, see the [**Recording Guide**](Recording.md).

---

## 1. Required Hardware & Software

### Hardware
(See the [master parts list](Recording_setup.md) for detailed recommendations and links).

-   PoE IP cameras
-   Network Switch (e.g., Aruba Instant On 1930 24G Class4 PoE)
-   10Gb SFP+ to RJ-45 Transceiver (for connecting the computer to the switch's high-speed uplink port)
-   Cat5/6 Ethernet cables (for connecting cameras to the switch)
-   Cat6a/8 Ethernet cable (for connecting the computer to the switch)
-   A dedicated computer for recording and analysis

### Software
-   [Advanced IP Scanner](https://advanced-ip-scanner.com/download/): For easily discovering all devices on your network.
-   [VLC Media Player](https://videolan.org/vlc): For testing the camera's RTSP stream directly.

---

## 2. Physical Connections

<p align="left">
    <img src=".//assets/switch.png" alt="Switch Connection Diagram" style="width: 300px; height: auto;">
</p>

1.  Insert the RJ-45 transceiver into one of the SFP+ uplink ports on the switch (typically on the far right).
2.  Using a high-quality ethernet cable (Cat6a/8), connect the computer's ethernet port to the transceiver in the switch.
3.  Using Cat5/6 cables, connect each PoE IP camera to one of the numbered PoE ports on the switch. The switch will provide power and data over this single cable.

---

## 3. Configure the Computer's Static IP Address

To ensure reliable communication on this isolated network, we must manually assign an IP address to the computer.

1.  In Windows Search, type **'Ethernet settings'** and open it.
2.  Find your ethernet adapter and click **Edit** next to "IP assignment".
3.  Change the dropdown from "Automatic (DHCP)" to **Manual**.
4.  Toggle the **IPv4** switch to **On**.
5.  Fill in the fields exactly as follows:
    -   **IP address:** `192.168.1.30`
    -   **Subnet mask:** `255.255.255.0`
    -   **Gateway:** (leave blank)
    -   **Preferred DNS:** (leave blank)
6.  Click **Save**.

---

## 4. Configure the Network Switch

1.  Open a web browser (Chrome, Edge, etc.).
2.  Navigate to the switch's default IP address: `http://192.168.1.1`
3.  The switch's login portal will appear. **Do not** connect it to the internet or an Aruba cloud account.
4.  Sign in with the default credentials (username: `admin`, password: *leave blank*).
5.  You will be prompted to create a new, secure password. Do so and log in again.
6.  On the Aruba dashboard, navigate to **Setup Network**.
7.  Under IPv4 Configuration, ensure the settings are as follows:
    -   **Management Address Type:** Static
    -   **IP Address:** `192.168.1.1/24`
    -   **Subnet:** `255.255.255.0`

---

## 5. Discover and Set Camera Static IPs

Your cameras will initially get a random IP address from the switch. We need to find these addresses and then assign each camera a permanent, static IP.

### Step 5.1: Discover Initial IP Addresses
You can use the switch's ARP table, but **Advanced IP Scanner** is often easier.
1.  Open Advanced IP Scanner.
2.  Set the scan range to `192.168.1.1-254` and click **Scan**.
3.  You will see a list of connected devices. Your computer (`192.168.1.30`), the switch (`192.168.1.1`), and your cameras will be listed. The camera manufacturer may appear as "Jinan Jovision Science & Technology Co., Ltd.".
4.  Note down the initial IP address for each camera.

### Step 5.2: Assign Static IPs
For each camera, you will log into its web interface to change its network settings.
1.  Open a new browser tab and type in one of the camera's discovered IP addresses (e.g., `192.168.1.38`).
2.  Log in with the camera's default credentials (username: `admin`, password: *leave blank*).
3.  You will be prompted to set a new, secure password for the camera. **Remember this password!**
4.  Navigate to the **Network -> Basic** configuration page.
5.  On the **TCP/IP** tab, configure the following:
    1.  **Uncheck DHCP**. This will allow you to set a static IP.
    2.  Manually change the **IP address**. It's best practice to assign them sequentially and outside the range the switch might use for DHCP. A good range is `192.168.1.51` to `192.168.1.100`. Assign each camera a unique IP (e.g., `.51`, `.52`, `.53`, ...).
    3.  Check **CloudSEE1.0 Compatibility Mode**.
    4.  Uncheck **Auto online/offline**.
    5.  Uncheck **IP self-adaption**.
    6.  Click **Lock IP** if available.
    7.  Click **Save**. The camera may reboot.
6.  Repeat this process for all cameras, giving each a unique, permanent IP address.

---

## 6. Configure Camera Video & Image Settings

Once you can access each camera's web interface at its new static IP, you should configure the video stream and image settings for optimal recording with CBAS.

*These are recommended settings, but you can adjust them for your specific needs.*

1.  **Video Stream (`Video and Audio -> Video Stream`):**
    -   **Main Stream (used for analysis):**
        -   **Codec:** H265
        -   **FPS:** 10
        -   **Quality:** Best
        -   **Bitrate Control:** VBR (Variable Bitrate)
        -   **Resolution:** 2304x1296 (or highest available)
        -   **Bitrate:** 3072
    -   **Sub Stream (used for live preview):**
        -   **Codec:** H265
        -   **FPS:** 10
        -   **Quality:** Good
        -   **Bitrate Control:** VBR
        -   **Resolution:** 720x480
        -   **Bitrate:** 256
2.  **Audio Stream (`Video and Audio -> Audio Stream`):**
    -   **Uncheck** "Enable audio stream".
3.  **Image Settings (`Display -> Image`):**
    -   **Brightness:** 100
    -   **Contrast:** 100
    -   **Saturation:** 0 (This creates a black and white image, which is ideal for IR recordings).
    -   **Sharpness:** 128
    -   **Mirror, Flip, SmartIR:** all **unchecked**.
    -   **Image Style:** Standard
4.  **Exposure Settings (`Display -> Exposure`):**
    -   **Anti-Flicker:** Off
    -   **Max exposure time:** 1/3 (Allows for maximum light sensitivity in dark conditions).
    -   **Min exposure time:** 1/100000
5.  **Day & Night Settings (`Display -> Day&Night`):**
    -   **Switch mode:** Auto
    -   **Sensitivity:** 4
6.  **On-Screen Display (`Display -> OSD`):**
    -   **Uncheck** all options (like Large Font, Name Position, Time Position) to disable the timestamp and camera name overlay. CBAS does not record this overlay, but disabling it ensures a clean stream.
7.  **Disable Unused Features:**
    -   `Privacy Mask`: Ensure "Enable privacy mask" is **unchecked**.
    -   `Alarm -> Motion Detection`: Ensure "Enable motion" is **unchecked**.
8.  Click **Save** after making changes in each section.

---

## 7. Final Verification

Before proceeding to use the cameras in CBAS, do a final check.

1.  Run **Advanced IP Scanner** again on the `192.168.1.1-254` range. Verify that all your cameras appear with their new, permanent static IPs.
2.  Open **VLC Media Player** and test a camera stream.
    -   Go to `Media -> Open Network Stream`.
    -   Enter the camera's URL in the format `rtsp://<username>:<password>@<camera_ip_address>:8554/profile0`.
    -   Example: `rtsp://admin:MySecurePwd@192.168.1.51:8554/profile0`
    -   Click Play. You should see a live video feed.

Your hardware and network are now fully configured and ready to be used with the CBAS software.