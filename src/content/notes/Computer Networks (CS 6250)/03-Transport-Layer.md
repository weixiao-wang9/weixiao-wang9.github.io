---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[02-Network-Layer-and-Routing]]"
---

# Transport Layer

> **Prerequisites**: [[02-Network-Layer-and-Routing]]
> **Learning Goals**: After reading this, you will understand the difference between UDP and TCP, explain TCP connection management, understand flow control vs congestion control, and grasp how TCP prevents network collapse.

## Introduction

The Transport Layer provides **logical communication between application processes** running on different hosts. While the Network Layer delivers packets between hosts, the Transport Layer ensures delivery between specific **applications** on those hosts.

**Key Gap It Fills**:
- Network Layer offers "best-effort" delivery (packets may be lost, reordered, duplicated)
- Transport Layer (specifically TCP) adds **reliability** so applications don't have to worry about these issues

---

## Transport Layer Fundamentals

### Purpose

**What It Does**:
- Provides **process-to-process** communication (not just host-to-host)
- Wraps application messages into **segments** by adding a transport header
- Delivers segments to the correct application using **port numbers**

**Key Services**:
1. **Multiplexing/Demultiplexing**: Directing data to the correct application
2. **Reliability** (TCP only): Ensuring data arrives correctly and in order
3. **Flow Control** (TCP only): Matching sender and receiver speeds
4. **Congestion Control** (TCP only): Preventing network overload

### Encapsulation

**At the Sender**:
```
Application Message
    ↓
Transport Layer adds header (ports, checksum, sequence numbers)
    ↓
Segment = [Transport Header | Application Message]
    ↓
Passed to Network Layer
```

**At the Receiver**:
```
Network Layer delivers segment
    ↓
Transport Layer removes header, checks integrity
    ↓
Delivers message to correct application (via port)
```

---

## Multiplexing and Demultiplexing

### The Problem

**Challenge**: IP addresses identify **hosts**, not specific **applications**

**Example**:
- Your laptop (IP: 192.168.1.10) runs:
  - Web browser (connects to Facebook)
  - Spotify
  - Email client

All three apps share the same IP. How does incoming data know which app to go to?

**Solution**: **Port numbers**

---

### Port Numbers

**Definition**: 16-bit identifiers (0-65535) that specify which application should receive data

**Port Ranges**:
- **0-1023**: Well-known ports (HTTP=80, HTTPS=443, SSH=22, DNS=53)
- **1024-49151**: Registered ports (application-specific)
- **49152-65535**: Ephemeral/Dynamic ports (temporary, assigned by OS)

**Example**:
```
Client (192.168.1.10:54321) → Server (93.184.216.34:443)
                               ↑           ↑
                               IP        Port (HTTPS)
```

---

### Connectionless Demultiplexing (UDP)

**How It Works**:
- Socket identified by a **2-tuple**: `(Destination IP, Destination Port)`
- Receiver looks at destination port and delivers to that socket
- **Source IP/port not considered** for demultiplexing

**Example**:
```
Server listening on port 5000

Client A (10.0.0.1:6000) → Server (192.168.1.100:5000)
Client B (10.0.0.2:7000) → Server (192.168.1.100:5000)

Both go to the same server socket (port 5000)
Server cannot distinguish clients unless it inspects source IP/port manually
```

**Use Case**: DNS, streaming (one server socket handles all clients)

---

### Connection-Oriented Demultiplexing (TCP)

**How It Works**:
- Socket identified by a **4-tuple**: `(Source IP, Source Port, Dest IP, Dest Port)`
- Each connection gets a **unique socket**
- Server can handle multiple clients simultaneously

**Example**:
```
Server listening on port 80

Client A (10.0.0.1:6000, 192.168.1.100:80) → Socket 1
Client B (10.0.0.2:7000, 192.168.1.100:80) → Socket 2

Two different sockets, even though both connect to port 80
```

**Use Case**: Web servers (handle thousands of concurrent connections)

---

## UDP (User Datagram Protocol)

### Characteristics

**Nature**:
- **Connectionless**: No handshake before sending data
- **Unreliable**: No guarantees (packets may be lost, reordered, duplicated)
- **No flow control**: Sender can send as fast as it wants
- **No congestion control**: Does not respond to network congestion

**When to Use UDP**:
1. **Real-time applications** sensitive to delay (VoIP, live video, gaming)
2. **DNS**: Small requests/responses (one packet)
3. **Applications that handle reliability themselves** (custom protocols)

### UDP Header

**Size**: 64 bits (8 bytes) - very lightweight

**Fields**:
```
0                   16                  32
+-------------------+-------------------+
|   Source Port     | Destination Port  |
+-------------------+-------------------+
|      Length       |     Checksum      |
+-------------------+-------------------+
|              Data (Payload)           |
+---------------------------------------+
```

1. **Source Port** (16 bits): Sender's port (optional, can be 0)
2. **Destination Port** (16 bits): Receiver's port
3. **Length** (16 bits): Total length of segment (header + data)
4. **Checksum** (16 bits): Error detection (optional in IPv4, mandatory in IPv6)

**Minimal Overhead**: Only 8 bytes (vs 20 bytes for TCP minimum)

---

### Why Use UDP?

**Advantages**:
1. **Low latency**: No connection setup delay
2. **Simple**: No connection state to maintain
3. **Small header**: Less overhead
4. **Full control**: Application decides when to send (no congestion control throttling)

**Trade-offs**:
- Application must handle packet loss, reordering, duplication
- Not suitable for reliable data transfer (file downloads, email)

---

## TCP (Transmission Control Protocol)

### Characteristics

**Nature**:
- **Connection-oriented**: Three-way handshake before data transfer
- **Reliable**: Guarantees delivery, correct order, no duplicates
- **Flow control**: Prevents sender from overwhelming receiver
- **Congestion control**: Prevents sender from overwhelming network
- **Full-duplex**: Bidirectional communication

**When to Use TCP**:
- File transfers (FTP, HTTP downloads)
- Email (SMTP)
- Remote access (SSH)
- Any application requiring reliable delivery

---

### TCP Connection Management

**Three-Way Handshake** (Establishing Connection):

```
Client                          Server
  |                               |
  | ---- SYN (seq=x) -----------> |  (1) Client requests connection
  |                               |
  | <--- SYNACK (seq=y, ack=x+1)- |  (2) Server acknowledges, sends its own SYN
  |                               |
  | ---- ACK (ack=y+1) ---------> |  (3) Client acknowledges server's SYN
  |                               |
  |       Connection Established  |
```

**Step-by-Step**:

**1. SYN** (Synchronize):
- Client sends: `SYN` flag set, initial sequence number `x`
- Indicates: "I want to establish a connection"

**2. SYNACK** (Synchronize-Acknowledge):
- Server sends: `SYN` and `ACK` flags set, sequence number `y`, acknowledgment `x+1`
- Indicates: "I acknowledge your SYN, here's my SYN"

**3. ACK** (Acknowledge):
- Client sends: `ACK` flag set, acknowledgment `y+1`
- Indicates: "I acknowledge your SYN, connection established"

**Why Three Steps?**
- Ensures both sides are ready
- Establishes initial sequence numbers for reliable delivery
- Prevents stale connection requests from causing issues

---

**Connection Teardown** (Four-Way Handshake):

```
Client                          Server
  |                               |
  | ---- FIN ------------------>  |  (1) Client done sending
  |                               |
  | <--- ACK --------------------  |  (2) Server acknowledges
  |                               |
  | <--- FIN --------------------  |  (3) Server done sending
  |                               |
  | ---- ACK ------------------>  |  (4) Client acknowledges
  |                               |
  |       Connection Closed       |
```

**Note**: Can be optimized to three steps if server sends FIN+ACK together.

---

### TCP Reliability (ARQ - Automatic Repeat Request)

**How TCP Ensures Reliability**:

**1. Sequence Numbers**:
- Each byte of data has a sequence number
- Receiver can detect missing or out-of-order data

**2. Acknowledgments (ACKs)**:
- Receiver sends ACK for successfully received data
- ACK number = next expected sequence number

**3. Timeouts and Retransmissions**:
- Sender starts a timer when sending a segment
- If ACK not received before timeout, retransmit
- **Timeout value**: Dynamically estimated based on RTT (Round-Trip Time)

**4. Cumulative ACKs**:
- ACK number `n` means "I've received all bytes up to `n-1`"
- Example: ACK 5000 means bytes 0-4999 received

**Example**:
```
Sender sends: Segment with seq=100, len=50 (bytes 100-149)

Receiver receives successfully:
  → Sends ACK 150 (expecting byte 150 next)

Sender receives ACK 150:
  → Knows bytes 100-149 delivered successfully

If ACK not received before timeout:
  → Sender retransmits segment with seq=100
```

---

## Flow Control

### Purpose

**Problem**: Sender transmits faster than receiver can process
- Receiver's buffer overflows
- Packets dropped
- Performance degrades

**Solution**: **Flow Control** - Receiver tells sender how much buffer space it has

---

### TCP Sliding Window

**Mechanism**: **Receive Window (rwnd)**

**How It Works**:
1. Receiver advertises **rwnd** in every ACK
   - `rwnd` = Available buffer space at receiver
2. Sender limits unacknowledged data to ≤ `rwnd`
   - `Unacknowledged Data = Last Byte Sent - Last Byte ACKed`
3. As receiver consumes data, `rwnd` increases
4. As sender sends more data, `rwnd` decreases

**Example**:
```
Receiver has 10KB buffer

Initially:
  rwnd = 10KB

Sender sends 5KB:
  Unacknowledged data = 5KB
  rwnd advertised = 5KB (10KB - 5KB consumed)

Receiver processes 3KB:
  rwnd advertised = 8KB (5KB still in buffer, 3KB freed)

Sender can send 3KB more (to fill up to 8KB unacknowledged)
```

**Key Insight**: Sender adapts to receiver's processing speed.

---

### Zero Window Problem

**Scenario**:
- Receiver's buffer full (`rwnd = 0`)
- Sender stops sending
- Receiver processes data, frees buffer
- **Problem**: How does sender know buffer is available again?

**Solution**: **Persist Timer**
- Sender periodically sends 1-byte "window probe"
- Receiver responds with updated `rwnd`
- Sender resumes when `rwnd > 0`

---

## Congestion Control

### Purpose

**Problem**: Sender transmits faster than the **network** can handle
- Routers' buffers overflow
- Packet loss increases
- Network collapse (congestion collapse)

**Difference from Flow Control**:
- **Flow Control**: Protects the **receiver**
- **Congestion Control**: Protects the **network**

---

### TCP Congestion Control Mechanisms

**Congestion Window (cwnd)**:
- **Sender's estimate** of how much data the network can handle
- Sender limits: `Unacknowledged Data ≤ min(cwnd, rwnd)`

**Goal**: Find the right `cwnd` to maximize throughput without causing congestion.

---

### AIMD (Additive Increase Multiplicative Decrease)

**Core Algorithm**:

**Additive Increase** (No congestion detected):
- Increase `cwnd` by **1 MSS** (Maximum Segment Size) per RTT
- Gradual, linear growth

**Multiplicative Decrease** (Congestion detected):
- Decrease `cwnd` by **half** when packet loss detected
- Aggressive backoff

**Congestion Detection**:
- **Timeout**: Severe congestion (no ACKs received)
- **3 Duplicate ACKs**: Mild congestion (some packets lost, others delivered)

**Sawtooth Pattern**:
```
cwnd
 ^
 |     /\      /\      /\
 |    /  \    /  \    /  \
 |   /    \  /    \  /    \
 |  /      \/      \/      \
 +----------------------------> Time
   Increase  Loss   Increase  Loss
```

**Why AIMD?**
- **Fairness**: Multiple connections converge to equal bandwidth shares
- **Efficiency**: Probes for available bandwidth without overloading

---

### Slow Start

**Purpose**: Quickly find the appropriate `cwnd` at connection start

**How It Works**:
1. Start with `cwnd = 1 MSS`
2. **Double `cwnd` every RTT** (exponential growth)
   - Each ACK increases `cwnd` by 1 MSS
   - If `cwnd = 4`, sending 4 segments → receive 4 ACKs → `cwnd = 8`
3. Continue until reaching **ssthresh** (slow start threshold) or detecting loss

**Example**:
```
RTT 1: cwnd = 1 → Send 1 segment
RTT 2: cwnd = 2 → Send 2 segments
RTT 3: cwnd = 4 → Send 4 segments
RTT 4: cwnd = 8 → Send 8 segments
...
```

**Why "Slow Start"?**
- Named ironically: It's actually **fast** (exponential)
- "Slow" compared to sending at full capacity immediately

**Transition to AIMD**:
- When `cwnd ≥ ssthresh`, switch to **Additive Increase** (linear growth)
- Or when packet loss detected

---

### Congestion Detection and Response

**1. Timeout** (Severe Congestion):
- No ACK received for a segment
- **Actions**:
  - Set `ssthresh = cwnd / 2`
  - Set `cwnd = 1 MSS`
  - Re-enter **Slow Start**

**2. Triple Duplicate ACKs** (Mild Congestion):
- Receiver sends same ACK 3 times (indicates missing segment)
- Some packets still getting through (ACKs arriving)
- **Actions** (Fast Recovery):
  - Set `ssthresh = cwnd / 2`
  - Set `cwnd = ssthresh` (cut in half)
  - Continue with **AIMD** (no slow start)

**Why Different Responses?**
- Timeout = severe (network might be congested)
- 3 Dup ACKs = mild (just one segment lost, others delivered)

---

### Modern TCP: TCP CUBIC

**Problem with Traditional TCP**:
- AIMD is **RTT-dependent**: Connections with smaller RTT grab more bandwidth
- Unfair in high-bandwidth, high-delay networks (long-distance links)

**TCP CUBIC** (Modern Default):

**Key Idea**: Use a **cubic function** for window growth, independent of RTT

**Window Growth Function**:
```
cwnd(t) = C × (t - K)³ + W_max

Where:
  t = Time since last congestion event
  W_max = Window size when last loss occurred
  K = Time to reach W_max again
  C = Scaling constant
```

**Behavior**:
```
cwnd
 ^
 |         ___---
 |      _--
 |   __/        Cubic growth
 | _/
 |/
 +---------------> Time
       W_max
```

**Phases**:
1. **Slow growth** near `W_max` (probing cautiously)
2. **Faster growth** far from `W_max` (recovering quickly)
3. **Independent of RTT**: All flows grow at same rate

**Advantages**:
- **RTT fairness**: Long-distance connections not penalized
- **Fast recovery**: Quickly reclaims bandwidth after loss
- **Stable**: Fewer oscillations near capacity

---

## TCP Header

**Size**: 20-60 bytes (20 bytes minimum, up to 40 bytes of options)

**Key Fields**:
```
0               16              32
+---------------+---------------+
| Source Port   | Dest Port     |
+---------------+---------------+
|        Sequence Number        |
+-------------------------------+
|     Acknowledgment Number     |
+-------------------------------+
| Hdr  | Flags  | Window Size   |
| Len  |        | (rwnd)        |
+-------------------------------+
| Checksum      | Urgent Ptr    |
+-------------------------------+
|      Options (if any)         |
+-------------------------------+
|          Data                 |
+-------------------------------+
```

**Important Fields**:

1. **Sequence Number** (32 bits): Byte number of first byte in this segment
2. **Acknowledgment Number** (32 bits): Next expected byte number
3. **Flags** (6 bits):
   - `SYN`: Synchronize (connection setup)
   - `ACK`: Acknowledgment valid
   - `FIN`: Finish (connection teardown)
   - `RST`: Reset connection
   - `PSH`: Push data to application immediately
   - `URG`: Urgent data
4. **Window Size (rwnd)** (16 bits): Receiver's buffer space
5. **Checksum** (16 bits): Error detection

---

## Summary

### Key Takeaways

1. **Transport Layer Role**: Provides process-to-process communication using port numbers

2. **UDP vs TCP**:
   - **UDP**: Fast, unreliable, connectionless (VoIP, DNS, gaming)
   - **TCP**: Reliable, connection-oriented, flow/congestion control (web, email, file transfer)

3. **TCP Connection Management**:
   - **Three-way handshake** to establish
   - **Four-way handshake** to teardown
   - Ensures both sides are synchronized

4. **Flow Control**: Protects receiver from overflow using sliding window (`rwnd`)

5. **Congestion Control**: Protects network from collapse
   - **AIMD**: Gradual increase, sharp decrease
   - **Slow Start**: Exponential growth to find capacity
   - **TCP CUBIC**: Modern, RTT-independent, cubic growth function

6. **Reliability**: Achieved through sequence numbers, ACKs, timeouts, and retransmissions

### Common Patterns

**Protocol Selection**:
- Need reliability? → TCP
- Need low latency? → UDP
- Custom reliability? → UDP with application-layer retransmission

**Congestion Response**:
- Timeout → Severe (restart slow start)
- 3 Dup ACKs → Mild (halve window, continue)

**Trade-offs**:
- Reliability vs. Speed (TCP vs UDP)
- Fairness vs. Throughput (AIMD vs. aggressive sending)
- RTT fairness vs. Simplicity (CUBIC vs. traditional TCP)

---

## See Also

- [[02-Network-Layer-and-Routing]] - IP addressing and routing
- [[04-Advanced-Routing-and-QoS]] - QoS mechanisms and scheduling
- [[06-Application-Layer-Services]] - Applications using UDP/TCP (VoIP, streaming)

**Next**: [[04-Advanced-Routing-and-QoS]]
