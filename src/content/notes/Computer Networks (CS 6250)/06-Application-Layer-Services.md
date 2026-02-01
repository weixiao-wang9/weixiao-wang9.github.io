---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[03-Transport-Layer]]"
---

# Application Layer Services: CDNs and Multimedia

> **Prerequisites**: [[03-Transport-Layer]]
> **Learning Goals**: After reading this, you will understand CDN architecture and server placement strategies, DNS-based and anycast redirection, consistent hashing for content distribution, multimedia streaming (DASH), bitrate adaptation algorithms, and VoIP mechanisms.

## Introduction

The Application Layer delivers content and services directly to users. As Internet traffic shifted from text (email, web pages) to multimedia (video, voice), new challenges emerged: latency, bandwidth, scalability. **Content Distribution Networks (CDNs)** address these by replicating content geographically, while **multimedia applications** adapt to network conditions. This file covers the architectures and algorithms that power Netflix, YouTube, Zoom, and the modern Internet.

**Key Challenges**:
- How to serve billions of users with low latency (CDNs)
- How to stream video smoothly despite varying bandwidth (adaptive bitrate)
- How to deliver real-time voice with <150ms delay (VoIP)

---

## Content Distribution Networks (CDNs)

### Motivation: The Single-Server Problem

**Traditional Approach**: Single data center serving all users

**Problems**:
1. **High latency**: Users far from server experience long delays (intercontinental RTT = 100-300ms)
2. **Bandwidth waste**: Popular content (viral videos) transmitted redundantly across the same links
3. **Single point of failure**: Server/network failure affects all users
4. **Poor scalability**: Limited by one location's capacity

**Example**:
```
Viral video (1 GB) viewed 1 million times:
  - Without CDN: 1 PB transferred from single server across backbone
  - Server link saturates, users worldwide experience buffering

With CDN:
  - 1000 edge servers each serve 1000 users locally
  - Total backbone traffic: 1 TB (to distribute to edge servers)
  - 1000× reduction in backbone load
```

---

### CDN Solution: Geographic Distribution

**Core Idea**: Replicate content across geographically distributed servers, serve users from the **nearest** server

**Benefits**:
1. **Low latency**: Users served from nearby servers (single-digit ms RTT)
2. **Reduced backbone load**: Content cached at edge, traffic stays local
3. **High availability**: Redundancy across servers (failure doesn't affect all users)
4. **Scalability**: Add servers to handle more users

**Key Players**:
- **Akamai**: 300,000+ servers in 130+ countries (largest CDN)
- **Cloudflare**: 300+ data centers globally
- **Google/YouTube**: Internal CDN
- **Netflix/AWS CloudFront**: Video streaming focus

---

### Internet Ecosystem Evolution

**Trend: Topological Flattening**

**Traditional Hierarchy**:
```
Content Origin
     ↓
Tier-1 ISPs (Global Backbone)
     ↓
Tier-2 ISPs (Regional)
     ↓
Tier-3 ISPs (Access)
     ↓
End Users
```

**Modern Reality**:
- Traffic increasingly exchanged **locally** at **IXPs** (Internet Exchange Points)
- CDNs and content providers place servers at IXPs
- Bypasses Tier-1 backbone (cheaper, faster)

**Result**: Tier-1 ISPs carry less traffic; IXPs become central hubs

**Example**:
```
User in Frankfurt watches YouTube video:
  Traditional: Frankfurt → Tier-1 → Google datacenter (California) → back
  Modern CDN: Frankfurt → DE-CIX (IXP) → Google cache server at DE-CIX
  Latency: 200ms → 5ms
```

---

## CDN Server Placement Strategies

### Enter Deep (Akamai)

**Strategy**: Deploy **many small clusters** deep inside access networks (ISPs)

**Characteristics**:
- **Number of locations**: 1000s of sites
- **Cluster size**: 10-100 servers per site
- **Placement**: Inside ISP networks, close to end-users

**Advantages**:
- **Minimal latency**: Servers 1-2 hops from users
- **Maximum locality**: Traffic never leaves ISP network

**Disadvantages**:
- **Management complexity**: Thousands of sites to maintain
- **Cost**: More locations = higher operational cost
- **Underutilization**: Small clusters may be idle during off-peak

**Example**:
```
Akamai server in Comcast network (Philadelphia):
  - Serves only Comcast Philadelphia users
  - Latency: 2-5ms
  - Management: Remote monitoring, occasional on-site maintenance
```

---

### Bring Home (Google, Netflix)

**Strategy**: Deploy **fewer large clusters** at key **IXPs** and major data centers

**Characteristics**:
- **Number of locations**: 10s-100s of sites
- **Cluster size**: 1000s of servers per site
- **Placement**: At major IXPs, Tier-2 ISPs, regional hubs

**Advantages**:
- **Easy management**: Centralized, professional facilities
- **Economies of scale**: Large clusters fully utilized
- **Cost-effective**: Fewer sites to maintain

**Disadvantages**:
- **Slightly higher latency**: Users may be 5-10 hops away (vs. 1-2 for Enter Deep)
- **Less locality**: Traffic crosses more ISP links

**Example**:
```
Google cache server at AMS-IX (Amsterdam):
  - Serves users across Netherlands, Belgium, parts of Germany
  - Latency: 10-20ms (vs. 2-5ms for Akamai)
  - Management: Large data center with on-site staff
```

---

### Comparison Table

| Aspect | Enter Deep (Akamai) | Bring Home (Google, Netflix) |
|--------|---------------------|------------------------------|
| **Locations** | 1000s (ISP networks) | 10s-100s (IXPs, major hubs) |
| **Cluster Size** | Small (10-100 servers) | Large (1000s of servers) |
| **Latency** | Minimal (1-5ms) | Low (10-20ms) |
| **Management** | Complex (many sites) | Simple (few sites) |
| **Cost** | Higher (more sites) | Lower (fewer sites) |
| **Use Case** | General content (web) | Video streaming (bulk traffic) |

---

## Server Selection: How to Choose the Best Server

**Problem**: CDN has 1000 servers; user requests content. Which server should serve the request?

**Two Questions**:
1. **Policy**: What criteria to use? (Geography, load, network metrics)
2. **Protocol**: How to redirect user to chosen server? (DNS, anycast, HTTP)

---

### Selection Policies

**1. Geographic Proximity**:
- Choose server closest to user's **geographic location**
- **Metric**: Physical distance (or IP geolocation database)
- **Advantage**: Simple, fast lookup
- **Disadvantage**: Geography ≠ network distance (fiber routes are not straight lines)

**Example**:
```
User in San Francisco, servers in:
  - Los Angeles (340 miles): 10ms latency
  - Seattle (680 miles): 15ms latency
  → Choose Los Angeles (closer geographically AND better latency)

But consider:
  - London (5300 miles): 80ms latency
  - New York (2500 miles): 90ms latency (due to congested peering link)
  → Closer ≠ always better
```

---

**2. Network Metrics**:
- Choose server with **best network performance**
- **Metrics**: RTT, loss rate, available bandwidth
- **Measurement**: Active probing or passive monitoring
- **Advantage**: Reflects actual network conditions
- **Disadvantage**: Overhead (probing), measurement delay

**Methods**:
- **Real-time probing**: CDN sends pings from servers to user → choose lowest RTT
- **Historical data**: Use past measurements to predict performance

---

**3. Load Balancing**:
- Distribute users evenly across servers to avoid overload
- **Metric**: Server CPU, memory, active connections
- **Example**: Round-robin DNS (cycle through servers)

**Hybrid Approach** (Most CDNs):
- Start with geography (narrow to nearby servers)
- Refine with network metrics (choose best-performing)
- Apply load balancing (avoid overload)

---

## Server Selection Protocols

### 1. DNS-Based Redirection (Primary Method)

**How It Works**:

**Step-by-step**:
1. User requests `www.example.com` (DNS query)
2. DNS resolver queries authoritative name server
3. **CDN's authoritative DNS** receives query
4. DNS server sees **client's IP** (or resolver's IP)
5. DNS server **selects best server** based on client location/metrics
6. DNS returns IP of chosen server
7. User connects directly to that server

**Example**:
```
User in Tokyo requests video.netflix.com:

1. DNS query: video.netflix.com → Where is this?
2. Netflix DNS (authoritative) sees query from Tokyo resolver (IP: 203.0.113.1)
3. Netflix DNS logic:
   - Geolocation: Tokyo → Choose server in Tokyo region
   - Load check: Tokyo-1 (80% load), Tokyo-2 (50% load) → Choose Tokyo-2
4. DNS response: video.netflix.com → 198.51.100.20 (Tokyo-2 server)
5. User's browser connects to 198.51.100.20
6. Video streams from Tokyo-2 server
```

**Advantages**:
- **Transparent**: No changes to client software
- **Standard protocol**: Works everywhere
- **Flexible**: Can change server selection dynamically

**Disadvantages**:
- **Granularity**: DNS sees resolver's IP, not end-user's IP (may be inaccurate if resolver is far from user)
- **TTL caching**: DNS responses cached (TTL = 60-300 seconds), cannot change server immediately
- **No client feedback**: DNS doesn't know if chosen server is actually working

**Optimization: EDNS Client Subnet**:
- DNS query includes user's subnet (not just resolver's IP)
- Improves accuracy when resolver is far from user

---

### 2. IP Anycast

**How It Works**:
- **Multiple servers share the same IP address**
- BGP routes user to **topologically closest** server

**Mechanism**:
1. CDN assigns IP `192.0.2.1` to servers in New York, London, Tokyo
2. Each server announces `192.0.2.1` via BGP
3. User in Paris sends packet to `192.0.2.1`
4. BGP routes packet to London server (closest AS path)

**Advantages**:
- **Automatic failover**: If London server fails, BGP reroutes to next-closest (e.g., New York)
- **DDoS mitigation**: Attack traffic distributed across all servers
- **No DNS overhead**: Same IP for all users

**Disadvantages**:
- **Coarse granularity**: Based on BGP routing (AS-level), not always optimal
- **TCP issues**: If BGP route changes mid-connection, TCP breaks (packets reach different server)
- **Limited control**: Cannot easily override BGP decisions

**Use Case**: Typically for DNS servers themselves (e.g., Cloudflare's 1.1.1.1 uses anycast)

---

### 3. HTTP Redirection

**How It Works**:
1. User requests `http://www.example.com/video.mp4`
2. Server responds with `HTTP 302 Redirect` to `http://cdn-server-5.example.com/video.mp4`
3. User's browser follows redirect, connects to `cdn-server-5.example.com`

**Advantages**:
- **Application-level control**: Server can inspect HTTP headers (user agent, cookies)
- **Dynamic**: Can redirect based on server load in real-time

**Disadvantages**:
- **Extra RTT**: User makes two HTTP requests (original + redirect)
- **Not transparent**: Requires HTTP protocol (doesn't work for raw TCP/UDP apps)

**Use Case**: Load balancing within a data center (rarely used for CDN selection)

---

## Content-to-Server Mapping: Consistent Hashing

**Problem**: CDN has N servers; how to decide which server stores which content?

**Naive Approach: Modulo Hashing**:
```
server_id = hash(content_id) % N

Example: N = 10 servers, content "video123":
  hash("video123") = 987654 → 987654 % 10 = 4 → Server 4
```

**Problem with Modulo Hashing**: When N changes (server added/removed), **all mappings change**

**Example**:
```
N = 10: video123 → Server 4
Server 5 fails, N = 9:
  hash("video123") = 987654 → 987654 % 9 = 6 → Server 6
  → video123 now expected on Server 6, but it's on Server 4!
  → Must move video123 from Server 4 to Server 6
  → ALL content must be remapped (cache miss storm)
```

---

### Consistent Hashing Solution

**Goal**: When N changes, only ~1/N of content needs to move (not all)

**How It Works**:

**1. Hash Space as a Ring**:
- Hash space is a ring [0, 2^32 - 1]
- Wrap around: 2^32 - 1 + 1 = 0

**2. Place Servers on Ring**:
- Hash server ID to get position on ring
- Example: hash("Server1") = 1000, hash("Server2") = 5000, etc.

**3. Place Content on Ring**:
- Hash content ID to get position
- Example: hash("video123") = 3000

**4. Assignment Rule**:
- Content assigned to **first server clockwise** on the ring
- Example: video123 (3000) → next server clockwise is Server2 (5000)

**Diagram**:
```
Ring (0 to 2^32):
        0
        |
    Server3 (7000)
      /         \
  Server1      Server2
  (1000)       (5000)
      \         /
     video123
      (3000)
```

**Lookup**:
- hash("video123") = 3000
- Next server clockwise from 3000 = Server2 (5000)
- Serve video123 from Server2

---

**Handling Server Changes**:

**Server Removed** (e.g., Server2 fails):
```
Before:
  video123 (3000) → Server2 (5000)

After Server2 removed:
  Next server clockwise from 3000 = Server3 (7000)
  → video123 now served by Server3
  → Only content between Server1 and Server2 needs to move
  → ~1/N of content affected (not all)
```

**Server Added** (e.g., Server4 at position 4000):
```
Before:
  video123 (3000) → Server2 (5000)

After Server4 added at 4000:
  Next server clockwise from 3000 = Server4 (4000)
  → video123 now served by Server4
  → Only content between Server1 and Server4 needs to move
  → ~1/N of content affected
```

**Key Property**: Only content in the "affected range" moves; rest stays put

---

**Virtual Nodes** (Load Balancing Enhancement):

**Problem**: Servers may be unevenly distributed on ring (one server handles too much)

**Solution**: Each physical server maps to **multiple virtual nodes**

**Example**:
```
Server1 → virtual nodes at positions 1000, 3000, 6000
Server2 → virtual nodes at positions 2000, 4000, 8000
Server3 → virtual nodes at positions 5000, 7000, 9000

Result: More even distribution of content across servers
```

**Benefits**:
- **Load balancing**: Even distribution even with few servers
- **Smooth scaling**: Adding/removing servers has minimal impact

**Use Case**: Amazon DynamoDB, Cassandra, Memcached

---

## Multimedia Applications

### Categories and Requirements

**1. Streaming Stored Media** (Netflix, YouTube):
- **Content**: Pre-recorded video/audio
- **Interaction**: Play, pause, skip (VCR-like controls)
- **Delay tolerance**: High (buffering up to 10s acceptable)
- **Jitter tolerance**: High (buffer smooths variations)
- **Loss tolerance**: Low (retransmit or error concealment)

**2. Conversational (VoIP, Video Calls)** (Zoom, Skype):
- **Content**: Real-time voice/video
- **Interaction**: Bi-directional conversation
- **Delay tolerance**: **Very low** (<150ms ideal, >400ms unusable)
- **Jitter tolerance**: Low (use jitter buffer, but limited)
- **Loss tolerance**: **High** (some loss acceptable; retransmit too slow)

**3. Streaming Live** (Twitch, sports broadcasts):
- **Content**: Real-time event (live stream)
- **Interaction**: Watch (no control over playback)
- **Delay tolerance**: Medium (5-30s behind real-time acceptable)
- **Jitter tolerance**: Medium (buffering helps)
- **Loss tolerance**: Medium

**Key Insight**: Different applications have **opposite trade-offs**
- Stored streaming: Tolerate delay, need reliability
- Conversational: Tolerate loss, need low delay

---

## VoIP (Voice over IP)

### Encoding: Analog to Digital

**Process**:
1. **Sampling**: Measure analog signal amplitude at regular intervals
   - **Nyquist theorem**: Sample rate ≥ 2× highest frequency
   - Voice frequency: 0-4000 Hz → Sample at 8000 Hz
2. **Quantization**: Round sampled values to discrete levels
   - Example: 8-bit quantization = 256 levels
3. **Encoding**: Convert quantized values to binary

**Example: PCM (Pulse Code Modulation)**:
- Sample rate: 8000 Hz
- Quantization: 8 bits per sample
- Bitrate: 8000 × 8 = 64 Kbps (uncompressed voice)

**Compression**: Reduce bitrate by exploiting redundancy
- **G.729**: 8 Kbps (8:1 compression)
- **Opus**: 6-510 Kbps (adaptive)

---

### VoIP QoS Requirements

**1. End-to-End Delay**:
- **Ideal**: <150ms (imperceptible)
- **Acceptable**: 150-400ms (noticeable but usable)
- **Poor**: >400ms (conversation difficult, interruptions)

**Delay Components**:
- **Encoding**: 5-20ms (codec processing)
- **Packetization**: 10-30ms (accumulate samples into packet)
- **Network propagation**: 5-100ms (depends on distance)
- **Jitter buffer**: 20-100ms (smooth delay variation)
- **Decoding**: 5-20ms

**Total**: Can easily exceed 150ms; requires optimization

---

**2. Jitter (Delay Variation)**:

**Problem**: Packets experience variable delay (queueing, routing changes)

**Example**:
```
Packet 1 sent at t=0, arrives at t=100ms
Packet 2 sent at t=20ms, arrives at t=140ms (should arrive at 120ms)
Jitter = 20ms
```

**Solution: Jitter Buffer** (Playout Buffer):
- Receiver buffers packets before playing
- Delays playout to absorb jitter
- Trade-off: Larger buffer = more jitter tolerance but higher delay

**Algorithm**:
```
On packet arrival:
  timestamp = packet.timestamp
  arrival_time = current_time
  delay = arrival_time - timestamp

  if delay < playout_delay:
    Buffer packet, play at (timestamp + playout_delay)
  else:
    Packet too late, discard (or play immediately)

Adaptive: Adjust playout_delay based on observed jitter
```

---

**3. Packet Loss**:

**Tolerance**: **High** (1-3% loss acceptable for voice)

**Why?**
- Human ear can tolerate some distortion
- Retransmission too slow (delay > 150ms not acceptable)

**Mitigation**:

**Forward Error Correction (FEC)**:
- Send redundant data to recover from loss
- Example: Send XOR of packets; can recover one lost packet
- Trade-off: Increases bandwidth

**Error Concealment**:
- Receiver fills in gaps with estimates
- **Repetition**: Repeat last packet
- **Interpolation**: Estimate based on previous and next packets
- Works well for short gaps (<50ms)

**Example**:
```
Packets: [1] [2] [LOST] [4]

Repetition: Play packet 2 again for gap
Interpolation: Generate packet 3 = average of packet 2 and 4
```

---

### VoIP Signaling: SIP (Session Initiation Protocol)

**Purpose**: Setup, manage, and terminate VoIP calls

**Functions**:
1. **User location**: Find callee's current IP address (may change as user moves)
2. **Session establishment**: Negotiate call parameters (codecs, ports)
3. **Session management**: Modify/terminate call

**SIP Messages**:

**1. INVITE**: Caller initiates call
```
INVITE sip:bob@example.com SIP/2.0
From: alice@example.org
To: bob@example.com
Content-Type: application/sdp

v=0
m=audio 49170 RTP/AVP 0  (audio on port 49170, codec 0=PCM)
```

**2. 200 OK**: Callee accepts call
```
SIP/2.0 200 OK
From: alice@example.org
To: bob@example.com
Content-Type: application/sdp

v=0
m=audio 38010 RTP/AVP 0  (callee's port)
```

**3. ACK**: Caller confirms
```
ACK sip:bob@example.com SIP/2.0
```

**4. BYE**: Terminate call
```
BYE sip:bob@example.com SIP/2.0
```

**Result**: Both parties know each other's IP, port, codec → Start sending RTP packets

---

## Video Streaming

### Video Compression

**Challenge**: Uncompressed video requires enormous bandwidth

**Example**:
```
1080p video (1920×1080 pixels, 30 fps, 24-bit color):
  = 1920 × 1080 × 30 × 24 bits/sec
  = 1.5 Gbps (uncompressed)

Compressed (H.264):
  = 5 Mbps (300:1 compression)
```

**Two Types of Redundancy**:

---

**1. Spatial Redundancy** (Within a single frame):

**Observation**: Adjacent pixels are similar (sky is all blue)

**Technique: DCT (Discrete Cosine Transform)**:
- Divide frame into blocks (e.g., 8×8 pixels)
- Transform pixel values to frequency domain
- High-frequency components (details) quantized more aggressively
- JPEG uses DCT for image compression

**Example**:
```
8×8 block of similar pixels (all ~100):
  [100, 101, 99, 100, 100, 101, 100, 99, ...]

DCT transform:
  [800, 1, 0, 0, 0, 0, 0, 0, ...]  (DC component + tiny AC components)

Quantization:
  [800, 0, 0, 0, 0, 0, 0, 0]  (zeros compress easily)
```

---

**2. Temporal Redundancy** (Across frames):

**Observation**: Consecutive frames are similar (most pixels don't change)

**Technique: Motion Compensation**:
- Encode difference between frames instead of full frame
- Track motion of objects (motion vectors)

**Frame Types**:

**I-frame (Intra-coded / Independent)**:
- Complete frame (no reference to other frames)
- Uses spatial compression only (JPEG-like)
- Large size (can be decoded independently)

**P-frame (Predicted)**:
- Encoded as difference from **previous I- or P-frame**
- Uses motion compensation
- Smaller size (depends on previous frame)

**B-frame (Bi-directional)**:
- Encoded as difference from **previous AND next frames**
- Highest compression
- Requires future frame (decoding delay)

**Example GOP (Group of Pictures)**:
```
I  B  B  P  B  B  P  B  B  I
↑     ↗  ↑     ↗  ↑     ↗  ↑
|   ↙    |   ↙    |   ↙    |
Independent  References I and P

I-frame: 50 KB
P-frame: 10 KB (80% reduction)
B-frame: 5 KB (90% reduction)
```

**Trade-off**:
- More B-frames = Higher compression but higher decoding complexity
- I-frame intervals: Every 1-2 seconds (allows seeking, error recovery)

---

## DASH (Dynamic Adaptive Streaming over HTTP)

### Architecture

**Traditional Streaming** (RTP/RTSP):
- Dedicated streaming servers
- Custom protocols
- Difficult to deploy (firewalls block)

**DASH** (Modern approach):
- Use **HTTP** (standard web protocol)
- Use **TCP** (reliable, firewall-friendly)
- Servers are **stateless** (standard web servers, CDNs)

**How It Works**:

**1. Content Preparation** (Server-side):
- Encode video at **multiple bitrates** (e.g., 500 Kbps, 1 Mbps, 5 Mbps, 10 Mbps)
- Chop each version into **chunks** (e.g., 2-10 seconds each)
- Generate **MPD (Media Presentation Description)** manifest file listing all chunks and bitrates

**2. Adaptive Playback** (Client-side):
- Client downloads MPD
- Client requests chunks one at a time (HTTP GET)
- Client **chooses bitrate** for each chunk based on:
  - Estimated bandwidth
  - Current buffer occupancy
  - Device capabilities

**Example**:
```
MPD file:
  Video ID: "movie123"
  Chunks: 0-299 (each 4 seconds, total 20 minutes)
  Bitrates:
    500 Kbps: chunk-0-500k.mp4, chunk-1-500k.mp4, ...
    2 Mbps: chunk-0-2000k.mp4, chunk-1-2000k.mp4, ...
    5 Mbps: chunk-0-5000k.mp4, chunk-1-5000k.mp4, ...

Client playback:
  Chunk 0: Request chunk-0-2000k.mp4 (2 Mbps)
  Chunk 1: Bandwidth good → Request chunk-1-5000k.mp4 (5 Mbps)
  Chunk 2: Bandwidth dropped → Request chunk-2-500k.mp4 (500 Kbps)
  ...
```

**Benefits**:
- **No buffering**: Client adapts to available bandwidth
- **CDN-friendly**: Standard HTTP, cached by CDNs
- **Firewall-friendly**: HTTP/TCP, same as web traffic

---

### Bitrate Adaptation Algorithms

**Goal**: Maximize video quality while avoiding re-buffering (playback stalls)

**Two Main Approaches**:

---

**1. Rate-Based Adaptation**:

**Idea**: Estimate future bandwidth based on past throughput; choose bitrate accordingly

**Algorithm**:
```
1. Measure throughput for previous chunk download
2. Estimate future bandwidth = moving average of past throughput
3. Choose chunk bitrate ≤ estimated bandwidth (with safety margin)

Example:
  Past chunk: 5 Mbps throughput
  Estimated future: 5 Mbps × 0.8 = 4 Mbps (safety margin)
  Choose chunk bitrate: 2 Mbps (highest ≤ 4 Mbps)
```

**Advantages**:
- Simple to implement
- Reacts quickly to bandwidth changes

**Disadvantages**:
- **Underestimation**: TCP's ON-OFF behavior (bursty) leads to conservative estimates
  - TCP fetches chunk at full speed → link idle while playing → next chunk underestimates bandwidth
- **Unstable**: Oscillates between high and low bitrates

**Example Problem**:
```
Real bandwidth: 5 Mbps (steady)

Chunk 1: Download at 5 Mbps (1 second download, 3 seconds idle)
  → Measured throughput: 5 Mbps
  → Choose 5 Mbps bitrate for chunk 2

Chunk 2: Download at 5 Mbps (1 second)
  → But algorithm only sees "active download time"
  → Underestimates future bandwidth as 2 Mbps (because link was idle 75% of the time)
  → Choose 2 Mbps bitrate for chunk 3

Result: Switches to low bitrate unnecessarily
```

---

**2. Buffer-Based Adaptation**:

**Idea**: Choose bitrate based on **current buffer occupancy** (how much video is buffered ahead)

**Algorithm**:
```
1. Check buffer occupancy (seconds of video buffered)
2. Map buffer level to bitrate:
   - Buffer low (<10s): Choose low bitrate (avoid re-buffering)
   - Buffer medium (10-30s): Choose medium bitrate
   - Buffer high (>30s): Choose high bitrate (quality)

Example:
  Buffer = 5s: Choose 500 Kbps (low bitrate to refill buffer fast)
  Buffer = 20s: Choose 2 Mbps (medium)
  Buffer = 40s: Choose 5 Mbps (high quality, buffer is safe)
```

**Advantages**:
- **Stable**: No oscillations (buffer changes slowly)
- **Avoids re-buffering**: Prioritizes buffer health over quality
- **No bandwidth estimation**: Sidesteps TCP ON-OFF measurement issues

**Disadvantages**:
- **Slow reaction**: Takes time for buffer to adjust after bandwidth change
- **Potential for oscillation**: If not tuned well, buffer can fluctuate

**Hybrid Approaches** (Modern Players):
- Use rate-based for quick reaction to sudden changes
- Use buffer-based for stability
- Add device constraints (screen resolution, CPU)

---

## Summary

### Key Takeaways

1. **CDNs**:
   - Solve latency and scalability by replicating content geographically
   - **Placement strategies**: Enter Deep (many small clusters) vs. Bring Home (few large clusters)
   - **Selection**: DNS-based redirection (primary), anycast (for DNS servers), HTTP redirect (rare)
   - **Content mapping**: Consistent hashing minimizes data movement when servers change

2. **Multimedia Categories**:
   - **Stored streaming**: High delay tolerance, low loss tolerance (buffering works)
   - **Conversational (VoIP)**: Low delay tolerance (<150ms), high loss tolerance (error concealment)
   - **Live streaming**: Medium delay/loss tolerance

3. **VoIP**:
   - **Encoding**: Analog → digital via sampling, quantization
   - **QoS**: Jitter buffer smooths delay variation; FEC/concealment handles loss
   - **Signaling**: SIP establishes calls, negotiates parameters

4. **Video Compression**:
   - **Spatial**: DCT within frames (JPEG-like)
   - **Temporal**: I/P/B frames exploit inter-frame similarity (motion compensation)
   - Achieves 100-300:1 compression

5. **DASH**:
   - Uses HTTP/TCP for compatibility (CDNs, firewalls)
   - Multiple bitrate versions of content (chunks)
   - **Rate-based adaptation**: Bandwidth estimation (fast but unstable)
   - **Buffer-based adaptation**: Buffer occupancy (stable, avoids re-buffering)

### Common Patterns

**CDN Design**:
- Geographic distribution → Low latency
- Consistent hashing → Scalable content mapping
- DNS redirection → Transparent to users

**Multimedia Trade-offs**:
- Delay vs. Loss tolerance (VoIP vs. streaming)
- Quality vs. Re-buffering (bitrate adaptation)
- Compression vs. Complexity (I/P/B frames)

**Adaptation Algorithms**:
- **Reactive**: Measure and adapt (rate-based)
- **Proactive**: Plan ahead (buffer-based)
- **Hybrid**: Combine both for best results

---

## See Also

- [[02-Network-Layer-and-Routing]] - BGP, IXPs, and Internet topology
- [[03-Transport-Layer]] - TCP congestion control (affects streaming)
- [[04-Advanced-Routing-and-QoS]] - QoS mechanisms for VoIP

**Next**: [[07-Security-and-Governance]]
