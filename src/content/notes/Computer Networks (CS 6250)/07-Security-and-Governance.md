---
type: source
course: "[[Computer Networks (CS 6250)]]"
created: 2026-02-01
prerequisites: "[[02-Network-Layer-and-Routing]]"
---

# Security and Governance

> **Prerequisites**: [[02-Network-Layer-and-Routing]]
> **Learning Goals**: After reading this, you will understand the four fundamental security properties, DNS abuse mechanisms (Fast-Flux), BGP hijacking types and defenses (ARTEMIS), DDoS attacks (reflection/amplification), mitigation techniques (BGP Flowspec, Blackholing), and Internet censorship methods and detection systems.

## Introduction

The Internet was designed for connectivity, not security. Early protocols (IP, BGP, DNS) assumed trusted participants, making them vulnerable to abuse. This file covers attacks that exploit protocol weaknesses and the defenses developed to mitigate them. We examine three domains: **DNS abuse** (hiding malicious infrastructure), **BGP hijacking** (stealing IP prefixes), **DDoS** (overwhelming targets), and **censorship** (governments restricting access).

**Key Insight**: Security often trades off with openness, flexibility, and performance.

---

## Fundamental Security Properties

### The Four Pillars

**1. Confidentiality**:
- **Definition**: Information is accessible only to authorized parties
- **Threat**: Eavesdropping (passive attacker reads data)
- **Defense**: Encryption (symmetric: AES; asymmetric: RSA, ECC)

**Example**:
```
Without confidentiality: HTTP sends password in plaintext
With confidentiality: HTTPS encrypts password using TLS
```

---

**2. Integrity**:
- **Definition**: Data is not modified in transit (or modifications detected)
- **Threat**: Tampering (active attacker changes data)
- **Defense**: Cryptographic hashes (SHA-256), MACs (HMAC)

**Example**:
```
Without integrity: Attacker changes "Transfer $100" to "Transfer $10,000"
With integrity: MAC detects modification, message rejected
```

---

**3. Authentication**:
- **Definition**: Verify the identity of communicating parties
- **Threat**: Impersonation (attacker pretends to be legitimate user/server)
- **Defense**: Passwords, certificates (PKI), digital signatures

**Example**:
```
Without authentication: Attacker creates fake "bankofamerica.com" site
With authentication: TLS certificate proves identity (signed by trusted CA)
```

---

**4. Availability**:
- **Definition**: Services remain accessible despite failures or attacks
- **Threat**: Denial of Service (attacker overwhelms system)
- **Defense**: Redundancy, rate limiting, filtering (blackholing)

**Example**:
```
Without availability: DDoS attack overwhelms server, legitimate users blocked
With availability: CDN distributes load, DDoS traffic filtered
```

**Note**: Availability is the hardest to guarantee (requires resources to absorb attacks).

---

## DNS Abuse

### DNS Basics Revisited

**Function**: Translate domain names to IP addresses

**Normal Operation**:
```
User queries DNS: "What is example.com?"
DNS response: "example.com = 93.184.216.34"
User connects to 93.184.216.34
```

**Abuse**: Attackers exploit DNS to hide infrastructure or distribute load.

---

### Round Robin DNS (RRDNS)

**Legitimate Use**: Simple load balancing

**How It Works**:
- DNS returns **multiple IP addresses** for one domain
- IPs rotated in response order (round-robin)
- Clients connect to the first IP in the list

**Example**:
```
example.com → [1.1.1.1, 2.2.2.2, 3.3.3.3]

Query 1: Response = [1.1.1.1, 2.2.2.2, 3.3.3.3]
Query 2: Response = [2.2.2.2, 3.3.3.3, 1.1.1.1]  (rotated)
Query 3: Response = [3.3.3.3, 1.1.1.1, 2.2.2.2]

Result: Load distributed across servers
```

**Abuse**: Attackers use RRDNS to rapidly change infrastructure (harder to block).

---

### Fast-Flux Service Networks (FFSN)

**Goal** (Attacker's perspective): Hide malicious servers (phishing, malware C&C) from takedown

**Traditional Malicious Setup**:
```
evil.com → 6.6.6.6 (malicious server)

Problem: Authorities find 6.6.6.6 → Shut down server → evil.com offline
```

**Fast-Flux Solution**:

**Architecture**:
1. **Mothership**: Actual malicious server (hidden, stable IP)
2. **Flux Agents**: Compromised machines (bots) acting as proxies (IPs change rapidly)
3. **DNS**: Returns flux agent IPs with **very short TTL** (seconds to minutes)

**Operation**:
```
evil.com DNS response (TTL = 300 seconds):
  Query 1 (time 0): evil.com → [10.0.0.1, 10.0.0.2]  (flux agents)
  Query 2 (time 300): evil.com → [10.0.0.5, 10.0.0.9]  (different agents)
  Query 3 (time 600): evil.com → [10.0.0.15, 10.0.0.20]  (changed again)

User connects to 10.0.0.1 (flux agent) → Proxies request to mothership (6.6.6.6)
```

**Why Effective**:
- **Rapid IP changes**: Short TTL (5-10 minutes) means IPs change faster than authorities can block
- **Distributed**: Blocking one flux agent doesn't affect others
- **Mothership hidden**: Real server never directly contacted by victims (hard to find)

**Detection**:
- **High DNS query rate**: Frequent queries due to short TTL
- **Many unique IPs**: Hundreds of IPs for one domain
- **Short TTL**: <1 hour (legitimate CDNs use 1-24 hours)
- **Geographic diversity**: IPs from many countries (compromised bots)

---

### Single-Flux vs. Double-Flux

**Single-Flux**: Only the **A records** (domain → IP) change rapidly
- Authoritative DNS servers are stable

**Double-Flux**: Both **A records** AND **NS records** (authoritative name servers) change rapidly
- Even DNS servers are part of the botnet
- Harder to take down (no stable DNS server to target)

**Example (Double-Flux)**:
```
evil.com NS records:
  Time 0: ns1.evil.com → 8.8.8.1 (flux DNS server)
  Time 300: ns1.evil.com → 8.8.8.5 (different flux DNS server)

Result: Both web servers AND DNS servers rotate rapidly
```

---

## Network Reputation: Identifying Malicious Networks

### FIRE (FInding Rogue nEtworks)

**Goal**: Identify malicious ASes (Autonomous Systems) hosting bad content

**Data Plane Approach**: Monitor malicious activity sources

**Data Sources**:
1. **Botnet C&C servers**: IPs hosting command and control
2. **Drive-by-download sites**: Websites with malware
3. **Phishing pages**: Fake login pages

**Key Insight**: Malicious content has **shorter lifespan** than legitimate content
- Legitimate: example.com hosted for years
- Malicious: phishing page taken down within days

**FIRE Algorithm**:
1. Collect malicious IPs from multiple sources (blacklists, honeypots)
2. Map IPs to ASes (BGP routing tables)
3. Compute **reputation score** per AS based on:
   - **Number of malicious IPs** hosted
   - **Longevity of malicious content** (how long before takedown)
   - **Diversity of attacks** (many attack types → worse)

**Result**: ASes with low scores are "rogue networks" (hosting attackers)

**Use Case**: ISPs can depeer (disconnect) from rogue ASes; blacklist traffic

---

### ASwatch

**Goal**: Identify "bulletproof hosting" ASes from control plane behavior

**Bulletproof Hosting**: ASes that ignore abuse complaints, allow malicious activity

**Control Plane Approach**: Analyze **BGP announcements** for suspicious patterns

**Observation**: Rogue ASes exhibit distinct wiring patterns

**Characteristics of Rogue ASes**:
1. **Frequent rewiring**: Change upstream providers often (to evade blocking)
2. **Short-lived routes**: BGP announcements withdrawn frequently
3. **New ASes**: Recently allocated ASNs (evade reputation)
4. **Small customer cone**: Few or no downstream customers (not a legitimate ISP)

**Example**:
```
Legitimate AS (Google AS15169):
  - Stable provider relationships (same upstreams for years)
  - Large customer cone (millions of users)
  - Long-lived routes

Rogue AS (AS12345):
  - Provider changes weekly: Provider A → Provider B → Provider C
  - No downstream customers (only hosts malicious servers)
  - BGP announcements appear/disappear daily
```

**ASwatch Algorithm**:
- Track BGP updates over time
- Compute features: rewiring frequency, route stability, customer cone size
- Classify ASes as rogue if features match bulletproof hosting patterns

**Validation**: Cross-reference with known malicious IPs (FIRE data) - high correlation

---

## BGP Hijacking

### Threat Model

**BGP Reminder**: ASes announce IP prefixes they own; routers propagate announcements

**Trust Assumption**: BGP has **no authentication** - any AS can announce any prefix

**Attack**: Announce a prefix you don't own to **steal traffic**

---

### Types of BGP Hijacking

**1. Exact Prefix Hijacking**:

**Attack**: Announce the **same prefix** as the legitimate owner

**Example**:
```
Legitimate: AS 100 announces 192.168.0.0/16
Attacker: AS 200 announces 192.168.0.0/16

Result:
  - Routers see two announcements for same prefix
  - Choose based on AS path length (shorter wins)
  - If AS 200's path is shorter → Traffic diverted to AS 200
```

**Impact**:
- **Partial hijack**: Only ASes closer to attacker see the malicious route
- **Blackhole**: Attacker drops traffic (DoS)
- **Man-in-the-Middle**: Attacker intercepts, inspects, forwards to legitimate destination

---

**2. Sub-Prefix Hijacking** (More Specific Prefix):

**Attack**: Announce a **more specific prefix** (longer prefix length)

**Example**:
```
Legitimate: AS 100 announces 192.168.0.0/16
Attacker: AS 200 announces 192.168.1.0/24 (sub-prefix of /16)

Result:
  - Longest prefix match: 192.168.1.0/24 is more specific
  - ALL routers prefer /24 over /16 (regardless of AS path)
  - Traffic to 192.168.1.0/24 goes to AS 200 (even if AS 100's path is shorter)
```

**Why Effective**: Exploits **longest prefix match** (fundamental to IP routing)

**Impact**:
- **Full hijack**: Traffic from all ASes diverted (more severe than exact prefix)
- **Hard to detect**: Looks like legitimate route announcement

---

**3. Squatting**:

**Attack**: Announce an **unallocated prefix** (not owned by anyone)

**Example**:
```
192.168.0.0/16 is unallocated (no one owns it)
Attacker: AS 200 announces 192.168.0.0/16

Result:
  - Routers accept announcement (no conflicting announcement)
  - AS 200 "claims" the prefix
```

**Use Case**: Attacker wants temporary IP space (e.g., for spam, DDoS)

---

### Real-World Examples

**Pakistan Telecom vs. YouTube (2008)**:
- Pakistan government ordered ISPs to block YouTube (domestic censorship)
- Pakistan Telecom announced **208.65.153.0/24** (YouTube's /24)
- Announcement leaked to global Internet (mistake)
- Result: YouTube unreachable worldwide for 2 hours

**China Telecom (2010)**:
- Announced **50,000+ prefixes** belonging to others (likely misconfiguration)
- Included US military and government networks
- Traffic rerouted through China for 15-20 minutes
- Concern: Espionage opportunity (man-in-the-middle)

---

## BGP Hijacking Defense: ARTEMIS

**Goal**: Detect and mitigate BGP hijacking in real-time

**Architecture**: Deployed at the **victim AS** (self-defense)

**Components**:

**1. Configuration File**:
- List of **legitimate prefixes** owned by the AS
- List of **trusted peers/upstreams**

**2. BGP Monitoring**:
- Listen to BGP announcements from route collectors (RouteViews, RIPE RIS)
- Detect announcements of own prefixes from other ASes

**3. Detection Logic**:
```
If announcement for owned prefix detected:
  Check origin AS:
    If origin AS != self → Potential hijack
  Check announcement type:
    Exact prefix → Type-0 hijack
    Sub-prefix → Type-1 hijack (more severe)
    Super-prefix → Type-2 hijack
  Alert operator
```

**4. Mitigation**:

**Prefix Deaggregation** (Fight fire with fire):
- If attacker announces 192.168.1.0/24 (sub-prefix hijack)
- Victim also announces 192.168.1.0/24 (legitimate owner)
- Now both announce same prefix → AS path determines winner
- Victim's path often shorter (legitimate peering) → Traffic restored

**Example**:
```
Normal: Victim announces 192.168.0.0/16

Attack: Attacker announces 192.168.1.0/24
  → Traffic to 192.168.1.0/24 hijacked (longest prefix match)

ARTEMIS Mitigation: Victim also announces 192.168.1.0/24
  → Now both announce /24
  → AS path tie-breaking: Victim's path is 2 hops, Attacker's is 4 hops
  → Victim wins (shorter path)
  → Traffic restored
```

**MOAS (Multiple Origin AS) Announcement**:
- Announce prefix from **multiple ASes** (if victim has multiple locations)
- Provides redundancy and faster propagation

**Advantages**:
- **Fast detection**: Seconds to minutes (BGP feed is real-time)
- **Automated mitigation**: No manual intervention needed
- **No protocol changes**: Works with existing BGP

**Limitations**:
- **Reactive**: Detects after hijack starts (not preventive)
- **Requires deployment**: Each AS must run ARTEMIS (not global solution)

---

## DDoS (Distributed Denial of Service)

### Attack Overview

**Goal**: Overwhelm target with traffic, making it unavailable

**Types**:

**1. Volumetric**: Flood target with traffic to saturate bandwidth
**2. Application-Layer**: Exhaust server resources (CPU, memory) with expensive requests
**3. Protocol Exploitation**: Exploit protocol weaknesses (SYN flood, amplification)

**This section focuses on**: **Reflection and Amplification** (volumetric)

---

### IP Spoofing

**Foundation of many DDoS attacks**: Fake the source IP address

**How It Works**:
```
Attacker's real IP: 1.1.1.1
Victim's IP: 2.2.2.2

Attacker sends packet with:
  Source IP: 2.2.2.2 (spoofed - pretends to be victim)
  Destination IP: 3.3.3.3 (reflector)

Reflector sees request from 2.2.2.2 → Responds to 2.2.2.2 (victim)
```

**Why It Works**: IP has no source authentication (design flaw)

**Defense: Ingress Filtering (BCP 38)**:
- ISPs check if source IP is valid for customer
- Example: Comcast customer's IP should be in Comcast's range
- If packet claims source IP outside Comcast → Drop

**Problem**: Not universally deployed (some ISPs don't filter)

---

### Reflection Attacks

**Goal**: Hide attacker's identity, amplify traffic

**Mechanism**:
1. Attacker sends requests to **reflectors** (legitimate servers) with **spoofed source IP = victim**
2. Reflectors respond to victim (who didn't request anything)
3. Victim overwhelmed by responses

**Example**:
```
Attacker (1.1.1.1) → DNS server (8.8.8.8)
  Packet: Source=2.2.2.2 (victim), Dest=8.8.8.8, Query="example.com"

DNS server → Victim (2.2.2.2)
  Packet: Source=8.8.8.8, Dest=2.2.2.2, Response="example.com = ..."

Victim receives unsolicited DNS response
```

**Amplification**: If response is **larger than request**, attacker amplifies traffic

---

### Amplification Attacks

**Key Metric: Amplification Factor** = Response Size / Request Size

**Example Protocols and Factors**:

| Protocol | Request Size | Response Size | Amplification Factor |
|----------|--------------|---------------|----------------------|
| DNS      | 60 bytes     | 3000 bytes    | 50x                  |
| NTP      | 8 bytes      | 468 bytes     | 58x                  |
| Memcached| 15 bytes     | 750 KB        | 51,000x              |
| SNMP     | 50 bytes     | 1500 bytes    | 30x                  |

**Attack Scenario (DNS Amplification)**:
```
Attacker controls botnet of 1,000 machines
Each bot sends 1 Mbps of DNS requests (with spoofed source = victim)

Without amplification:
  Victim receives 1,000 Mbps = 1 Gbps

With 50x amplification:
  Victim receives 1,000 Mbps × 50 = 50 Gbps

Result: Attacker generates 50 Gbps attack with only 1 Gbps of outbound bandwidth
```

**Why DNS?**
- **Open resolvers**: Publicly accessible DNS servers respond to anyone
- **Large responses**: DNSSEC, ANY queries return kilobytes
- **Ubiquitous**: DNS servers everywhere (easy to find reflectors)

**Famous Attack: Memcached (2018)**:
- GitHub hit with **1.35 Tbps** DDoS (largest at the time)
- Amplification factor: 51,000x
- Attacker needed only ~26 Mbps to generate 1.35 Tbps

---

### DDoS Mitigation Techniques

**Challenge**: Distinguish legitimate traffic from attack traffic

**Mitigation is hard because**:
- Volumetric attacks can exceed victim's capacity (nothing victim can do alone)
- Reflection uses legitimate servers (blocking reflectors harms innocents)

---

**1. BGP Flowspec** (Flow Specification):

**Goal**: Fine-grained traffic filtering propagated via BGP

**How It Works**:
- **Traditional BGP**: Announces IP prefixes (coarse-grained)
- **Flowspec**: Announces **traffic flow rules** (match on multiple fields)

**Flow Rule Example**:
```
Match:
  Source IP: 6.6.6.0/24 (attacker's subnet)
  Destination IP: 2.2.2.2 (victim)
  Destination Port: 80 (HTTP)
  Protocol: TCP
  Packet Length: > 1000 bytes (large packets)

Action:
  Drop (or rate-limit to 1 Mbps)
```

**Propagation**:
1. Victim AS detects attack traffic (using IDS, traffic analysis)
2. Victim creates Flowspec rule matching attack pattern
3. Victim announces rule via **BGP Flowspec** to upstream ISPs
4. ISPs install rule in routers → Drop/rate-limit matching traffic **before it reaches victim**

**Benefits**:
- **Distributed filtering**: Attack traffic dropped upstream (saves victim's bandwidth)
- **Surgical**: Can target specific flows (not all traffic to victim)
- **Automated**: No manual ISP coordination needed

**Limitations**:
- **Requires support**: Routers must support Flowspec (modern routers do)
- **False positives**: Broad rules may drop legitimate traffic
- **Limited scope**: Only works for ASes that propagate the rule

---

**2. BGP Blackholing** (RTBH - Remotely Triggered Black Hole):

**Goal**: Drop **all traffic** to a specific destination (victim)

**How It Works**:
1. Victim AS announces victim's IP with **special BGP community tag** (e.g., 666)
2. ISPs recognize tag → Install **blackhole route** (drop all traffic to that IP)
3. Result: Attack traffic dropped before reaching victim

**Example**:
```
Normal: Victim AS announces 2.2.2.2/32 → Route traffic normally

Under attack:
  Victim AS announces 2.2.2.2/32 with BGP community 666 (blackhole tag)
  ISPs see tag → Install rule: "DROP all traffic to 2.2.2.2"

Result:
  - Attack traffic dropped at ISP edge (victim's bandwidth saved)
  - But ALSO legitimate traffic dropped (collateral damage)
```

**Benefits**:
- **Fast**: Can be triggered in seconds
- **Effective**: Stops attack immediately (no traffic reaches victim)
- **Simple**: Widely supported (standard BGP communities)

**Limitations**:
- **Collateral damage**: Victim becomes **completely unreachable** (kills the patient to cure the disease)
- **Not surgical**: Cannot distinguish legitimate from attack traffic
- **Use case**: Last resort when victim is already overwhelmed (sacrifice availability to protect infrastructure)

**Granularity**:
- Can blackhole specific /32 (single IP) to minimize collateral damage
- Cannot blackhole specific ports or protocols (coarse-grained)

---

## Internet Censorship

### Techniques

**Censorship**: Government or organization restricts access to information

**Three Layers**:

---

**1. DNS Censorship** (Manipulation):

**Method**: Prevent domain name resolution or return fake IP

**Example: Great Firewall of China (GFW)**:
```
User in China queries: "What is twitter.com?"

Normal DNS: twitter.com → 104.244.42.1
GFW Injection: twitter.com → 127.0.0.1 (localhost - fake response)

User's browser connects to 127.0.0.1 → Connection fails
```

**How GFW Works**:
- **On-path injection**: GFW sits between user and DNS server
- GFW sees DNS query for blocked domain → Sends fake response **faster than real server**
- User's resolver accepts first response (fake) → Ignores later real response

**Why Effective**:
- **Invisible**: User sees "DNS lookup failed" (looks like network problem)
- **Fast**: Blocks before real response arrives
- **Scalable**: One GFW box can monitor millions of DNS queries

**Bypass**: Use encrypted DNS (DNS-over-HTTPS, DNS-over-TLS) - GFW cannot see queries

---

**2. Packet Dropping/Filtering**:

**Method**: Block traffic to specific IPs or containing keywords

**Example**:
```
Block all traffic to IP 1.2.3.4 (website server)
  → Firewall rule: DROP packets with destination=1.2.3.4

Block traffic containing keyword "protest"
  → Deep Packet Inspection (DPI): Scan packet payload → DROP if contains "protest"
```

**Granularity**:
- **IP-based**: Simple but causes **overblocking** (one IP may host many sites)
- **Keyword-based**: Precise but **expensive** (requires DPI)

**Collateral Damage**:
```
Block IP 1.2.3.4 (hosting blocked blog)
But 1.2.3.4 also hosts 100 other sites (shared hosting)
→ All 100 sites blocked (overblocking)
```

---

**3. TCP Resets (Connection Termination)**:

**Method**: Send forged TCP RST packets to tear down connections

**Example: GFW Keyword Filtering**:
```
User in China connects to website (foreign server)
Connection established (TCP handshake complete)

User sends HTTP request: "GET /article-about-censorship HTTP/1.1"

GFW inspects packet payload → Detects keyword "censorship"
GFW sends forged TCP RST to both user and server:
  To user: Source=server IP, Dest=user IP, RST flag set
  To server: Source=user IP, Dest=server IP, RST flag set

Both sides receive RST → Close connection (think other side terminated)

User sees: "Connection reset by peer"
```

**Why Effective**:
- **Stateless**: GFW doesn't need to track connections (just inspects packets)
- **Fast**: Immediate termination (no prolonged blocking)
- **Hard to detect**: Looks like normal network error

**Limitations**:
- **Requires on-path position**: GFW must see traffic (doesn't work for encrypted connections)
- **Bypass**: Use encryption (TLS/VPN) - GFW cannot inspect payload to find keywords

---

**4. BGP Hijacking (Routing Disruption)**:

**Method**: Withdraw BGP routes to make networks unreachable

**Example: Egypt 2011**:
```
Egyptian government orders ISPs to disconnect Internet during protests

Method:
  - ISPs withdraw BGP announcements for Egyptian prefixes
  - Global routers remove routes to Egypt
  - Result: Egypt's networks unreachable from outside world

Duration: 5 days (Jan 27 - Feb 2, 2011)
```

**Detection**: BGP monitoring systems (RouteViews, RIPE RIS) saw sudden withdrawal of routes

---

## Censorship Detection

**Challenge**: Measure censorship from outside the censored country (no direct access)

---

### Iris: DNS Manipulation Detection

**Goal**: Detect DNS censorship by comparing responses from different resolvers

**Method**: Use **open DNS resolvers** worldwide as vantage points

**How It Works**:
1. Identify open resolvers in many countries (scan for port 53)
2. Send DNS queries for suspected censored domains to all resolvers
3. Compare responses:
   - **Consistent**: All resolvers return same IP → Not censored
   - **Inconsistent**: Resolvers in country X return different IP → Likely censored

**Example**:
```
Query: "What is bbc.com?"

Resolvers in US, EU, Japan: bbc.com → 151.101.0.81
Resolvers in China: bbc.com → 127.0.0.1 (or no response)

Conclusion: China censors bbc.com via DNS manipulation
```

**Validation Metrics**:

**1. Consistency**:
- Are responses consistent across resolvers in same region?
- Inconsistency → Potential censorship

**2. Independent Verifiability**:
- Can we reach the returned IP from outside the censored region?
- If returned IP is unreachable globally → Fake response

**Limitations**:
- **Relies on open resolvers**: If none exist in target country, cannot measure
- **DNS only**: Doesn't detect IP-based blocking or content filtering

---

### Augur: Connectivity Disruption Detection

**Goal**: Detect if two hosts can communicate (even without access to either host)

**Challenge**: We control neither host; how to test connectivity?

**Method**: Exploit **TCP/IP side channels** (Global IP ID counter)

---

**IP ID Side Channel**:

**Observation**: Some hosts use a **global counter** for IP packet ID field
- Each packet increments counter: Packet 1 (ID=100), Packet 2 (ID=101), ...

**Inference**:
```
If host's IP ID counter increments → Host sent a packet recently
If counter doesn't increment → Host didn't send a packet (or is offline)
```

**Augur Algorithm** (Test if host A can reach host B):

**1. Measure baseline**:
- Send probe to A → Measure A's IP ID (e.g., ID=1000)
- Send probe to A again after 1 second → Measure ID=1010
- Inference: A sent ~10 packets in 1 second (background traffic)

**2. Trigger communication**:
- Send **spoofed SYN packet** from A to B (spoofed source IP = A)
- If A and B can communicate:
  - B receives SYN from A → Sends SYN-ACK to A
  - A receives SYN-ACK (unexpected) → Sends RST to B
  - A's IP ID increments (sent RST packet)

**3. Measure after trigger**:
- Send probe to A → Measure A's IP ID (e.g., ID=1015)
- Expected: ID=1020 (10 background + expected increment)
- If ID=1020 → A sent extra packet (RST) → A can reach B
- If ID=1015 → A didn't send RST → A cannot reach B (blocked)

**Example**:
```
Test: Can host in Iran (A) reach Twitter server (B)?

Step 1: Baseline IP ID for A: 1000 → 1010 (1 second)
Step 2: Send spoofed SYN from A to B
Step 3: Measure A's IP ID after 1 second: 1015

Expected if reachable: 1020 (10 background + RST)
Actual: 1015 → No RST sent → A cannot reach B (blocked)

Conclusion: Iran blocks access to Twitter
```

**Advantages**:
- **No access needed**: Don't need to control A or B
- **Stealthy**: Measurement looks like normal probing (hard to detect)
- **Broad applicability**: Works for any two hosts (if A uses global IP ID)

**Limitations**:
- **Requires global IP ID**: Modern hosts use random IP ID (side channel doesn't work)
- **Inference only**: Cannot definitively prove blocking (could be network failure)

---

## Summary

### Key Takeaways

1. **Security Properties**:
   - **Confidentiality**: Encryption (TLS)
   - **Integrity**: Hashes/MACs
   - **Authentication**: Certificates (PKI)
   - **Availability**: Hardest to guarantee (requires resources)

2. **DNS Abuse**:
   - **RRDNS**: Simple load balancing (or evasion)
   - **Fast-Flux**: Rapidly changing IPs to hide infrastructure (short TTL, many IPs)
   - **Detection**: High query rate, many unique IPs, short TTL

3. **BGP Hijacking**:
   - **Types**: Exact prefix, sub-prefix (most effective), squatting
   - **ARTEMIS Defense**: Real-time detection + prefix deaggregation mitigation
   - **No authentication**: BGP trusts all announcements (fundamental flaw)

4. **DDoS**:
   - **Reflection/Amplification**: Attacker uses legitimate servers to amplify traffic (up to 51,000x)
   - **Mitigation**: BGP Flowspec (surgical filtering), Blackholing (collateral damage)
   - **Spoofing**: Ingress filtering helps but not universally deployed

5. **Censorship**:
   - **Techniques**: DNS injection (GFW), packet dropping, TCP resets, BGP disruption
   - **Detection**: Iris (DNS consistency), Augur (IP ID side channel)
   - **Bypass**: Encryption (VPN, TLS) defeats keyword filtering

### Common Patterns

**Attack Patterns**:
- **Exploit trust**: IP spoofing, BGP hijacking (no authentication)
- **Amplification**: Small input → Large output (DNS, NTP, Memcached)
- **Indirection**: Reflection (hide attacker), Fast-Flux (hide infrastructure)

**Defense Patterns**:
- **Detection**: Monitoring (BGP feeds, traffic analysis)
- **Mitigation**: Filtering (Flowspec), Deaggregation (ARTEMIS), Blackholing (last resort)
- **Trade-offs**: Precision vs. collateral damage

**Measurement Patterns**:
- **Vantage points**: Use distributed infrastructure (open resolvers, route collectors)
- **Side channels**: Infer hidden state (IP ID counters)
- **Comparison**: Consistency checks (Iris DNS comparison)

---

## See Also

- [[02-Network-Layer-and-Routing]] - BGP fundamentals and routing
- [[05-Modern-Architectures]] - SDN security implications
- [[06-Application-Layer-Services]] - CDN and DNS architecture

**Next**: Review and integration of all topics
