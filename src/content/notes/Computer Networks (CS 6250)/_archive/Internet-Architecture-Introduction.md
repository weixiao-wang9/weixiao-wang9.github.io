***Goal***: 
Connecting hosts running the same applications but located in different types of networks

The functionalities in the network architecture are implemented by diving the architectural model into layers. Each layer offers different services.

***Layered architecture advantages***: scalability, modularity, and flexibility.

The International Organization for Standardization (ISO) proposed the seven-layered OSI model shown below. And ***Internet architecture model*** on the right.
The application layer/presentation layer/session layer are combined into a single layer called ***Application layer*** in the Internet architecture model.

The interface between the application layer and the transport layer are the sockets
![](</images/Screenshot 2025-08-21 at 6.57.19 AM.png>)

Some of the ***disadvantages*** include:
1. Some layers functionality depends on the information from other layers, which can violate the goal of layer separation.
2. One layer may duplicate lower layer functionalities. For example, the functionality of error recovery can occur in lower layers, but also on upper layers as well.
3. Some additional overhead that is caused by the abstraction between layers.


#### The Application Layer
Includes multiple protocols like:
1. The HTTP protocol (web)
2. SMTP (e-mail)
3. The FTP protocol (transfers files between two end hosts)
4. DNS protocol (translate domain names to IP address)

At the application layer, we refer to the <mark style="background: #FFB8EBA6;">packet of information as a message</mark>.

Every OSI layer can be explained by three words
![](</images/Screenshot 2025-08-21 at 7.15.49 AM.png>)**Service** → What the layer provides to the layer above.
For the Application Layer, the service is: “Let apps communicate across the network.” Example: Gmail sending an email.

**Interface** → How the layer above accesses this service.
For Application Layer: your app’s APIs (e.g., browser’s use of HTTP).

**Protocol** → The rules/agreements followed at this layer between _peers_ across the network.
Example: HTTP, FTP, DNS protocols — define the _how_.

#### The Presentation Layer:
Plays the intermediate role of formatting the information(translating)
![](</images/Screenshot 2025-08-22 at 2.14.31 AM.png>)

#### The Session Layer
Is responsible for the mechanism that manages the different transport steams that belong to the same session between end-user application(human interacts directly with the program) process.![](</images/Screenshot 2025-08-22 at 2.21.37 AM.png>)

#### Transport Layer
Is responsible for the end-to-end communication between end hosts. Two important protocols ***TCP*** and ***UDP***

***TCP*** : offers include a connection-oriented service to the applications that are running on the layer above, guaranteed delivery of the application-layer messages, flow-control which in a nutshell matches the sender's and receiver's speed and a congestion-control mechanism so that the sender slows its transmission rate when it perceives the network to be congested.


***UDP***: provides a connectionless best-effort service to the applications that are running in the layer above, without reliability, flow or congestion control.

At the transport layer, we refer to the <mark style="background: #FFB8EBA6;">packet of information as a segment.</mark>
![](</images/Screenshot 2025-08-22 at 3.42.20 AM.png>)
#### Network Layer

 **Role of the Network Layer**
1. the **network layer** is responsible for **moving datagrams from the source host to the destination host across the Internet**.
2. It provides the end-to-end delivery **between machines** (hosts), while the **transport layer** provides delivery **between applications**.

***How it works:***
- The **source host** hands a _segment_ (from the transport layer, e.g., TCP or UDP) to the network layer.
- The network layer **wraps that segment in a datagram** and attaches the **destination IP address**.
- Then it sends the datagram into the Internet.
- The **destination host’s network layer** receives that datagram and passes the _segment_ up to the **transport layer**.
***Protocols***
IP: 
All internet hosts and devices that have a network layer must run the IP protocol
It defines the fields in the datagram and how the source/destination hosts and the intermediate routers use these fields

Routing Protocol:
Determine the routes that the datagrams can take between sources and destinations.


At this layer, we refer to the <mark style="background: #FFB8EBA6;">packet of information as a datagram</mark>
![](</images/Screenshot 2025-08-22 at 3.42.04 AM.png>)

#### Data Link Layer
The data link layer is responsible to move the frames from one node (host or router) to the next node.

How it works:
- Example: Sender host → Router A → Router B → Receiver host.
    1. Sender’s **network layer** creates a datagram.
    2. Sender’s **data link layer** wraps the datagram into a frame and transmits it over the first link.
    3. Router A receives the frame, passes the datagram up to its network layer, which decides the next hop.
    4. Router A’s data link layer then wraps the datagram into a new frame and sends it across the next link to Router B.
    5. This repeats until the datagram reaches the final destination host.

So the data link layer = **responsible for one hop**, while the network layer = **responsible for the full journey**.

Service: Reliable delivery, that covers the transmission of the data from one transmitting node, across one link, and finally to the receiving node.

> it is different from the TCP which offers reliability from the source host to the destination end host.

At this layer, we refer to the the<mark style="background: #FFB8EBA6;"> packet of information as frames.</mark>
![](</images/Screenshot 2025-08-22 at 3.41.45 AM.png>)
#### The Physical Layer.
to transfer bits within a frame between two nodes that are connected through a physical link.

Protocols depend on the link and on the actual transmission medium of the link![](</images/Screenshot 2025-08-22 at 3.41.32 AM.png>)


#### Layers Encapsulation

How do the layers and the protocols that run on each layer communicates with each other? ***Encapsulation and de-encapsulation***

Encapsulation:
- **Definition**: At the sender’s side, each layer of the protocol stack **wraps** the data it receives from the layer above with its own **header** (sometimes a trailer too).
- **Purpose**: Each header provides information that the _same layer on the receiving side_ needs to process the data.

**Example (sending “Hello” via a browser):**
1. **Application Layer**: Creates message = "Hello".
2. **Transport Layer**: Adds header Ht (port numbers, checksum, sequence number) → forms a **Segment**.
3. **Network Layer**: Adds header Hn (source & destination IP addresses, TTL) → forms a **Datagram (Packet)**.
4. **Link Layer**: Adds header Hl (source & destination MAC addresses, CRC) → forms a **Frame**.
5. **Physical Layer**: Sends bits across wire/radio.

**De-encapsulation**
- Happens at the **receiver’s side**.
- Each layer **strips off its header**, interprets it, and hands the remaining payload up to the layer above.

Continuing the example:

1. **Link Layer**: removes MAC header, passes packet up.
2. **Network Layer**: removes IP header, passes segment up.
3. **Transport Layer**: removes TCP/UDP header, passes message up.
4.  **Application Layer**: sees "Hello" and delivers it to the right app (browser, chat app, etc.).

***Intermediate Devices***(end to end philosophy)
**Routers** → only layers 1–3 (Physical, Link, Network). They look at **IP headers** to forward packets.
**Switches** → only layers 1–2 (Physical, Link). They look at **MAC headers** to forward frames.

They do _partial de-encapsulation_:
- A switch only opens the **Link Layer header** to see where to forward.
- A router opens the **IP header** to see where the datagram should go.

***End to End Principle***

The e2e principle suggests that specific application-level functions usually cannot, and preferably should not be built into the lower levels of the system at the core of the network.

The network core should be<mark style="background: #FFB8EBA6;"> simple and minimal</mark>, while the <mark style="background: #FFB8EBA6;">end systems should carry the intelligence
</mark>
![](</images/Screenshot 2025-08-22 at 3.51.58 AM.png>)

> **What were the designers' original goals that led to the e2e principle?** 

>Moving functions and services closer to the applications that use them, increases the flexibility and the autonomy of the application designer to offer these services to the needs of the specific application

***Violation of E2E***
1. Fire violation
2. NAT boxes
![](</images/Screenshot 2025-08-22 at 3.58.33 AM.png>)
NAT works like a translator: 
Outgoing traffic: rewrites the source IP/Port of private devices into the router's public IP/port.

Incoming traffic:  rewrites the **destination IP/port** from the public-facing IP → into the correct private IP/port, using a **NAT translation table**.

#### Hourglass Shape of Internet Architecture

![](</images/Screenshot 2025-08-22 at 4.20.53 AM.png>)
**Evolutionary Architecture** model
***Step up***
- **Layers (L)**: Just like the OSI/TCP stack, EvoArch has layers. Each layer is a stage where protocols “live.”
- **Nodes**: Each protocol (e.g., Ethernet, IP, TCP, HTTP) is represented as a **node**. The layer of node _u_ is written as l(u).
- **Edges**: Show **dependencies**:
	- If protocol u (say HTTP at layer 7) uses protocol w (say TCP at layer 4), then we draw an edge from w → u.
- So the model is a **directed acyclic graph (DAG)** that shows how protocols stack on each other.
***Substrates and Products***
- **Substrates (S(u))**: The protocols a node depends on.
    - Example: TCP’s substrate = IP.
- **Products (P(u))**: The protocols that use this node.
    - Example: TCP’s products = HTTP, SMTP, FTP, etc.
***Layer Generality (S(l))***
- Lower layers = **more general** → more protocols use them.
- Higher layers = **less general** → more specialized protocols.

s(l) = probability that a node in layer l+1 picks a substrate in l. This probability **decreases as you go up**.

***Evolutionary Value (v(u))***： The “value” of a protocol is not just its own features but also **how many valuable protocols depend on it**.( It computed recursively, if many high-value protocols use you, your value is high.)

**Competitors (C(u))**: Nodes at the same layer that share enough products (≥ fraction c).
**Competition Threshold (c)**: How much overlap is needed to count as competition.

A node is more likely to **die** if its competitors have **higher evolutionary value**.

**Birth & Death**
- **Death**:
    - If a protocol has lower value than its competitor(s), it may “die.”
    - If it dies, its products also die unless they have other substrates.
    - Example: If TCP died (hypothetically), HTTP would die too (unless it also used QUIC).
- **Birth**:
    - New protocols randomly introduced at layers (like new species).
    - Growth rate tied to size of network: bigger stacks → more innovation attempts.



***EvoArch Iteration Process***
For each round it has three phases:
1. Birth of new protocols(nodes)
- A small fraction of new nodes are added
- Each is placed randomly into a layer(Layer1 = bottom, layer10 = top)
2. Update each layer (top to bottom)
- New node at layer l selects:
        - **Substrates** (dependencies) from layer l-1 with probability s(l-1).
        - **Products** (higher dependents) from layer l+1 with probability s(l).
- Update values
	- recompute evolutionary values $v(u)$ for all noes, since new products may have appeared
	- value = recursive measure of important a node is based on its dependents.
- Competition and death
	- Within the same layer, protocols that share products are competitors
	- If a node has lower value than its competitor, it may die
	- if it dies, its products die too
1. Stopping
- Keep iterating until the stack has the desired total number of protocols

Run EvoArch for a stack with 10 layers over many rounds, you'll get ***hourglass*** shape
- Broad at bottom → narrow in middle → broad at top.![](</images/Screenshot 2025-08-22 at 9.59.06 AM.png>)
- **Inward edges (arrows pointing into a node from below)**
    → Those are the **substrates** S(u).
    → They represent the protocols that node u _depends on_.
    
- **Outward edges (arrows leaving a node upward)**
    
    → Those are the **products** P(u).
    → They represent the protocols in the layer above that _depend on_ u.

#### Implications for the Internet Architecture and future Internet architecture

Why IPV4, TCP and UDP hard to replace?
**EvoArch & the Stability of IPv4/TCP/UDP**

- **Hourglass shape:** Many protocols at bottom (link/physical) and top (apps), but narrow waist in the middle (IPv4, TCP, UDP).
- **High evolutionary value:** IPv4/TCP/UDP survive because almost all higher-layer protocols depend on them.
- **Evolutionary shield:** TCP/UDP’s stability blocks new transport protocols, which in turn protects IPv4 (since new transports would likely need a new network layer).
- **Result:** Despite alternatives (IPv6, SCTP, etc.), IPv4/TCP/UDP remain dominant due to network effects and competitive advantage.
***Ramification:***
1. Many technologies that were not originally designed for the internet have been modified so that they have versions that can communicate over the internet (such as Radio over IP).
2. hard to transition to IPV6

#### **Clean-Slate Internet Architecture Redesign**
 **Why Clean-Slate?**
- Current Internet design principles: layering, packet switching, end-to-end argument.
- Challenges: security, resilience, scalability, QoS, management, economics.
- Idea: redesign from scratch to test new assumptions, architectures, and services.
  
**Process**
- Treat clean-slate as a **design process**, not a single solution.
- Use experimental facilities with:
    - Large-scale infrastructure, multiple technologies, real users, parallel experiments.
- Outcomes:
    - New services adopted in today’s Internet.
    - Entirely new architecture.
    - Proof that current Internet is already optimal.
 **Accountable Internet Protocol (AIP):**
    - Improves **accountability** at network layer.
    - Address format: **AD:EID** (network ID + unique host ID)
    - **Source accountability:** verify packet sources, stop spoofing, shut-off message for unwanted traffic.(trace actions to a particular end host and stop that host from misbehaving)
    - **Control-plane accountability(the ability to pinpoint and prevent attacks on routing):** origin & path authentication for routing.

#### Interconnecting Hosts and Networks
1. Repeaters and hubs(L1): 
	They receive and forward digital signals to connect different Ethernet segment. They provide connectivity between hosts that are directly connected. Simple, cheap; extend signals. Downside: single collision domain.
2. Bridges and Layer2-Switches(L2): 
	These devices can enable communication between hosts that are not directly connected. Forward by MAC address; risk of buffer overflow and packet drops.

3. Routers and and Layer3 Switches(L3):  Forward by IP; use routing protocols

#### Bridges
A bridge is a device with multiple inputs/outputs. A bridge transfers frames from an input to one (or multiple) outputs.

***Learning bridges***
A learnings bridge learns, populates and maintains, a forwarding table. The bridge consults that table so that it only forwards frames on specific ports, rather than over all ports.![](</images/Screenshot 2025-08-22 at 10.34.16 AM.png>)
***Setup***
the bridge has **two collision domains** (Port 1’s LAN, Port 2’s LAN).
***Bridge Behavior***
- If the **destination host is on the same port** as the source, the bridge does **not** forward the frame.
- if the **destination host is on the other port**, the bridge forwards the frame there.
Uses a forwarding table.
***when a frame arrives:***
- Bridge looks up the **destination MAC** in the table.
- If it knows the port → forward only there.
- If source and destination are on the same port → drop (don’t waste bandwidth).
- If unknown → flood to all ports (except incoming one).

If the network topology results in loops, the bridges would loop through packets forever. Solution: Spanning Tree Algorithm
***Spanning Tree Algorithm***
- Every bridge starts by assuming **itself is the root**.
- Each bridge sends a **configuration message** <RootID, DistanceToRoot, SenderID>.
- Bridges exchange these messages in rounds and adopt the **best configuration**:
    
    1. Smaller RootID wins.
    2. If RootIDs equal → smaller distance wins.
    3. If still equal → smaller SenderID wins.
    
- The bridge with **lowest ID** in the network becomes the **root bridge**.
- Each bridge picks one **root port** (shortest path to root).
- Each LAN segment elects one **designated bridge** (closest to root). Other bridges disable their ports to break loops.