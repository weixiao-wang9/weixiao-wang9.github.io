---
id: 202512170315
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 4: Interconnection & BGP"
title: Autonomous Systems and Internet Interconnection
status: Finished
tags:
  - input
  - networks
  - bgp
created: 2025-12-17
---

# Autonomous Systems and Internet Interconnection

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture scales up from routing within a single domain to routing across the entire Internet. It introduces the "Network of Networks" hierarchy (ISPs, CDNs, IXPs), the concept of **Autonomous Systems (AS)**, and the **Border Gateway Protocol (BGP)** used to glue them together. It focuses heavily on "Policy Routing" (business decisions over speed) and the modern role of IXPs.

## 📝 Notes & Highlights

### The Internet Ecosystem
- **Hierarchy:**
    - **Tier-1 ISPs:** Global backbone (e.g., AT&T, NTT). Connect to each other via peering.
    - **Tier-2/Regional ISPs:** Connect to Tier-1.
    - **Tier-3/Access ISPs:** Connect end-users.
- **Infrastructure:**
    - **IXPs (Internet Exchange Points):** Physical locations where networks interconnect locally.
    - **CDNs (Content Delivery Networks):** Distributed servers (e.g., Google, Netflix) to push content closer to users.

### Autonomous Systems (AS) & BGP
- An **AS** is a group of routers under the same administrative authority.
- **BGP (Border Gateway Protocol):** The standard for exchanging routing info *between* ASes.
- **IGPs (OSPF, RIP):** Used *within* an AS.

### Business Relationships & Policies
- **Customer-Provider (Transit):** Customer pays provider; provider forwards all traffic.
- **Peering:** Settlement-free exchange of traffic between two networks (usually restricted to their own customers).
- **Route Selection:**
    - **Import Preference:** Customer > Peer > Provider (Maximize revenue, minimize cost).
    - **Export Rules:** Do not export Peer/Provider routes to other Peers/Providers (no free transit).

### BGP Mechanics
- **eBGP:** Between different ASes.
- **iBGP:** Between routers in the same AS (disseminates external routes).
- **Attributes:**
    - **LocalPref:** Controls *outbound* traffic (preferred exit).
    - **MED:** Controls *inbound* traffic (preferred entrance).

## 🔗 Extracted Concepts
- [[Concept - Internet Ecosystem Hierarchy]]
- [[Concept - Autonomous Systems]]
- [[Concept - BGP Protocol Basics]]
- [[Concept - BGP Routing Policies]]
- [[Concept - Internet Exchange Points]]
- [[Concept - Remote Peering]]