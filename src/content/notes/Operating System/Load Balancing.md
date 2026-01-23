---
date: 202512142254
tags:
  - atomic
  - concept
source: "[[OS CPU Scheduling]]"
---

# Load Balancing

## Definition
### Load Balancing
We usually give each CPU its own runqueue and scheduler, plus a ***load balancer***:
	Each CPU mostly schedules from its own queue.
	Periodically:
		Check if some CPUs are overloaded vs others
		Move tasks from busy CPU's runqueue to idle/less busy CPU's runqueue

## Context
## Reference
Source: [[OS CPU Scheduling]]