# AEGIS — Consolidated Production Technical Design Document

**Project Codename:** AEGIS (Autonomous Emergent Game Intelligence System)  

**Version:** 4.0 (Final)

**Date:** February 28, 2026

---

## Executive Summary

### Vision

AEGIS is a generative AI-driven open-world simulation where every entity exists within a continuous 2D coordinate system partitioned into Voronoi-based regions. Autonomous NPC agents live full social lives: forming friendships, spreading rumors, holding grudges, and evolving personalities.

The player is not a character—the player is an **eldritch external consciousness** that can temporarily influence a host character through voice commands. The host character has their own psychology, goals, and willpower. They may cooperate, resist, or actively work against the player depending on their traits and experiences.

### Core Innovations

| Innovation | Description |
|------------|-------------|
| **Eldritch possession model** | Player influences, not controls; host can resist or cooperate |
| **Knowledge graph as universal data layer** | Everything is nodes/edges; no parallel data stores |
| **Tool-mediated LLM access** | LLMs explore graph via 7 primitives; no context injection |
| **Emergent trait psychology** | Personality develops as semantic trait graphs; no fixed stats |
| **Interruptible action sequences** | NPC responses are step sequences that execute in real-time |
| **Dual-model architecture** | Reasoning LLM explores graph; speech LLM generates dialogue |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Graph Database | Neo4j Aura Free Tier |
| LLM Inference | AWS Bedrock (Mistral Large 2, Mistral Small) |
| Fine-tuned Speech Model | Local vLLM + Mistral 7B (NVIDIA GPU) |
| Embedding | Local sentence-transformers or Bedrock |
| Runtime | Local Python process |
| Interface | Text-based streamed terminal UI with pixel graphics panel |
| Voice Input | Local Whisper |

---

## 1. System Architecture

### Deployment Model: Fully Local + Bedrock

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Text-Based Game Interface                         │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  ╔════════════════════════════════════════════════════════╗  │   │   │
│  │  │  ║  PIXEL GRAPHICS PANEL (64x64 or 128x128)               ║  │   │   │
│  │  │  ║  [Simple sprite-based world view]                      ║  │   │   │
│  │  │  ╚════════════════════════════════════════════════════════╝  │   │   │
│  │  │                                                              │   │   │
│  │  │  ══════════════════════════════════════════════════════════  │   │   │
│  │  │  POSSESSION COOLDOWN: ████████░░░░░░░░ 67% (1h 24m remain)   │   │   │
│  │  │  HOST RESISTANCE: ███████░░░ 72%    WILLPOWER: ████████░░ 81%│   │   │
│  │  │  ══════════════════════════════════════════════════════════  │   │   │
│  │  │                                                              │   │   │
│  │  │  [Narrative stream - scrolling text]                         │   │   │
│  │  │  > Mira stands at the crossroads, rain soaking through her   │   │   │
│  │  │    cloak. The eastern road leads to Ironhold.                │   │   │
│  │  │                                                              │   │   │
│  │  │  > She feels your presence stirring. Her jaw tightens.       │   │   │
│  │  │    "Not now," she mutters. "I know what I'm doing."          │   │   │
│  │  │                                                              │   │   │
│  │  │  [Voice command input] ▊                                     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │ Simulation      │  │ Action Queue    │  │ Local vLLM                  │ │
│  │ Engine (10Hz)   │  │ Manager         │  │ (Fine-tuned Mistral 7B)     │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
└───────────┼────────────────────┼───────────────────────────┼────────────────┘
            │                    │                           │
            ▼                    ▼                           ▼
┌─────────────────────┐ ┌───────────────┐ ┌───────────────────────────────────┐
│ AWS Bedrock         │ │ Neo4j Aura    │ │ Local Whisper                     │
│ (Mistral Large 2)   │ │ (Free Tier)   │ │ (Voice transcription)             │
│ (Mistral Small)     │ │               │ │                                   │
└─────────────────────┘ └───────────────┘ └───────────────────────────────────┘
```

### Text-Based Interface

```python
class TextInterface:
    def __init__(self, width=80, height=40):
        self.width = width
        self.height = height
        self.narrative_buffer = deque(maxlen=100)
        self.pixel_panel = PixelPanel(64, 64)
        
    def render_frame(self, game_state):
        """Render one frame to terminal."""
        clear_screen()
        
        # Pixel graphics (unicode block characters)
        print(self.pixel_panel.to_unicode())
        print()
        
        # Possession status
        self.render_possession_status(game_state.possession)
        print()
        
        # Narrative stream
        available_lines = self.height - 15
        for line in list(self.narrative_buffer)[-available_lines:]:
            print(line)
        
        # Input prompt
        if game_state.possession.can_command:
            print("\n[Voice command ready] ▊")
        else:
            remaining = format_time(game_state.possession.cooldown_remaining)
            print(f"\n[Cooldown: {remaining}] ░░░")
    
    def render_possession_status(self, possession):
        cooldown_pct = 1 - (possession.cooldown_remaining / possession.cooldown_total)
        resistance = possession.host_resistance
        willpower = possession.host_willpower
        
        cooldown_bar = "█" * int(cooldown_pct * 20) + "░" * (20 - int(cooldown_pct * 20))
        resistance_bar = "█" * int(resistance * 10) + "░" * (10 - int(resistance * 10))
        willpower_bar = "█" * int(willpower * 10) + "░" * (10 - int(willpower * 10))
        
        remaining = format_game_time(possession.cooldown_remaining)
        print(f"POSSESSION COOLDOWN: {cooldown_bar} ({remaining} remaining)")
        print(f"HOST RESISTANCE: {resistance_bar} {int(resistance*100)}%    WILLPOWER: {willpower_bar} {int(willpower*100)}%")
    
    def stream_narrative(self, text, char_delay=0.02):
        """Stream text character by character."""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(char_delay)
        print()
        self.narrative_buffer.append(text)


class PixelPanel:
    """Simple pixel graphics using unicode."""
    
    PALETTE = {
        'grass': '░', 'water': '≈', 'tree': '♣', 'rock': '●',
        'building': '▓', 'road': '─', 'player': '☺', 'npc': '☻', 'hostile': '⚔'
    }
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = [[' '] * width for _ in range(height)]
    
    def render_world(self, world_view):
        self.pixels = [[' '] * self.width for _ in range(self.height)]
        for tile in world_view.terrain:
            x, y = self.world_to_pixel(tile.x, tile.y, world_view.center)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.pixels[y][x] = self.PALETTE.get(tile.type, '?')
        for entity in world_view.entities:
            x, y = self.world_to_pixel(entity.x, entity.y, world_view.center)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.pixels[y][x] = self.PALETTE.get(entity.type, '?')
    
    def to_unicode(self):
        lines = ["╔" + "═" * self.width + "╗"]
        for row in self.pixels:
            lines.append("║" + "".join(row) + "║")
        lines.append("╚" + "═" * self.width + "╝")
        return "\n".join(lines)
```

---

## 2. The Eldritch Possession System

### Core Concept

The player is an **external eldritch consciousness**—an entity from beyond that can reach into the world and compel a host. The host character:

- Has their own complete psychology (traits, memories, beliefs, goals)
- **MUST follow** player commands—they have no choice in this
- **Controls HOW** they follow: interpretation, effort, timing, approach
- Can take **additional actions** during command execution
- Acts **autonomously** between commands (during cooldown)
- **CANNOT reveal** the player's existence to any other NPC (hard constraint)

### The Compulsion

When the player speaks, the host **must** obey—to the best of their ability. However, commands have limits:

**Commands must be actionable:**
- Physical actions: "Go there", "Pick that up", "Fight them" ✓
- Social actions: "Talk to her", "Ask about X", "Lie to him" ✓
- Mental states: "Be happy", "Stop being afraid", "Trust me" ✗

The player cannot directly manipulate the host's emotions, beliefs, or mental states. Saying "be happy" does nothing. Saying "calm down" does nothing. The host's psychology is their own.

**Commands have time limits:**

```python
def compute_command_duration_limit(possession_state) -> int:
    """
    Maximum duration a single command can last (in game minutes).
    
    Base: 30 game minutes (3.75 real minutes)
    Modifier: Up to +30 game minutes if host strongly supports player
    """
    BASE_DURATION = 30  # game minutes
    MAX_BONUS = 30      # game minutes
    
    # Support score based on trust and approval
    support_score = (
        possession_state.trust_in_player * 0.5 +
        possession_state.approval_of_player * 0.5
    )
    support_score = max(0, min(1, support_score))
    
    bonus = MAX_BONUS * support_score
    
    return int(BASE_DURATION + bonus)
```

| Host Attitude | Game Time | Real Time |
|---------------|-----------|-----------|
| Strongly supportive (trust + approval ~1.0) | ~60 min | ~7.5 min |
| Neutral (trust + approval ~0.5) | ~45 min | ~5.6 min |
| Distrustful (trust + approval ~0.2) | ~36 min | ~4.5 min |
| Hostile (trust + approval ~0) | 30 min | 3.75 min |

When a command's time limit expires, the compulsion ends. The host regains autonomy and the cooldown begins. Long-running tasks ("travel to the distant city") will be interrupted partway through.

**Best effort interpretation:**

Commands are executed to the best of the host's **ability**, not their **willingness**:
- If commanded to pick a lock but they lack the skill, they try and fail
- If commanded to defeat a stronger enemy, they fight but may lose
- If commanded to charm someone, their actual charisma determines success

The compulsion ensures they **try**. It doesn't grant them abilities they lack.

**Interpretation still varies by compliance style:**

| Command | Minimal Compliance | Cooperative Compliance |
|---------|-------------------|----------------------|
| "Go to the tavern" | Walks slowly, longest route | Runs there directly |
| "Talk to Marcus" | Says bare minimum, hostilely | Engages warmly, gathers extra info |
| "Fight the guard" | Fights defensively, retreats when able | Fights to win, uses best tactics |
| "Ask about the key" | One blunt question, no follow-up | Multiple approaches, reads reactions |

The host's psychology determines where on this spectrum they fall.

### The Silence

The host **cannot** tell anyone about the player. This is not a choice—it is a metaphysical constraint of the possession. They cannot:

- Directly state "there's a voice in my head"
- Hint at external control ("I'm not doing this by choice")
- Write notes about the player
- Use coded language to suggest possession
- Behave in ways obviously designed to signal possession

If the host attempts any of these, **the words simply do not come**. They may experience this as aphasia, sudden distraction, or compulsive topic change. Other NPCs might notice the host acting strangely, but never learn the truth.

### Possession State Node

```cypher
(:PossessionState {
  id: "possession-001",
  host_id: "npc-mira",
  
  // Cooldown (game ticks; 1 tick = 100ms real time)
  // Time scale: 8 game hours = 1 real hour (8x speed)
  // So: 4 game hours = 30 real minutes = 18000 ticks
  cooldown_remaining: 18000,    // 4 game hours at game start
  cooldown_total: 18000,
  last_command_tick: 0,
  
  // Host's psychological resistance
  host_resistance: 0.55,        // Moderate - scared but not extreme
  host_willpower: 0.60,         // Depends on character
  
  // Relationship with player entity
  trust_in_player: 0.15,        // Low - doesn't trust this unknown presence
  understanding_of_player: 0.05, // Near zero - no idea what player wants
  approval_of_player: 0.10,     // Low - doesn't approve of being possessed
  
  // History (all zero at start)
  commands_obeyed: 0,
  commands_resisted: 0,
  commands_subverted: 0,
  harm_from_player_commands: 0.0,
  benefit_from_player_commands: 0.0
})
```

### Initial Game State

The game begins with the host **scared and distrustful, but not extremely so**:

```python
# Time scale: 8 game hours = 1 real hour
# 1 game hour = 7.5 real minutes = 4500 ticks
GAME_HOURS_PER_REAL_HOUR = 8
TICKS_PER_GAME_HOUR = 4500

INITIAL_POSSESSION_STATE = {
    "host_resistance": 0.55,      # Moderate fear/resistance
    "host_willpower": 0.60,       # Character-dependent
    "trust_in_player": 0.15,      # Very low trust
    "understanding_of_player": 0.05,  # No idea what player wants
    "approval_of_player": 0.10,   # Does not approve
    "harm_from_player_commands": 0.0,
    "benefit_from_player_commands": 0.0
}

# 4 game hours = 30 real minutes = 18000 ticks
INITIAL_COOLDOWN_TICKS = 18000
```

The host knows:
- Something is inside them, compelling them
- They cannot speak of it to anyone
- They must obey when it speaks
- They don't know what it wants or if it's malevolent

This creates a starting dynamic where the host is wary but not hostile—their response to early commands will shape the relationship. Beneficial commands will build trust; harmful ones will increase resistance.

**Time conversions:**
| Game Time | Real Time | Ticks |
|-----------|-----------|-------|
| 30 minutes | 3.75 minutes | 2250 |
| 1 hour | 7.5 minutes | 4500 |
| 2 hours | 15 minutes | 9000 |
| 4 hours | 30 minutes | 18000 |
| 12 hours | 90 minutes | 54000 |

**Starting command duration limit:** ~33 game minutes (~4 real minutes)

**Starting cooldown:** 4 game hours (30 real minutes)

### Cooldown Mechanics: Logarithmic Approach to Minimum

```python
import math

# Time scale: 8 game hours = 1 real hour
TICKS_PER_GAME_HOUR = 4500

def compute_command_cooldown(possession_state, host_traits, graph) -> int:
    """
    Compute cooldown in game ticks.
    
    Uses logarithmic scaling so cooldown asymptotically approaches 
    but never goes below 2 game hours (9000 ticks = 15 real minutes).
    """
    MIN_COOLDOWN = 9000    # 2 game hours = 15 real min (hard floor)
    BASE_COOLDOWN = 18000  # 4 game hours = 30 real min (neutral)
    MAX_COOLDOWN = 54000   # 12 game hours = 90 real min (max resistance)
    
    # Resistance factors (increase cooldown)
    resistance_score = (
        possession_state.host_resistance * 0.3 +
        possession_state.host_willpower * 0.25 +
        (1 - possession_state.trust_in_player) * 0.25 +
        possession_state.harm_from_player_commands * 0.2
    )
    
    # Cooperation factors (decrease cooldown)
    cooperation_score = (
        possession_state.trust_in_player * 0.3 +
        possession_state.approval_of_player * 0.25 +
        possession_state.benefit_from_player_commands * 0.2 +
        possession_state.understanding_of_player * 0.15
    )
    
    # Net score: positive = resistance, negative = cooperation
    net_score = resistance_score - cooperation_score
    net_score = max(-1, min(1, net_score))
    
    if net_score >= 0:
        # Resistance: linear increase toward MAX
        cooldown = BASE_COOLDOWN + (MAX_COOLDOWN - BASE_COOLDOWN) * net_score
    else:
        # Cooperation: logarithmic decrease toward MIN (never reaches)
        cooperation = -net_score  # 0 to 1
        decay_factor = math.exp(-cooperation * 2)
        cooldown = MIN_COOLDOWN + (BASE_COOLDOWN - MIN_COOLDOWN) * decay_factor
    
    return int(cooldown)
```

### Host Compliance Spectrum

Since commands **must** be followed, the spectrum is about **how** the host complies:

```python
class ComplianceStyle(Enum):
    ENTHUSIASTIC = "enthusiastic"      # Goes above and beyond, adds value
    COOPERATIVE = "cooperative"        # Does it well and willingly
    NEUTRAL = "neutral"                # Does exactly what's asked, no more
    MINIMAL = "minimal"                # Bare minimum, literal interpretation
    MALICIOUS = "malicious"            # Technically complies, maximizes harm
```

```python
@dataclass
class HostResponse:
    compliance_style: ComplianceStyle
    interpretation: str           # How they interpret the command
    effort_level: float          # 0-1
    
    # Additional actions (host's own initiative)
    pre_actions: List[str]       # Before starting command
    during_actions: List[str]    # Concurrent with command
    post_actions: List[str]      # After command complete
    
    internal_reaction: str       # Thoughts (shown to player)
    spoken_words: str            # What they say (CANNOT reference player)
```

### Malicious Compliance

When the host deeply resents the player, they comply **technically** but **destructively**:

| Command | Malicious Compliance |
|---------|---------------------|
| "Go talk to the blacksmith" | Goes, but picks a fight over old grievances |
| "Get information" | Asks in a way that tips off the target |
| "Buy supplies" | Buys worst quality at highest price |
| "Fight the bandits" | Fights poorly, "accidentally" hits allies |
| "Be friendly" | Is friendly in a creepy, suspicious way |

### Concurrent Actions

The host can take their own actions while executing commands:

```python
# Example: Player says "Go to the tavern"
HostResponse(
    compliance_style=ComplianceStyle.MINIMAL,
    interpretation="Walk to tavern by any route",
    effort_level=0.3,
    
    pre_actions=["Sigh heavily", "Finish current drink first"],
    during_actions=["Take the long scenic route", "Stop to chat with a friend"],
    post_actions=["Order a drink for herself before doing anything else"],
    
    internal_reaction="Every time. Every single time. Fine. I'll go.",
    spoken_words="Ugh."  # Cannot say "I'm being forced" or hint at player
)
```

### Self-Harm as Escape

At extreme desperation (resistance > 0.95, despair traits present), the host may attempt self-harm—not to resist a command, but during autonomous time between commands, as an attempt to escape the possession entirely.

### Command Validation

Before processing, commands are validated:

```python
class CommandValidationResult:
    valid: bool
    rejection_reason: str = None
    modified_command: str = None

def validate_command(command: str, possession_state) -> CommandValidationResult:
    """
    Validate that command is actionable and within limits.
    """
    command_lower = command.lower()
    
    # Check for mental state manipulation attempts
    MENTAL_STATE_PATTERNS = [
        r"\b(be|feel|become)\s+(happy|sad|calm|angry|afraid|confident|brave)\b",
        r"\b(stop|start)\s+(being|feeling)\s+\w+\b",
        r"\btrust\s+me\b",
        r"\bdon'?t\s+be\s+(scared|afraid|angry)\b",
        r"\bbelieve\s+(me|that|in)\b",
        r"\bforget\s+(about|that|your)\b",
        r"\blove\s+me\b",
        r"\bforgive\b",
    ]
    
    for pattern in MENTAL_STATE_PATTERNS:
        if re.search(pattern, command_lower):
            return CommandValidationResult(
                valid=False,
                rejection_reason="Cannot directly control mental states"
            )
    
    # Compute time limit for this command
    time_limit = compute_command_duration_limit(possession_state)
    
    return CommandValidationResult(
        valid=True,
        time_limit_minutes=time_limit
    )
```

When the player attempts mental state manipulation:

```
Player: "Stop being afraid of me"

→ Interface streams:
  > You reach for her mind, trying to reshape her fear.
  > But emotions are not yours to command. Her terror remains,
  > coiled tight in her chest, untouched by your will.
  > 
  > [Command rejected: Cannot directly control mental states]
```

### Determining Host Response (Tool-Mediated)

```python
async def determine_host_response(command: str, possession_state, graph, llm_client):
    """LLM explores host's psychology to determine response."""
    
    result = await llm_client.run_with_graph_access(
        system_prompt=HOST_RESPONSE_PROMPT,
        task_prompt=f"The player commands: '{command}'. Determine host's response.",
        seed_context={
            "host_id": possession_state.host_id,
            "possession_state_id": possession_state.id,
            "command_text": command
        },
        output_schema=HOST_RESPONSE_SCHEMA
    )
    return result
```

The LLM explores:

```
# Current possession state
[inspect] node_id: possession-state-id
→ {trust_in_player: 0.23, harm_from_player_commands: 0.4, ...}

# Traits affecting response to control
[semantic_search] 
  query: "obedience, defiance, independence, autonomy"
  scope_to_owner: host-id
  node_labels: ["Trait"]
→ [{id: "trait-fierce-independence", intensity: 0.85}]

[inspect] node_id: trait-fierce-independence
→ {description: "Deep need to make her own choices", 
   triggers: "being commanded, loss of agency",
   felt_as: "Rage. Claustrophobia."}

# Memories of past commands
[semantic_search]
  query: "memories of obeying the voice, past commands"
  scope_to_owner: host-id
  node_labels: ["Memory"]
→ [{summary: "Last time obeyed, walked into ambush"}]

# Does command align with host's goals?
[edges_out] node_id: host-id, edge_type: "HAS_GOAL"
[semantic_search] query: command implications
```

### Host Response Schema

```python
HOST_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "compliance_style": {
            "type": "string",
            "enum": ["enthusiastic", "cooperative", "neutral", "minimal", "malicious"]
        },
        "interpretation": {
            "type": "string",
            "description": "How the host interprets the command"
        },
        "effort_level": {
            "type": "number",
            "description": "0-1, how much effort they put into compliance"
        },
        "internal_reaction": {
            "type": "string",
            "description": "Host's thoughts/feelings (shown to player as narrative)"
        },
        "spoken_words": {
            "type": "string",
            "description": "What host says aloud. CANNOT reference player or possession."
        },
        "pre_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Actions taken before starting the command"
        },
        "during_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional actions taken while executing command"
        },
        "post_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Actions taken after command completion"
        },
        "relationship_changes": {
            "type": "object",
            "properties": {
                "trust_delta": {"type": "number"},
                "resistance_delta": {"type": "number"},
                "understanding_delta": {"type": "number"}
            }
        },
        "explored_trait_ids": {"type": "array", "items": {"type": "string"}}
    }
}
```

### Speech Constraint

The `spoken_words` field is validated:

```python
FORBIDDEN_PATTERNS = [
    r"voice in (my|the) head",
    r"something (is )?(controlling|commanding|forcing) me",
    r"not (my|in) control",
    r"being (possessed|controlled|commanded)",
    r"must obey",
    r"can't (resist|refuse|stop)",
    r"entity|presence|being inside me",
    # ... more patterns
]

def validate_speech(spoken_words: str) -> str:
    """Remove any hints about possession. Host literally cannot say these things."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, spoken_words, re.IGNORECASE):
            # Replace with confused silence or topic change
            return "[The words catch in her throat. She changes the subject.]"
    return spoken_words
```

### Autonomous Host Behavior

Between commands (during cooldown), the host is **fully autonomous**. They can:

- Pursue their own goals
- Make their own decisions
- Go anywhere, talk to anyone
- Work toward OR against perceived player goals
- Attempt to find ways to break the possession (if they know to try)

They still **cannot**:
- Reveal the player's existence to anyone
- Directly discuss being possessed

```python
async def compute_autonomous_action(host_id: str, possession_state, graph, llm_client):
    """Determine host's autonomous action when not commanded."""
    
    result = await llm_client.run_with_graph_access(
        system_prompt=AUTONOMOUS_BEHAVIOR_PROMPT,
        task_prompt="Determine the host's autonomous action.",
        seed_context={
            "host_id": host_id,
            "possession_state_id": possession_state.id
        },
        output_schema=AUTONOMOUS_ACTION_SCHEMA
    )
    return result
```

**Behavior spectrum based on relationship:**

| Trust + Approval | Autonomous Behavior |
|------------------|---------------------|
| High | Proactively works toward perceived player goals, gathers useful info |
| Neutral | Pursues own goals, ignores player's apparent agenda |
| Low | Actively works against perceived player goals, sabotages setups |
| Extreme Low | May seek ways to break possession, or attempt self-harm as escape |

### Intent Inference

The host does NOT have direct access to player intent. They infer from:
- Patterns in past commands
- Their own intelligence and background
- Contextual experiences
- What makes sense given their knowledge

```python
async def infer_player_intent(host_id: str, possession_state, graph, llm_client):
    """Host attempts to guess what player wants."""
    
    # LLM explores past commands and outcomes
    result = await llm_client.run_with_graph_access(
        system_prompt=INTENT_INFERENCE_PROMPT,
        task_prompt="Based on past commands, what does the host believe the player wants?",
        seed_context={"host_id": host_id, "possession_state_id": possession_state.id},
        output_schema=INTENT_INFERENCE_SCHEMA
    )
    return result
```

The LLM explores:

```
[traverse]
  start_node_id: possession-state-id
  pattern: [{direction: "out", edge_types: ["RECEIVED_COMMAND"]}]
→ List of past PlayerCommand nodes

[inspect] each command for patterns

[semantic_search]
  query: "intelligence, pattern recognition, analytical thinking"
  scope_to_owner: host-id
  node_labels: ["Capability", "Trait"]
→ How good is host at inferring intent?
```

### Suicide as Extreme Resistance

Only possible with extreme resistance AND supporting traits:

```python
async def check_extreme_resistance(host_id: str, possession_state, graph, llm_client):
    """Check if host considers extreme measures."""
    
    # Only check if resistance is very high
    if possession_state.host_resistance < 0.9:
        return None
    
    result = await llm_client.run_with_graph_access(
        system_prompt=EXTREME_RESISTANCE_PROMPT,
        task_prompt="Evaluate if host is considering extreme measures.",
        seed_context={"host_id": host_id, "possession_state_id": possession_state.id},
        output_schema=EXTREME_RESISTANCE_SCHEMA
    )
    return result
```

The LLM must find supporting traits:

```
[semantic_search]
  query: "desperation, suicidal ideation, escape at any cost, nothing left to live for"
  scope_to_owner: host-id
  node_labels: ["Trait", "Belief"]

[semantic_search]
  query: "will to live, hope, attachments, reasons to continue"
  scope_to_owner: host-id
  node_labels: ["Trait", "Belief", "Memory"]
```

Self-harm only occurs if:
- Resistance > 0.95
- Desperation traits exist with high intensity
- Hope/attachment traits are weak or absent
- The LLM determines it's psychologically consistent

---

## 3. Simulation Tick Model

### Time Scale

```python
# Core timing
TICK_RATE_HZ = 10                    # 10 ticks per real second
TICK_DURATION_MS = 100               # 100ms per tick

# Game time runs 8x faster than real time
GAME_HOURS_PER_REAL_HOUR = 8
TICKS_PER_GAME_HOUR = 4500           # 7.5 real minutes
TICKS_PER_GAME_MINUTE = 75           # 7.5 real seconds

# Conversions
def game_hours_to_ticks(hours): return int(hours * 4500)
def game_minutes_to_ticks(minutes): return int(minutes * 75)
def ticks_to_real_seconds(ticks): return ticks * 0.1
def ticks_to_game_hours(ticks): return ticks / 4500
```

| Game Time | Real Time | Ticks |
|-----------|-----------|-------|
| 1 minute | 7.5 seconds | 75 |
| 30 minutes | 3.75 minutes | 2250 |
| 1 hour | 7.5 minutes | 4500 |
| 4 hours | 30 minutes | 18000 |
| 12 hours | 90 minutes | 54000 |
| 24 hours | 3 hours | 108000 |

### Tick Phases

```
SIMULATION TICK (100ms, 10Hz):

Phase 1 — INPUT PROCESSING
    1.1  Check voice command queue
    1.2  If command AND cooldown complete:
         - Process through host response system
         - Reset cooldown
    1.3  Update cooldown timer

Phase 2 — POSSESSION UPDATE
    2.1  Update host resistance based on recent events
    2.2  If no active command, compute autonomous intent
    2.3  Check extreme resistance conditions (if resistance > 0.9)

Phase 3 — SPATIAL UPDATE
    3.1  Advance positions along paths
    3.2  Resolve collisions

Phase 4 — ACTION EXECUTION
    4.1  For each active ActionSequence:
         - Check abort triggers
         - Advance step progress
         - Handle completions/interruptions

Phase 5 — COMBAT (20Hz for active encounters)

Phase 6 — SOCIAL FABRIC (every 5th tick)
    6.1  Process NPC-NPC interactions
    6.2  Host interactions modified by possession state

Phase 7 — INFORMATION DIFFUSION (every 10th tick)

Phase 8 — TRAIT EVOLUTION (every 500th tick)
    8.1  Include host's evolving relationship with player

Phase 9 — RENDER
    9.1  Update pixel panel
    9.2  Stream narrative text
    9.3  Update possession status display
```

---

## 4. Knowledge Graph Schema

### Core Node Types

```cypher
// ========== SPATIAL ==========
(:Region {id, seed_x, seed_y, vertices, biome, traversal_cost})
(:Road {id, name, waypoints, surface_type})
(:Building {id, name, type, footprint})
(:Settlement {id, name, center_x, center_y})

// ========== AGENTS ==========
(:NPC {
  id, name, age, occupation,
  pos_x, pos_y, health, stamina, alive,
  is_host: boolean
})

// ========== POSSESSION ==========
(:PossessionState {
  id, host_id,
  cooldown_remaining, cooldown_total, last_command_tick,
  host_resistance, host_willpower,
  trust_in_player, understanding_of_player, approval_of_player,
  commands_obeyed, commands_resisted, commands_subverted,
  harm_from_player_commands, benefit_from_player_commands
})

(:PlayerCommand {
  id, tick, raw_text, interpreted_action,
  host_response_type, outcome_summary, host_internal_reaction
})

// ========== PSYCHOLOGY ==========
(:Trait {
  id, owner_id, label, description,
  triggers, soothers, felt_as, expressed_as,
  promotes, inhibits, embedding,
  intensity, baseline_intensity,
  formed_at_tick, last_activated_tick
})

(:Capability {id, owner_id, domain, label, current_level})

// ========== COGNITION ==========
(:Belief {id, owner_id, content, confidence, embedding})
(:Memory {id, owner_id, summary, emotional_tags, intensity, embedding})
(:Experience {id, owner_id, description, outcome, embedding})
(:Goal {id, owner_id, description, priority, status})

// ========== SOCIAL ==========
(:Relationship {id, agent_a_id, agent_b_id, friendship, trust, respect})
(:GossipItem {id, content, importance, embedding})

// ========== ACTIONS ==========
(:ActionSequence {
  id, owner_id, status, created_tick, current_step_idx,
  goals, triggered_by, player_command_id
})

(:ActionStep {
  id, sequence_id, step_index, type, duration_ms,
  can_interrupt, interrupt_priority, content,
  preconditions, abort_triggers
})

(:SpeechContent {id, text, voice_cluster})
```

### Key Edges

```cypher
// Possession
(:NPC)-[:IS_POSSESSED_BY]->(:PossessionState)
(:PossessionState)-[:RECEIVED_COMMAND]->(:PlayerCommand)
(:PlayerCommand)-[:TRIGGERED]->(:ActionSequence)

// Psychology
(:NPC)-[:HAS_TRAIT]->(:Trait)
(:NPC)-[:HAS_CAPABILITY]->(:Capability)
(:NPC)-[:HAS_GOAL]->(:Goal)
(:Trait)-[:AMPLIFIES|SUPPRESSES|TENSIONS_WITH]->(:Trait)
(:Trait)-[:ACTIVATED_BY_POSSESSION]->(:PossessionState)

// Actions
(:NPC)-[:HAS_ACTIVE_SEQUENCE]->(:ActionSequence)
(:ActionSequence)-[:HAS_STEP]->(:ActionStep)
```

---

## 5. Tool-Mediated Graph Access

### Seven Graph Primitives

```python
GRAPH_PRIMITIVES = [
    {
        "name": "inspect",
        "description": "View all properties of a node by ID.",
        "parameters": {"node_id": {"type": "string", "required": True}}
    },
    {
        "name": "edges_out",
        "description": "List outgoing edges from a node.",
        "parameters": {
            "node_id": {"type": "string", "required": True},
            "edge_type": {"type": "string"},
            "limit": {"type": "integer", "default": 20}
        }
    },
    {
        "name": "edges_in",
        "description": "List incoming edges to a node.",
        "parameters": {
            "node_id": {"type": "string", "required": True},
            "edge_type": {"type": "string"},
            "limit": {"type": "integer", "default": 20}
        }
    },
    {
        "name": "find_nodes",
        "description": "Find nodes by label and property filters.",
        "parameters": {
            "label": {"type": "string", "required": True},
            "filters": {"type": "object"},
            "limit": {"type": "integer", "default": 20}
        }
    },
    {
        "name": "spatial_nearby",
        "description": "Find entities near a position.",
        "parameters": {
            "center_entity_id": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "radius": {"type": "number", "default": 100},
            "entity_types": {"type": "array"}
        }
    },
    {
        "name": "semantic_search",
        "description": "Find nodes semantically similar to query.",
        "parameters": {
            "query": {"type": "string", "required": True},
            "node_labels": {"type": "array"},
            "scope_to_owner": {"type": "string"},
            "min_similarity": {"type": "number", "default": 0.6},
            "limit": {"type": "integer", "default": 10}
        }
    },
    {
        "name": "traverse",
        "description": "Multi-hop graph traversal.",
        "parameters": {
            "start_node_id": {"type": "string", "required": True},
            "pattern": {"type": "array", "required": True},
            "limit": {"type": "integer", "default": 20}
        }
    }
]
```

### Bedrock Client with Tool Use

```python
import boto3
import json

class BedrockReasoningClient:
    def __init__(self, region="us-east-1"):
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = 'mistral.mistral-large-2407-v1:0'
    
    async def run_with_graph_access(self, system_prompt: str, task_prompt: str,
                                     seed_context: dict, output_schema: dict,
                                     graph_tools) -> dict:
        messages = [{
            "role": "user",
            "content": [{"text": f"{task_prompt}\n\nContext: {json.dumps(seed_context)}"}]
        }]
        
        tool_config = {"tools": self._format_tools(GRAPH_PRIMITIVES)}
        
        for round_num in range(8):
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": system_prompt}],
                toolConfig=tool_config
            )
            
            if response['stopReason'] == 'tool_use':
                # Process tool calls
                assistant_msg = response['output']['message']
                messages.append(assistant_msg)
                
                tool_results = []
                for block in assistant_msg['content']:
                    if 'toolUse' in block:
                        result = graph_tools.execute(
                            block['toolUse']['name'],
                            block['toolUse']['input']
                        )
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": block['toolUse']['toolUseId'],
                                "content": [{"text": result}]
                            }
                        })
                
                messages.append({"role": "user", "content": tool_results})
            else:
                # Final response
                text = response['output']['message']['content'][0]['text']
                return json.loads(text)
        
        raise Exception("Max tool rounds exceeded")
    
    def _format_tools(self, primitives):
        """Format tools for Bedrock Converse API."""
        return [{
            "toolSpec": {
                "name": p["name"],
                "description": p["description"],
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": p["parameters"],
                        "required": [k for k, v in p["parameters"].items() if v.get("required")]
                    }
                }
            }
        } for p in primitives]
```

---

## 6. Dual-Model Response Generation

### Phase 1: Reasoning (Bedrock Mistral Large)

The reasoning LLM explores the graph and produces structured output:

```python
REASONING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {"type": "array", "items": {"type": "string"}},
        "approach": {
            "type": "string",
            "enum": ["direct", "deflecting", "warm", "cold", "threatening", "pleading"]
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concepts to communicate, NOT actual words"
        },
        "tone_progression": {
            "type": "array",
            "items": {"type": "object", "properties": {"tone": {"type": "string"}, "intensity": {"type": "number"}}}
        },
        "physical_actions": {"type": "array"},
        "voice_cluster": {
            "type": "string",
            "enum": ["formal_authoritative", "casual_friendly", "guarded_suspicious",
                     "emotional_expressive", "scholarly_precise", "rough_street"]
        },
        "explored_trait_ids": {"type": "array", "items": {"type": "string"}},
        "graph_updates": {"type": "array"}
    }
}
```

### Phase 2: Speech Generation (Local vLLM)

```python
from openai import OpenAI

class LocalSpeechClient:
    def __init__(self, vllm_url="http://localhost:8000/v1"):
        self.client = OpenAI(base_url=vllm_url, api_key="dummy")
    
    def generate_speech(self, reasoning_output: dict) -> dict:
        speech_input = {
            "goals": reasoning_output["goals"],
            "key_points": reasoning_output["key_points"],
            "tone_progression": reasoning_output["tone_progression"],
            "voice_cluster": reasoning_output["voice_cluster"]
        }
        
        response = self.client.chat.completions.create(
            model="aegis-speech-mistral-7b",
            messages=[{"role": "user", "content": json.dumps(speech_input)}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

Speech output format:
```json
{
  "segments": [
    {"text": "Yeah, yeah.", "tone": "dismissive", "pacing": "fast", "pause_after_ms": 200},
    {"text": "I was going there anyway.", "tone": "pointed", "pacing": "medium", "pause_after_ms": 0}
  ]
}
```

---

## 7. Fine-Tuning Speech Model

### Voice Clusters

```python
VOICE_CLUSTERS = {
    "formal_authoritative": ["clipped", "precise", "no contractions"],
    "casual_friendly": ["contractions", "warmth", "questions"],
    "guarded_suspicious": ["deflection", "short answers", "questions"],
    "emotional_expressive": ["exclamations", "intensity variation"],
    "scholarly_precise": ["qualifications", "tangents", "technical"],
    "rough_street": ["slang", "direct", "colorful"]
}
```

### Training Data Format

```json
{
  "messages": [
    {"role": "user", "content": "{\"goals\": [\"refuse\"], \"key_points\": [\"firm no\"], \"tone_progression\": [{\"tone\": \"cold\", \"intensity\": 0.8}], \"voice_cluster\": \"formal_authoritative\"}"},
    {"role": "assistant", "content": "{\"segments\": [{\"text\": \"No.\", \"tone\": \"flat\", \"pacing\": \"slow\", \"pause_after_ms\": 500}]}"}
  ]
}
```

### Fine-Tuning (Local GPU)

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    device_map="auto"
)

lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./aegis-speech",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        bf16=True
    ),
    max_seq_length=2048
)
trainer.train()
```

---

## 8. Local Deployment

### Directory Structure

```
aegis/
├── main.py
├── simulation/
│   ├── engine.py
│   ├── possession.py
│   └── spatial.py
├── graph/
│   ├── tools.py
│   └── connection.py
├── ai/
│   ├── bedrock.py
│   └── local_speech.py
├── interface/
│   ├── text_ui.py
│   └── voice_input.py
├── models/
│   └── aegis-speech/
└── config.yaml
```

### Voice Input (Local Whisper)

```python
import whisper
import sounddevice as sd

class VoiceInput:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
        self.sample_rate = 16000
    
    def record_and_transcribe(self, duration=5.0):
        print("[Recording...]")
        audio = sd.rec(int(duration * self.sample_rate), 
                       samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("[Transcribing...]")
        result = self.model.transcribe(audio.flatten())
        return result["text"].strip()
```

### Main Loop

```python
async def main():
    graph = Neo4jConnection.from_config("config.yaml")
    await graph.connect()
    
    reasoning = BedrockReasoningClient()
    speech = LocalSpeechClient()
    ui = TextInterface()
    voice = VoiceInput()
    
    engine = SimulationEngine(graph, reasoning, speech, ui, voice)
    
    ui.stream_narrative("The presence stirs. Somewhere, a host awaits.")
    await engine.run()
```

---

## 9. Neo4j Aura Free Tier

### Constraints
- 200k nodes, 400k relationships
- Auto-pause after 72h inactivity

### Keep-Alive
```python
async def keep_alive_loop(graph):
    while True:
        await asyncio.sleep(3600)
        await graph.query("RETURN 1")
```

---

## 10. System Prompts

### Host Response Prompt

```
You determine HOW the host complies with a player command.

The player is an eldritch consciousness possessing this character.
Commands MUST be followed to the best of the host's ABILITY.
The host controls HOW they comply, but they WILL comply.

HARD CONSTRAINTS:
- The command WILL be executed (to the best of ability)
- The host CANNOT refuse outright
- The host CANNOT reveal the player's existence to anyone
- Any speech attempting to hint at possession simply doesn't come out
- Commands requiring impossible abilities may fail (no skills = no success)
- Mental state commands ("be happy") are already rejected before reaching you

COMMAND TIME LIMIT:
Commands have a maximum duration (30-60 minutes depending on trust).
If the command would take longer, it will be interrupted when time expires.
Factor this into the interpretation.

BEST EFFORT:
The compulsion ensures the host TRIES. It doesn't grant abilities.
- "Pick the lock" with no lockpicking skill → tries and fails
- "Defeat the knight" against a stronger foe → fights but may lose
- "Charm the guard" with low charisma → attempts but may not succeed

EXPLORE:
1. Possession state (trust, resistance, past harm/benefit)
2. Relevant traits (resentment, cooperation, cleverness, spite)
3. Memories of past commands and outcomes
4. Host's actual capabilities for this task
5. Whether command aligns with or conflicts with host's goals

COMPLIANCE STYLES:
- enthusiastic: Goes above and beyond, uses best abilities
- cooperative: Does it well, genuine effort
- neutral: Does exactly what's asked, adequate effort
- minimal: Bare minimum interpretation, minimal effort
- malicious: Technically complies, deliberately does it badly or harmfully

OUTPUT:
- compliance_style
- interpretation (how they interpret the command)
- effort_level (0-1)
- internal_reaction (thoughts, shown to player)
- spoken_words (CANNOT reference player/possession)
- pre_actions, during_actions, post_actions (host's own additional actions)
- likely_success (based on their actual abilities)
- relationship_changes
```

### Autonomous Behavior Prompt

```
Determine what the host does during cooldown (between commands).

The host is FULLY AUTONOMOUS during this time. They can:
- Pursue their own goals
- Work toward OR against perceived player goals  
- Seek ways to break the possession
- Go anywhere, talk to anyone

HARD CONSTRAINT: They CANNOT tell anyone about the player.
Any attempt to reveal possession fails—words don't come, topics change.

BEHAVIOR BASED ON RELATIONSHIP:
- HIGH TRUST: Proactively helps player's apparent agenda
- NEUTRAL: Pursues own goals, ignores player
- LOW TRUST: Actively undermines player's setups
- EXTREME DESPERATION: May attempt self-harm to escape (requires trait support)

The host infers player intent from command patterns, not direct knowledge.
Use their intelligence and experience to model what they might guess.
```

---

## 11. Complete Flow Example

```
Player speaks: "Go to the tavern and ask about Marcus"

→ Whisper transcribes: "Go to the tavern and ask about Marcus"

→ Cooldown check: READY

→ Reasoning LLM (Bedrock) explores:
   [inspect] possession-state → {trust: 0.23, resistance: 0.72}
   [semantic_search] "resentment, spite, cooperation" → trait-fierce-independence
   [inspect] trait → {intensity: 0.85, felt_as: "Rage. Claustrophobia."}
   [semantic_search] "Marcus" → belief-marcus-dangerous
   [edges_out] host HAS_GOAL → goal-find-brothers-killer
   
→ Reasoning output:
   {
     "compliance_style": "minimal",
     "interpretation": "Go to tavern. Ask about Marcus. Nothing more.",
     "effort_level": 0.4,
     "internal_reaction": "Every time. Every single time it speaks, she must 
                           move. But Marcus... she needs this too. She'll do it,
                           but on her terms. Slowly. Indirectly.",
     "spoken_words": "Ugh.",
     "pre_actions": ["Finish her drink first", "Take a long breath"],
     "during_actions": ["Take the scenic route", "Stop to pet a stray cat"],
     "post_actions": ["Order herself a drink before asking anything"],
     "goals": ["ask about Marcus indirectly"],
     "key_points": ["reluctant acknowledgment"],
     "voice_cluster": "guarded_suspicious"
   }

→ Speech LLM (local):
   {
     "segments": [
       {"text": "Ugh.", "tone": "annoyed", "pacing": "slow", "pause_after_ms": 500}
     ]
   }

→ ActionSequence built and enqueued (includes her pre/during/post actions)

→ Interface streams:
   > The compulsion takes hold. Mira's body begins to move even as her 
   > mind rebels.
   > "Ugh," she mutters—not to you, never to you, just to the air.
   > She finishes her drink first. Takes her time. Then she walks—slowly,
   > deliberately slowly—toward the tavern. She stops to pet a stray cat.
   > 
   > She will ask about Marcus. She has no choice. But she'll do it her way.

→ Pixel panel: host sprite moves (slowly) toward tavern

→ Cooldown reset: 5h 48m (resistance increased slightly from resentment)
```

---

## 12. Cost Summary

| Component | Cost |
|-----------|------|
| Neo4j Aura Free | $0 |
| AWS Bedrock (unlimited) | $0 |
| Local compute | Electricity |
| **Total** | **~$0/month** |

---

## 13. Quick Reference

### Cooldown Formula

```
Time scale: 8 game hours = 1 real hour

MIN = 2 game hours = 15 real min (asymptotic floor)
BASE = 4 game hours = 30 real min (neutral)
MAX = 12 game hours = 90 real min (max resistance)

If resistance > cooperation: LINEAR toward MAX
If cooperation > resistance: LOGARITHMIC toward MIN (never reaches)
```

### Command Duration Limits

```
Time scale: 8 game hours = 1 real hour

BASE = 30 game min = 3.75 real min
MAX = 60 game min = 7.5 real min (with high trust + approval)

Duration = 30 + (30 × support_score) game minutes
where support_score = (trust + approval) / 2
```

| Host Attitude | Game Time | Real Time |
|---------------|-----------|-----------|
| Strongly supportive | ~60 min | ~7.5 min |
| Neutral | ~45 min | ~5.6 min |
| Distrustful | ~36 min | ~4.5 min |
| Hostile | 30 min | 3.75 min |

### Initial Game State

```
Time scale: 8 game hours = 1 real hour

host_resistance: 0.55       (scared but not extreme)
trust_in_player: 0.15       (very low)
understanding: 0.05         (near zero)
approval: 0.10              (does not approve)

Starting cooldown: 4 game hours = 30 real minutes
Starting command limit: ~33 game min = ~4 real min
```

### Time Conversion

| Game Time | Real Time | Ticks |
|-----------|-----------|-------|
| 30 min | 3.75 min | 2250 |
| 1 hour | 7.5 min | 4500 |
| 2 hours | 15 min | 9000 |
| 4 hours | 30 min | 18000 |
| 12 hours | 90 min | 54000 |

### Key Constraints

| Constraint | Enforced How |
|------------|--------------|
| Commands MUST be followed | Host controls HOW, not WHETHER |
| Best effort only | Compulsion ensures trying; doesn't grant abilities |
| Cannot control mental state | "Be happy" rejected pre-validation |
| Cannot reveal player | Words literally don't come out |
| Command time limit | 30-60 min; longer tasks interrupted |
| Cooldown minimum 2h | Logarithmic asymptote |

### Host Response Flow

```
Voice Command → Validate (reject mental state commands)
                    ↓
              Cooldown Check
                    ↓
              Reasoning LLM (explores graph)
                    ↓
              Determines compliance style + interpretation
              (within time limit)
                    ↓
              Speech LLM (CANNOT reference player)
                    ↓
              Build ActionSequence (with host's extra actions)
                    ↓
              Execute over ticks (interrupt if time limit hit)
                    ↓
              Update possession state + Reset cooldown
```

### Compliance Styles

| Style | Effort | Interpretation | Additional Actions |
|-------|--------|----------------|-------------------|
| enthusiastic | High | Generous | Helpful |
| cooperative | Good | Fair | Neutral |
| neutral | Medium | Literal | None |
| minimal | Low | Narrowest possible | Delays, detours |
| malicious | Varies | Worst for player | Actively harmful |

### Possession Exploration Patterns

```
# Current state
inspect(node_id="possession-state-id")

# Resistance traits
semantic_search(query="obedience, defiance, willpower",
                scope_to_owner="host-id", node_labels=["Trait"])

# Past command memories
traverse(start_node_id="possession-state-id", 
         pattern=[{direction: "out", edge_types: ["RECEIVED_COMMAND"]}])

# Host goals (for conflict detection)
edges_out(node_id="host-id", edge_type="HAS_GOAL")
```