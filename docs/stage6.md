# Stage 6: Online Softmax (Fused Attention)

Process K/V in chunks to avoid materializing the full attention matrix.

## Matrix Layout Overview

```mermaid
graph TB
    subgraph Input["Input Matrices"]
        Q["<b>Q Tile</b><br/>128 × 64<br/>(solid)"]
        K["<b>K Matrix (full context)</b><br/>4096 × 64<br/>(load in chunks)"]
        V["<b>V Matrix (full context)</b><br/>4096 × 64<br/>(load in chunks)"]
    end
    
    subgraph Process["Chunked Processing"]
        K0["K Chunk 0<br/>64 × 64"]
        K1["K Chunk 1<br/>64 × 64"]
        K2["K Chunk 2<br/>64 × 64"]
        Kdots["..."]
        K63["K Chunk 63<br/>64 × 64"]
    end
    
    subgraph Output["Output & State"]
        O["<b>Output O</b><br/>128 × 64<br/>(accumulated)"]
        State["<b>Running State</b><br/>m_i (max)<br/>l_i (sum)"]
    end
    
    Q -->|Q @ K^T| K0
    K -->|Split into<br/>64 chunks| K0
    K --> K1
    K --> K2
    K --> Kdots
    K --> K63
    
    K0 -->|scores| V
    V -->|V Chunk 0| O
    K0 -.->|update| State
    State -.->|correction| O
    
    style Q fill:#bbdefb,stroke:#1976d2,stroke-width:3px
    style K fill:#e0e0e0,stroke:#757575,stroke-width:2px,stroke-dasharray: 5 5
    style V fill:#e0e0e0,stroke:#757575,stroke-width:2px,stroke-dasharray: 5 5
    style O fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    style State fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style K0 fill:#ffcdd2,stroke:#c62828,stroke-width:2px
```

## Online Softmax Algorithm Flow

```mermaid
flowchart TD
    Start([Start: Loop over chunks i=0..63]) --> Init[Initialize:<br/>m_0 = -∞<br/>l_0 = 0<br/>O_0 = 0]
    
    Init --> LoadK[Load K chunk i<br/>64 × 64]
    LoadK --> QK[Compute Q @ K^T<br/>128 × 64 @ 64 × 64<br/>= 128 × 64 scores]
    
    QK --> Scale[Scale scores<br/>scores * 1/√d_k]
    
    Scale --> MaxChunk[Find max in chunk<br/>m_chunk = max(scores)]
    MaxChunk --> UpdateMax[Update global max<br/>m_new = max(m_old, m_chunk)]
    
    UpdateMax --> ExpScores[Compute exponentials<br/>exp(scores - m_new)]
    ExpScores --> SumChunk[Sum exponentials<br/>l_chunk = Σ exp(scores)]
    
    SumChunk --> Correction[Compute correction<br/>α = exp(m_old - m_new)]
    Correction --> UpdateSum[Update sum<br/>l_new = l_old * α + l_chunk]
    
    UpdateSum --> LoadV[Load V chunk i<br/>64 × 64]
    LoadV --> RescaleO[Rescale old output<br/>O_old * α]
    RescaleO --> AccumO[Accumulate output<br/>O_new = O_old * α + exp(scores) @ V]
    
    AccumO --> Check{More<br/>chunks?}
    Check -->|Yes| LoadK
    Check -->|No| Normalize[Final normalization<br/>O_final = O_accumulated / l_final]
    
    Normalize --> End([End: Output ready])
    
    style Start fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style End fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style UpdateMax fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style UpdateSum fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style AccumO fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Normalize fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
```

## Chunk Processing Detail

```mermaid
sequenceDiagram
    participant Q as Q Tile<br/>(128×64)
    participant K as K Chunks<br/>(64×64 each)
    participant V as V Chunks<br/>(64×64 each)
    participant State as Running State<br/>(m_i, l_i)
    participant O as Output O<br/>(128×64)
    
    Note over Q,O: Initialize: m_0=-∞, l_0=0, O_0=0
    
    loop For each chunk i = 0..63
        K->>Q: Load K chunk i
        Q->>Q: Compute Q @ K^T → scores (128×64)
        Q->>Q: Scale scores
        Q->>State: Find max → m_chunk
        State->>State: m_new = max(m_old, m_chunk)
        Q->>Q: exp(scores - m_new)
        Q->>State: Sum exp → l_chunk
        State->>State: α = exp(m_old - m_new)
        State->>State: l_new = l_old * α + l_chunk
        V->>O: Load V chunk i
        O->>O: O_old * α (rescale)
        O->>O: O += exp(scores) @ V
        State->>State: Update m_old, l_old
    end
    
    Note over O: Final: O_final = O / l_final
```

## Visual Representation of Sliding Window

```
Iteration 0:
┌─────────┐     ┌────┬────┬────┬─────────────────────────┐
│         │     │ 0  │ 1  │ 2  │ ... (64 chunks total)   │
│    Q    │  @  ├────┴────┴────┴─────────────────────────┤
│ (128×64)│     │    K Matrix (4096 × 64)                 │
│         │     │    [Chunk 0 highlighted]                │
└─────────┘     └──────────────────────────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Partial scores     │
                    │  Update m_0, l_0    │
                    │  Accumulate to O_0  │
                    └─────────────────────┘

Iteration 1:
┌─────────┐     ┌────┬────┬────┬─────────────────────────┐
│         │     │    │ 1  │ 2  │ ...                     │
│    Q    │  @  ├────┴────┴────┴─────────────────────────┤
│ (128×64)│     │    K Matrix (4096 × 64)                 │
│         │     │    [Chunk 1 highlighted]                │
└─────────┘     └──────────────────────────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Partial scores     │
                    │  Update m_1, l_1    │
                    │  Accumulate to O_1  │
                    └─────────────────────┘

... continues for 64 iterations ...
```

## Key Benefits

```mermaid
mindmap
  root((Online<br/>Softmax))
    Memory Efficiency
      No 128×4096 matrix
      Only 128×64 scores
      ~32x reduction
    Numerical Stability
      Running max update
      Prevents overflow
      Standard softmax trick
    Parallelism
      Each workgroup independent
      512 workgroups total
      16 batch×head × 32 tiles
    Hardware Optimization
      Tiles fit in registers
      DPAS instruction reuse
      Reduced memory bandwidth
```

## Implementation Details

### Sub-chunking K and V

Each 64-column chunk is further divided into 4 sub-chunks of 16 columns:

```mermaid
graph LR
    subgraph "K/V Chunk (64 cols)"
        K0[Sub 0<br/>16 cols]
        K1[Sub 1<br/>16 cols]
        K2[Sub 2<br/>16 cols]
        K3[Sub 3<br/>16 cols]
    end
    
    K0 --> DPAS[DPAS Operations<br/>128×64 @ 64×16]
    K1 --> DPAS
    K2 --> DPAS
    K3 --> DPAS
    
    DPAS --> Scores[4 partial scores<br/>each 128×16]
    Scores --> Max[Find max across all 4]
    Max --> Update[Update state & output]
    
    style DPAS fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    style Update fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
```

## State Variables

| Variable | Shape | Purpose |
|----------|-------|---------|
| `m_i` | `128×1` | Running maximum value per row |
| `l_i` | `128×1` | Running sum of exponentials per row |
| `O_i` | `128×64` | Running output accumulator |
| `Q` | `128×64` | Query tile (constant per workgroup) |
| `K_chunk` | `64×64` | Current K chunk being processed |
| `V_chunk` | `64×64` | Current V chunk being processed |

## Comparison: Standard vs Online Softmax

```mermaid
graph TB
    subgraph Standard["Standard Attention (Stage 4-5)"]
        SQ[Q: 128×64]
        SK[K: 4096×64]
        SQK["Q@K^T<br/><b>128×4096</b><br/>⚠️ Full matrix"]
        SSoft[Softmax<br/>128×4096]
        SV[V: 4096×64]
        SO[O: 128×64]
        
        SQ --> SQK
        SK --> SQK
        SQK --> SSoft
        SSoft --> SO
        SV --> SO
    end
    
    subgraph Online["Online Softmax (Stage 6)"]
        OQ[Q: 128×64]
        OK[K chunks:<br/>64×64]
        OQK["Q@K^T<br/><b>128×64</b><br/>✓ Chunk only"]
        OState[State:<br/>m_i, l_i]
        OV[V chunks:<br/>64×64]
        OO[O: 128×64<br/>accumulated]
        
        OQ --> OQK
        OK --> OQK
        OQK --> OState
        OState -.correction.-> OO
        OV --> OO
        OQK -.exp.-> OO
    end
    
    style SQK fill:#ffcdd2,stroke:#c62828,stroke-width:3px
    style OQK fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

## Mathematical Formulation

For each chunk $i$:

```math
\begin{align}
\text{scores}_i &= Q \cdot K_i^T \cdot \frac{1}{\sqrt{d_k}} \\
m_i &= \max(m_{i-1}, \max(\text{scores}_i)) \\
\alpha_i &= \exp(m_{i-1} - m_i) \\
l_i &= l_{i-1} \cdot \alpha_i + \sum \exp(\text{scores}_i - m_i) \\
O_i &= O_{i-1} \cdot \alpha_i + \exp(\text{scores}_i - m_i) \cdot V_i \\
O_{\text{final}} &= \frac{O_n}{l_n}
\end{align}
```

Where:
- $m_i$ is the running maximum
- $l_i$ is the running sum of exponentials
- $\alpha_i$ is the correction factor for previous chunks
- $O_i$ is the accumulated output
