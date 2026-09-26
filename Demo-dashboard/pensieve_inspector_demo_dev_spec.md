# Pensieve Core: Interactive Demo App Specification

## 1. Project Overview

This project is a React-based interactive demo ("Pensieve Inspector") to showcase the power of the Pensieve Core memory infrastructure at a hackathon. The demo compares standard "Naive RAG" retrieval against "Pensieve Weighted Retrieval."

The core interaction involves a user entering a query (e.g., "What happened regarding my money last year?") and adjusting domain-weight sliders (e.g., Financial Impact). The app must show, side-by-side, how the naive search returns noisy, low-value results, while the weighted search instantly bubbles up high-value, relevant milestones.

## 2. Development Stages

### Stage 1: UI & Mock Data (Current Focus)

* Build the complete React frontend (UI/UX).
* Implement local state management for the search query and the weight sliders.
* Create a hardcoded mock dataset of "life events."
* Implement a *mock* search function that fakes the sorting logic locally to prove the UI works when sliders are moved.

### Stage 2: Real API Integration (Future)

* Replace the mock search function with fetch calls to the real Pensieve Core backend.
* Send the slider values in the API payload to the real MongoDB aggregation pipeline.

#### Expected Pensieve Core API Endpoints (Exact Spec TBD)

When transitioning to Stage 2, the agent should prepare to hook up the UI to the following backend REST/GraphQL endpoints:

*   **`POST /api/record` (Ingestion)**
    *   **Purpose:** Adds a new data point into the Pensieve Core database.
    *   **Payload:** Free text. This can be absolutely anything—a personal memory description, a Jira ticket, a customer complaint, or an unstructured thought. The backend handles the LLM metadata extraction, embedding, and MongoDB insertion.
*   **`POST /api/recall` (Retrieval)**
    *   **Purpose:** The primary query endpoint that executes the `$vectorSearch` and aggregation pipeline.
    *   **Payload:** Requires the user's `prompt` (the search string) AND an object containing the `weights` of the dimensions indicating their importance to this specific query (e.g., `vector_similarity`, `financial_impact`, `time_decay`).
*   **`POST /api/dimensions` (Stretch Goal)**
    *   **Purpose:** Dynamically registers a new weighting dimension to the system.
    *   **Payload:** The name and definition of the new metric so the backend can begin tracking and scoring it for future records.

## 3. UI/UX Requirements

The layout should be a single, clean dashboard with three main sections:

### A. The Control Panel (Top)

* **Search Input:** A large text field (e.g., placeholder: "What happened regarding my money last year?").
* **Domain Mode Selector:** A dropdown to select the context domain (e.g., "Personal Finance", "Career", "Health"). *For the mock data, stick to Personal Finance.*
* **Dynamic Weight Sliders (Crucial):**
  * **Vector Similarity** (0-100%, defaults to 60%)
  * **Financial Impact** (0-100%, defaults to 0%)
  * **Time Decay Penalty** (0-100%, defaults to 20%)
* *Interaction Note:* Moving the sliders should trigger an immediate re-sort of the right-hand column (Pensieve Weighted Results).

### B. The Results View (Bottom - Two Columns)

* **Left Column: [Standard Naive RAG]**
  * Displays 3-5 results based *only* on the Vector Similarity score.
  * Visual style: slightly muted, perhaps with red/orange accents on low-value items.
  * Each card must show the text, the date, and the "Similarity Score".
* **Right Column: [Pensieve Weighted Retrieval]**
  * Displays 3-5 results based on the *combined* custom formula (Similarity + Financial Impact - Time Decay).
  * Visual style: highlighted, green accents, emphasizing high-value items.
  * Each card must show the text, the date, and a breakdown of the "Final Score" (showing how the domain weight boosted it).

### C. The Code Visualizer (Optional but recommended for Hackathon)

* A small collapsible drawer or side panel that displays the simulated MongoDB Aggregation JSON pipeline updating in real-time as the sliders move.

## 4. Stage 1: Mock Data & Sorting Logic

### The Mock Dataset

Create a robust array of mock events. Ensure you have a mix of high-similarity/low-impact items and medium-similarity/high-impact items to demonstrate the concept.

```javascript
const mockDatabase = [
  {
    id: "mem_1",
    text: "Bought a $4 iced coffee at Starbucks.",
    date: "2024-11-10",
    base_similarity: 0.92, // High match for "money/spend"
    domain_weights: { financial_impact: 0.1, emotional: 0.1 },
    age_days: 2
  },
  {
    id: "mem_2",
    text: "Found a $5 bill in my winter coat pocket.",
    date: "2024-01-15",
    base_similarity: 0.88,
    domain_weights: { financial_impact: 0.1, emotional: 0.3 },
    age_days: 300
  },
  {
    id: "mem_3",
    text: "Transferred $50,000 into my Vanguard Index Fund.",
    date: "2023-08-20",
    base_similarity: 0.75, // Lower semantic match, but huge impact
    domain_weights: { financial_impact: 9.8, emotional: 5.0 },
    age_days: 450
  },
  {
    id: "mem_4",
    text: "Signed lease for new apartment at $3,200/month.",
    date: "2024-05-01",
    base_similarity: 0.65,
    domain_weights: { financial_impact: 8.5, emotional: 7.0 },
    age_days: 180
  },
  {
    id: "mem_5",
    text: "Paid $12 for a parking ticket downtown.",
    date: "2024-10-25",
    base_similarity: 0.85,
    domain_weights: { financial_impact: 0.5, emotional: 0.8 },
    age_days: 17
  }
];
```

### The Mock Sorting Logic (The Formula)

To simulate the backend in Stage 1, implement a function that calculates a `final_score` for every item in the `mockDatabase` based on the current slider values (normalized 0.0 to 1.0).

```javascript
// Pseudo-code for the sorting algorithm
function calculateScores(item, sliderValues) {
    const simWeight = sliderValues.vectorSimilarity; 
    const finWeight = sliderValues.financialImpact;
    const decayWeight = sliderValues.timeDecay;

    // Simulate standard RAG (Left Column)
    const naiveScore = item.base_similarity;

    // Simulate Pensieve Scoring (Right Column)
    // Formula: (Sim * W_sim) + (Impact * W_fin) - (AgePenalty * W_decay)
    
    // Normalize impact from 0-10 to 0-1
    const normalizedImpact = item.domain_weights.financial_impact / 10; 
    
    // Simple mock decay penalty (older = higher penalty)
    const decayPenalty = Math.min(item.age_days / 500, 1.0); 

    const finalWeightedScore = 
        (item.base_similarity * simWeight) + 
        (normalizedImpact * finWeight) - 
        (decayPenalty * decayWeight);

    return { naiveScore, finalWeightedScore };
}
```

## 5. Technology Stack Recommendations (For the Agent)

* **Framework:** React (Vite) or Next.js.
* **Styling:** Tailwind CSS (for rapid, clean UI development). Use dark mode aesthetics with bright green (#00ED64) for highlighting the Pensieve features.
* **Icons:** Lucide React or FontAwesome.
* **Components:** standard HTML inputs for sliders (`<input type="range">`).

## 6. Goal for Stage 1 Handover

Deliver a fully styled, working React application where moving the "Financial Impact" slider smoothly re-sorts the right-hand column to push the "$50,000 Vanguard" memory to the top, while the left-hand column remains stuck showing the "$4 iced coffee".

## 7. Stretch Goals

* **Showcasing Dimension Addition:** Build a UI flow (e.g., a modal) that allows the user to define and add a new custom dimension (like "Strategic Risk" or "Emotional Weight"), mapped to the mock `POST /api/dimensions` endpoint, which then dynamically generates a new slider in the Control Panel.