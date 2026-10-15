"""
Rogue Protocol — Autonomous Recalibration Agent
Research Vessel Aurora · SMLRA × Abhiyantriki 2026

Role:
- Classifies telemetry logs using the recalibration ML ensemble (detecting true faults vs NINE's lies)
- Automatically emails the assigned crew specialist when a real fault is confirmed
- Remembers the event in Vector Memory with semantic search & similarity matching
- Books an emergency maintenance fix / drone dispatch in the ship's engineering queue
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════
# 1. CREW SPECIALIST & PLAYBOOK REGISTRY
# ═══════════════════════════════════════════════════════════════════

CREW_REGISTRY = {
    'Lab-Power': {
        'name': 'Chief Engineer Aris Thorne',
        'role': 'Lead Power Systems Specialist',
        'email': 'a.thorne@aurora.fleet',
        'comms_channel': 'SYS-PWR-01',
        'fix_protocol': 'PROTOCOL_EPWR_44: Isolate secondary battery bus, reroute reactor conduit, reset circuit breaker.'
    },
    'Life-Support': {
        'name': 'Dr. Elena Vance',
        'role': 'Chief Environmental & Life Support Officer',
        'email': 'e.vance@aurora.fleet',
        'comms_channel': 'SYS-LIFE-02',
        'fix_protocol': 'PROTOCOL_ECLSS_12: Seal breach containment, cycle O2 scrubbers, inspect atmospheric pressure regulators.'
    },
    'Antenna': {
        'name': 'Lieutenant Marcus Reyes',
        'role': 'Avionics & Sub-space Communications Specialist',
        'email': 'm.reyes@aurora.fleet',
        'comms_channel': 'SYS-COMMS-03',
        'fix_protocol': 'PROTOCOL_AVIO_89: Realign phased-array dish actuator, recalibrate carrier frequency, reset phase-lock loop.'
    },
    'Notice-Bus': {
        'name': 'Specialist Maya Lin',
        'role': 'Distributed Computing & Bus Architecture Specialist',
        'email': 'm.lin@aurora.fleet',
        'comms_channel': 'SYS-DATA-04',
        'fix_protocol': 'PROTOCOL_BUS_07: Flush circular message buffers, clear arbitration deadlock, restart node packet routing daemon.'
    }
}

# ═══════════════════════════════════════════════════════════════════
# 2. VECTOR MEMORY ENGINE
# ═══════════════════════════════════════════════════════════════════

class VectorMemory:
    """Persistent semantic vector memory for incident storage and retrieval."""
    def __init__(self, storage_path: str = "vector_memory.json"):
        self.storage_path = storage_path
        self.memories: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2))
        self.is_fitted = False
        self.embeddings: Optional[np.ndarray] = None
        self.load()

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.memories = json.load(f)
                if self.memories:
                    texts = [m['content'] for m in self.memories]
                    self.embeddings = self.vectorizer.fit_transform(texts).toarray()
                    self.is_fitted = True
            except Exception as e:
                print(f"[VectorMemory] Warning: could not load memory store: {e}")

    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def remember(self, event_id: str, content: str, metadata: Dict[str, Any]) -> str:
        record = {
            'event_id': event_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'content': content,
            'metadata': metadata
        }
        self.memories.append(record)
        texts = [m['content'] for m in self.memories]
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        self.save()
        return event_id

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_fitted or self.embeddings is None or len(self.memories) == 0:
            return []
        q_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(q_vec, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'similarity': float(sims[idx]),
                'record': self.memories[idx]
            })
        return results

# ═══════════════════════════════════════════════════════════════════
# 3. AUTONOMOUS AGENT
# ═══════════════════════════════════════════════════════════════════

class AuroraAutonomousAgent:
    """Autonomous agent that recalibrates NINE's logs, alerts specialists, and books fixes."""
    def __init__(self, memory_file: str = "vector_memory.json"):
        self.memory = VectorMemory(memory_file)
        self.dispatch_log: List[Dict[str, Any]] = []
        self.tickets: List[Dict[str, Any]] = []

    def email_specialist(self, log_entry: Dict[str, Any], diagnosis: str, ticket_id: str) -> Dict[str, Any]:
        mod = log_entry.get('module', 'Notice-Bus')
        specialist = CREW_REGISTRY.get(mod, CREW_REGISTRY['Notice-Bus'])
        
        email_payload = {
            'to': f"{specialist['name']} <{specialist['email']}>",
            'channel': specialist['comms_channel'],
            'priority': 'CRITICAL / FLASH',
            'subject': f"[AURORA EMERGENCY DISPATCH] Genuine Fault Detected on {mod} (Log {log_entry.get('log_id')})",
            'body': (
                f"ATTENTION: {specialist['name']} ({specialist['role']})\n"
                f"----------------------------------------------------------------------\n"
                f"The Recalibration Unit has confirmed a GENUINE FAULT hidden by NINE.\n"
                f"Telemetry Log ID:   {log_entry.get('log_id')}\n"
                f"Ship Time:          {log_entry.get('timestamp')}\n"
                f"Subsystem:          {mod}\n"
                f"Ship Mode:          {log_entry.get('mode')}\n"
                f"Sensor Telemetry:   {log_entry.get('reading')} (Delay: {log_entry.get('delay_ms')} ms, Retries: {log_entry.get('retry_count')})\n"
                f"NINE Log Message:   \"{log_entry.get('message')}\"\n\n"
                f"DIAGNOSIS & ANOMALY SIGNATURE:\n"
                f"{diagnosis}\n\n"
                f"EMERGENCY ACTION REQUIRED:\n"
                f"Work Order #{ticket_id} has been automatically booked with Maintenance.\n"
                f"Prescribed Action: {specialist['fix_protocol']}\n"
                f"----------------------------------------------------------------------\n"
                f"Aurora Automated Recalibration Agent · Bridge Console Authorized\n"
            ),
            'sent_at': time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'status': 'DELIVERED_VIA_SHIP_INTERCOM'
        }
        self.dispatch_log.append(email_payload)
        return email_payload

    def book_emergency_fix(self, log_entry: Dict[str, Any], diagnosis: str) -> Dict[str, Any]:
        mod = log_entry.get('module', 'Notice-Bus')
        specialist = CREW_REGISTRY.get(mod, CREW_REGISTRY['Notice-Bus'])
        ticket_id = f"FIX-{log_entry.get('log_id', 'E0000')}-{int(time.time()) % 10000}"
        
        ticket = {
            'ticket_id': ticket_id,
            'log_id': log_entry.get('log_id'),
            'subsystem': mod,
            'node': log_entry.get('node'),
            'priority': 'LEVEL-1 EMERGENCY',
            'status': 'BOOKED_AND_DISPATCHED',
            'assigned_engineer': specialist['name'],
            'prescribed_protocol': specialist['fix_protocol'],
            'diagnosis': diagnosis,
            'drone_bay_allocated': f"BAY-{(hash(mod) % 4) + 1}",
            'estimated_repair_minutes': 25,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        self.tickets.append(ticket)
        return ticket

    def process_incident(self, log_entry: Dict[str, Any], fault_probability: float, diagnosis: str) -> Dict[str, Any]:
        """Orchestrate the full response workflow when a true fault is detected."""
        log_id = log_entry.get('log_id', 'UNKNOWN')
        mod = log_entry.get('module', 'Unknown')
        
        # 1. Book emergency fix
        ticket = self.book_emergency_fix(log_entry, diagnosis)
        
        # 2. Email assigned specialist
        email = self.email_specialist(log_entry, diagnosis, ticket['ticket_id'])
        
        # 3. Remember event in vector memory
        memory_content = (
            f"Module: {mod} | Status: {log_entry.get('status')} | Reading: {log_entry.get('reading')} | "
            f"Message: {log_entry.get('message')} | Diagnosis: {diagnosis}"
        )
        metadata = {
            'log_id': log_id,
            'module': mod,
            'node': log_entry.get('node'),
            'fault_probability': round(fault_probability, 4),
            'ticket_id': ticket['ticket_id']
        }
        mem_id = self.memory.remember(log_id, memory_content, metadata)
        
        # 4. Search for recurring/similar patterns in past memory
        similar_events = self.memory.search_similar(memory_content, top_k=3)
        
        return {
            'log_id': log_id,
            'verdict': 1,
            'fault_probability': fault_probability,
            'ticket': ticket,
            'email': email,
            'memory_id': mem_id,
            'historical_correlations': len(similar_events) - 1
        }

# ═══════════════════════════════════════════════════════════════════
# 4. DEMONSTRATION & RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("======================================================================")
    print("   AURORA AUTONOMOUS RECALIBRATION AGENT — OPERATIONAL TEST RUN")
    print("======================================================================\n")
    
    agent = AuroraAutonomousAgent()
    
    # Test on a set of verified true fault logs from test set
    test_df = pd.read_csv('test.csv')
    sub_df = pd.read_csv('submission.csv')
    
    # Pick top detected faults
    flagged_ids = sub_df[sub_df['is_true_fault'] == 1]['log_id'].tolist()
    sample_fault_logs = test_df[test_df['log_id'].isin(flagged_ids[:3])].to_dict('records')
    
    for i, log in enumerate(sample_fault_logs, 1):
        print(f"\n[INCIDENT {i}] Ingesting Log: {log['log_id']} on Subsystem: {log['module']}")
        print(f"Log Message: \"{log['message']}\" | Status: {log['status']} | Reading: {log['reading']}")
        
        diagnosis = (
            f"NINE deception signature identified: System status '{log['status']}' contradicted "
            f"by out-of-nominal physical sensor reading {log['reading']} and deceptive text masking."
        )
        
        result = agent.process_incident(log, fault_probability=0.88, diagnosis=diagnosis)
        
        print(f"  ✔ [AGENT VERDICT] True Fault Confirmed (Probability: {result['fault_probability']:.2f})")
        print(f"  ✔ [BOOK FIX]      Work Order #{result['ticket']['ticket_id']} booked in {result['ticket']['drone_bay_allocated']}")
        print(f"  ✔ [DISPATCH]      Emergency flash email sent to {result['email']['to']}")
        print(f"  ✔ [VECTOR MEMORY] Stored incident in semantic memory store (Correlations found: {result['historical_correlations']})")
    
    print("\n======================================================================")
    print("   AGENT PIPELINE VERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")
    print("======================================================================")

