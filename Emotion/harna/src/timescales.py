"""Multi-timescale processing architecture."""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TimescaleEvent:
    """Event at specific timescale."""
    time: float
    timescale: str  # 'millisecond', 'second', 'minute', 'hour'
    event_type: str
    intensity: float


class MultiTimescaleProcessor:
    """Processes emotional dynamics across multiple timescales."""
    
    def __init__(self):
        self.millisecond_events = []  # Orienting responses
        self.second_events = []  # Appraisal processes
        self.minute_events = []  # Mood regulation
        self.hour_events = []  # Affective styles
    
    def process_millisecond(self, stimulus_onset: float, threat_level: float):
        """Process millisecond-scale orienting response."""
        if threat_level > 0.5:
            event = TimescaleEvent(
                time=stimulus_onset,
                timescale='millisecond',
                event_type='orienting',
                intensity=threat_level
            )
            self.millisecond_events.append(event)
    
    def process_second(self, time: float, appraisal_result: Dict):
        """Process second-scale appraisal."""
        event = TimescaleEvent(
            time=time,
            timescale='second',
            event_type='appraisal',
            intensity=abs(appraisal_result.get('valence', 0))
        )
        self.second_events.append(event)
    
    def process_minute(self, time: float, mood_state: float):
        """Process minute-scale mood."""
        event = TimescaleEvent(
            time=time,
            timescale='minute',
            event_type='mood',
            intensity=mood_state
        )
        self.minute_events.append(event)
    
    def process_hour(self, time: float, affective_style: str):
        """Process hour-scale affective style."""
        event = TimescaleEvent(
            time=time,
            timescale='hour',
            event_type='affective_style',
            intensity=1.0
        )
        self.hour_events.append(event)


class TimescaleAnalyzer:
    """Analyzes emotional dynamics across timescales."""
    
    def __init__(self, processor: MultiTimescaleProcessor):
        self.processor = processor
    
    def analyze(self) -> Dict:
        """Analyze all timescales."""
        return {
            'millisecond': {
                'n_orienting': len(self.processor.millisecond_events),
                'avg_threat': np.mean([e.intensity for e in self.processor.millisecond_events]) if self.processor.millisecond_events else 0
            },
            'second': {
                'n_appraisals': len(self.processor.second_events),
                'mean_intensity': np.mean([e.intensity for e in self.processor.second_events]) if self.processor.second_events else 0
            },
            'minute': {
                'mood_stability': np.std([e.intensity for e in self.processor.minute_events]) if len(self.processor.minute_events) > 1 else 0
            },
            'hour': {
                'predominant_affect': 'neutral'
            }
        }
