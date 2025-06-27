from dataclasses import dataclass


@dataclass
class TopicStepResult:
    topic_prompt: str
    result: str
