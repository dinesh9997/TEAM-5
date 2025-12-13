# rag/knowledge_base.py
"""
Knowledge Base for Speech Analysis RAG System
Contains expert knowledge about communication, confidence, and personality analysis
"""

# Expert knowledge documents for RAG retrieval
KNOWLEDGE_DOCUMENTS = [
    # Communication Analysis Knowledge
    {
        "id": "comm_001",
        "category": "communication",
        "content": """Speech Rate Analysis Guidelines:
        - Normal speech rate: 120-160 words per minute (WPM)
        - Fast speech (>160 WPM): May indicate excitement, nervousness, or urgency
        - Slow speech (<120 WPM): May indicate thoughtfulness, uncertainty, or emphasis
        - Optimal presentation rate: 140-150 WPM for clarity and engagement
        - Professional speakers typically maintain 130-150 WPM for maximum comprehension"""
    },
    {
        "id": "comm_002", 
        "category": "communication",
        "content": """Clarity and Fluency Indicators:
        - High clarity: Complete sentences, logical flow, minimal filler words
        - Moderate clarity: Some incomplete thoughts, occasional hesitations
        - Low clarity: Frequent restarts, unclear references, disorganized structure
        - Fluency markers: Smooth transitions, consistent pacing, natural pauses
        - Disfluency markers: Excessive 'um', 'uh', false starts, word repetitions"""
    },
    {
        "id": "comm_003",
        "category": "communication", 
        "content": """Vocabulary and Speech Structure:
        - Advanced vocabulary: Domain-specific terms, varied word choice, precise language
        - Intermediate vocabulary: Common professional terms, adequate variety
        - Basic vocabulary: Simple words, limited range, repetitive expressions
        - Well-structured speech: Clear introduction, logical progression, strong conclusion
        - Disorganized speech: Random topic jumps, missing transitions, unclear purpose"""
    },
    
    # Confidence Analysis Knowledge
    {
        "id": "conf_001",
        "category": "confidence",
        "content": """Vocal Confidence Indicators:
        - High confidence: Steady pitch, consistent volume, decisive pauses
        - Moderate confidence: Some pitch variation, adequate projection
        - Low confidence: Pitch instability, volume drops, excessive hesitation
        - Pitch variance 15-25 Hz: Normal, indicates engaged speaking
        - Pitch variance >40 Hz: May indicate nervousness or emotional arousal
        - Pitch variance <10 Hz: May indicate monotone delivery or disengagement"""
    },
    {
        "id": "conf_002",
        "category": "confidence",
        "content": """Pause Analysis for Confidence Assessment:
        - Strategic pauses (0.5-1.5s): Indicate confidence and emphasis
        - Hesitation pauses (>2s): May indicate uncertainty or searching for words
        - Pause ratio <0.15: Good fluency, confident delivery
        - Pause ratio 0.15-0.25: Normal conversational pauses
        - Pause ratio >0.25: May indicate reduced fluency or nervousness
        - Filler-free pauses suggest intentional emphasis and control"""
    },
    {
        "id": "conf_003",
        "category": "confidence",
        "content": """Energy Level and Emotional Tone:
        - High energy: Strong projection, dynamic intonation, engaged delivery
        - Medium energy: Adequate volume, some variation, professional tone
        - Low energy: Soft voice, flat intonation, possible fatigue or disinterest
        - Calm emotion: Steady baseline, controlled responses, measured delivery
        - Engaged emotion: Appropriate enthusiasm, responsive to content
        - Nervous emotion: Elevated baseline, rushed delivery, vocal tension"""
    },
    
    # Personality Analysis Knowledge
    {
        "id": "pers_001",
        "category": "personality",
        "content": """Communication-Based Personality Indicators:
        - Extroverted tendencies: Higher speech rate, more expressive, longer speaking turns
        - Introverted tendencies: Measured pace, thoughtful pauses, concise responses
        - Balanced personality: Adaptable style, moderate expressiveness, situational awareness
        - Note: These are behavioral tendencies, not diagnostic classifications
        - Communication style reflects situational behavior, not fixed personality traits"""
    },
    {
        "id": "pers_002",
        "category": "personality",
        "content": """Assertiveness in Speech Patterns:
        - High assertiveness: Direct statements, clear opinions, confident tone
        - Moderate assertiveness: Balanced approach, considers alternatives
        - Low assertiveness: Hedging language, tentative statements, seeking approval
        - Assertive markers: 'I believe', 'I recommend', definitive conclusions
        - Non-assertive markers: 'Maybe', 'I think perhaps', 'if that's okay'"""
    },
    {
        "id": "pers_003",
        "category": "personality",
        "content": """Expressiveness and Communication Style:
        - High expressiveness: Rich intonation, emotional variety, animated delivery
        - Moderate expressiveness: Professional warmth, appropriate emphasis
        - Low expressiveness: Monotone delivery, limited emotional range
        - Expressive speakers connect better with audiences but may seem less formal
        - Reserved speakers appear more professional but may seem distant
        - Optimal expressiveness depends on context and communication goals"""
    },
    
    # Improvement Recommendations
    {
        "id": "improve_001",
        "category": "improvement",
        "content": """Speech Rate Improvement Tips:
        - If too fast: Practice deliberate pausing, use breathing techniques
        - If too slow: Record and review, focus on key message points
        - Use varied pacing for emphasis - slow down for important points
        - Practice with a metronome or pacing app for consistency
        - Record practice sessions and compare to target rate"""
    },
    {
        "id": "improve_002",
        "category": "improvement",
        "content": """Confidence Building Techniques:
        - Power posing before speaking can increase confidence hormones
        - Preparation reduces uncertainty and builds natural confidence
        - Start with strong opening statements to establish authority
        - Practice diaphragmatic breathing for voice steadiness
        - Record yourself to identify and address nervous habits
        - Focus on message value rather than self-evaluation during delivery"""
    },
    {
        "id": "improve_003",
        "category": "improvement",
        "content": """Communication Clarity Enhancement:
        - Use the PREP method: Point, Reason, Example, Point
        - Eliminate filler words through awareness and practice
        - Structure thoughts before speaking using mental outlines
        - Practice transitional phrases for smooth topic changes
        - Use concrete examples to illustrate abstract concepts
        - Pause before complex explanations to organize thoughts"""
    }
]


class KnowledgeBase:
    """Manages the knowledge documents for RAG retrieval"""
    
    def __init__(self):
        self.documents = KNOWLEDGE_DOCUMENTS
    
    def get_all_documents(self):
        """Return all knowledge documents"""
        return self.documents
    
    def get_documents_by_category(self, category: str):
        """Filter documents by category"""
        return [doc for doc in self.documents if doc["category"] == category]
    
    def get_document_texts(self):
        """Return list of document contents for embedding"""
        return [doc["content"] for doc in self.documents]
    
    def get_document_metadatas(self):
        """Return metadata for each document"""
        return [{"id": doc["id"], "category": doc["category"]} for doc in self.documents]
