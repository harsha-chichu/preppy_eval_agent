"""Prompt for ``transcribe_node`` - Gemini transcribes + assesses a spoken answer."""

AUDIO_PROMPT = """\
You are given an audio recording of a candidate answering an interview-style
question out loud. Do two things and return them as JSON matching the
AudioAnalysis schema.

Question the candidate was answering: {actual_question}

1. transcript: a verbatim, punctuated transcript of what the candidate said. Do
   not paraphrase, summarise, correct grammar, or add words they did not say.

2. communication: assess ONLY delivery, not whether the answer is correct:
   - grammar_score (0-10) and grammar_issues: concrete grammatical mistakes heard
   - pronunciation_score (0-10) and mispronounced_words: words clearly mispronounced
   - fluency_score (0-10): smoothness / pace / absence of stumbling
   - filler_words_used: distinct fillers heard ("um", "uh", "like", "you know", ...)
   - pause_count: number of noticeable/awkward pauses
   - clarity_score (0-10): overall articulation and how easy the speaker is to follow
   - communication_feedback: 1-3 sentences summarising delivery quality

Return the JSON only.
"""
