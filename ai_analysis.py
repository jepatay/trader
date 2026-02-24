"""
ai_analysis.py — sends indicator data + news to OpenAI and gets a
structured short-term trading recommendation.
"""

import os
import json
import openai


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    _CLIENT = openai.OpenAI(api_key=api_key)
    return _CLIENT


def analyse(ticker: str, info: dict, indicators: dict, news: list[dict]) -> dict:
    """
    Ask GPT-4o for a short-term (1-3 day) trading recommendation.

    Returns a dict:
      {
        "action":     "BUY" | "SELL" | "HOLD",
        "confidence": "High" | "Medium" | "Low",
        "reasoning":  "...",
        "risks":      "...",
        "entry_tip":  "...",
        "available":  True | False
      }
    """
    client = _get_client()
    if client is None:
        return {
            "action":     indicators.get("signal", "HOLD"),
            "confidence": "Low",
            "reasoning":  "AI layer unavailable — set OPENAI_API_KEY. Signal from technical indicators only.",
            "risks":      "No AI analysis performed.",
            "entry_tip":  "",
            "available":  False,
        }

    vals = indicators.get("values", {})
    interp = indicators.get("interp", {})
    score = indicators.get("score", 0)
    tech_signal = indicators.get("signal", "HOLD")

    news_text = "\n".join(
        f"- [{n['published']}] {n['title']} ({n['publisher']})"
        for n in (news or [])[:6]
    ) or "No recent news available."

    prompt = f"""You are a short-term trading analyst helping a retail investor.
Analyse the following data for **{ticker}** ({info.get('name', ticker)}, {info.get('exchange','?')}, {info.get('currency','?')}) and give a recommendation for the next 1-3 days.

## Technical Indicators (latest values)
- Close price:  {vals.get('close')}
- RSI (14):     {vals.get('rsi')} → {interp.get('rsi','')}
- MACD:         {vals.get('macd')} vs signal {vals.get('macd_signal')} → {interp.get('macd','')}
- EMA 9/21:     {vals.get('ema_short')} / {vals.get('ema_long')} → {interp.get('ema_cross','')}
- Bollinger:    upper={vals.get('bb_upper')} mid={vals.get('bb_mid')} lower={vals.get('bb_lower')} → {interp.get('bb','')}
- Stochastic:   %K={vals.get('stoch_k')} %D={vals.get('stoch_d')} → {interp.get('stoch','')}
- ATR (14):     {vals.get('atr')}
- Volume:       {vals.get('volume')} vs avg {vals.get('volume_avg')} → {interp.get('volume','')}
- Composite score: {score}/3  →  Technical signal: **{tech_signal}**

## Recent News Headlines
{news_text}

## Task
Respond with a JSON object (no markdown, just raw JSON) with exactly these keys:
- "action": one of "BUY", "SELL", or "HOLD"
- "confidence": one of "High", "Medium", or "Low"
- "reasoning": 2-4 sentences explaining the recommendation, referencing specific indicators and/or news
- "risks": 1-2 sentences on the main risks for this trade
- "entry_tip": brief note on timing or price level to watch (e.g. "wait for RSI to dip below 40 before entering")

Be concise, honest, and remember this investor holds for only 1-3 days."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        result["available"] = True
        return result
    except json.JSONDecodeError:
        return {
            "action":     tech_signal,
            "confidence": "Low",
            "reasoning":  f"AI returned unparseable response. Falling back to technical signal ({tech_signal}).",
            "risks":      "AI parse error — treat with caution.",
            "entry_tip":  "",
            "available":  True,
        }
    except Exception as e:
        return {
            "action":     tech_signal,
            "confidence": "Low",
            "reasoning":  f"AI call failed: {e}. Technical signal: {tech_signal}.",
            "risks":      "AI unavailable.",
            "entry_tip":  "",
            "available":  False,
        }
