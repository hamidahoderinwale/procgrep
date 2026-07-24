"""Procgrep project palette and Altair theme."""

import altair as alt

## Gradient within Anthropic family (older to newer to latest); ANTHROPIC is the family anchor
ANTHROPIC_D = "#1A5276"  # Claude-3 Opus (darkest)
ANTHROPIC = "#2471A3"  # Claude-3.5 Sonnet; steel blue anchors the Claude family
ANTHROPIC_M = "#5DADE2"  # Claude-4 Sonnet (medium)
ANTHROPIC_L = "#AED6F1"  # Claude-4.5 Sonnet (lightest)

## Gradient within OpenAI family; OPENAI is the family anchor
OPENAI_D = "#7B241C"  # GPT-4 (darker)
OPENAI = "#B03A2E"  # GPT-4o; terracotta anchors the GPT family
OPENAI_L = "#E59866"  # GPT-5 (lighter)

## SFT-distilled (single agent)
SFT = "#1A7A4A"

## Generic
NEAR_BLACK = "#212121"
GRAY = "#9E9E9E"
GRAY_LIGHT = "#E0E0E0"

## Aliases expected by skill boilerplate
BLUE = ANTHROPIC
GREEN = SFT
MAGENTA = OPENAI
COPPER = "#CB4D20"
OLIVE = "#6D6B5E"


AGENT_COLORS = {
    "Claude-3 Opus": "#1A5276",
    "Claude-3.5 Sonnet": "#2471A3",
    "Claude-4 Sonnet": "#5DADE2",
    "Claude-4.5 Sonnet": "#AED6F1",
    "GPT-4": "#7B241C",
    "GPT-4o": "#B03A2E",
    "GPT-5": "#E59866",
    "SWE-agent-LM-32B": "#1A7A4A",
}

FAMILY_COLORS = {
    "Anthropic": "#2471A3",
    "OpenAI": "#B03A2E",
    "SFT-distilled": "#1A7A4A",
}


def _theme():
    return {
        "config": {
            "view": {"strokeOpacity": 0},
            "axis": {
                "grid": False,
                "domain": True,
                "domainColor": NEAR_BLACK,
                "tickColor": NEAR_BLACK,
                "labelColor": NEAR_BLACK,
                "titleColor": NEAR_BLACK,
                "labelFontSize": 10,
                "titleFontSize": 11,
                "labelFont": "sans-serif",
                "titleFont": "sans-serif",
            },
            "header": {
                "labelFontSize": 10,
                "titleFontSize": 11,
                "labelFont": "sans-serif",
            },
            "legend": {
                "labelFontSize": 9,
                "titleFontSize": 10,
                "labelFont": "sans-serif",
                "titleFont": "sans-serif",
                "strokeColor": None,
                "fillColor": "white",
            },
            "title": {
                "fontSize": 12,
                "anchor": "start",
                "fontWeight": "normal",
                "font": "sans-serif",
                "color": NEAR_BLACK,
            },
            "background": "white",
            "padding": 14,
            "mark": {"font": "sans-serif"},
        }
    }


def register():
    alt.themes.register("procgrep", _theme)
    alt.themes.enable("procgrep")
