"""Offline evaluation utilities for TradingAgents.

Currently: rating-calibration scoring over the persistent decision log
(`tradingagents.eval.calibration_report`). Everything in this package must run
fully offline — no LLM calls, no network, no API keys.
"""
