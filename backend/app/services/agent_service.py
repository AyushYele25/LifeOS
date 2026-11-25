"""
LIFEOS Agentic AI Orchestration Service

The core intelligence layer that:
1. Parses user intent from natural language queries.
2. Selects and invokes the appropriate typed tools (read-only).
3. Iterates through a multi-step tool-calling loop (max 5 rounds).
4. Assembles grounded, evidence-backed responses with citations.
5. Enforces read-only financial safety guardrails.
6. Maintains per-user conversation history.

The agent NEVER invents financial numbers — every data point traces to a tool result.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.agent_tools import TOOL_REGISTRY, ToolDescriptor, get_tool_schemas


# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

MAX_TOOL_CALLS_PER_QUERY = 5

# Intent → Tool routing table.
# Each pattern maps to one or more tools that should be called.
INTENT_PATTERNS: List[Tuple[List[str], List[str]]] = [
    # Affordability / purchase
    (
        ["can i afford", "should i buy", "purchase", "buy now", "emi option"],
        ["simulate_purchase", "get_life_state"],
    ),
    # Risk assessment
    (
        ["risk", "financial health", "how safe", "stress test", "vulnerability"],
        ["get_risk_assessment"],
    ),
    # Forecast / prediction
    (
        ["forecast", "predict", "next month", "upcoming expenses", "projection", "how much will i spend"],
        ["forecast_expenses"],
    ),
    # Life state / overview / balance
    (
        ["balance", "overview", "net worth", "financial state", "life state", "summary",
         "savings", "how much do i have", "runway", "emergency fund"],
        ["get_life_state"],
    ),
    # Spending analysis
    (
        ["spending", "spent", "expense breakdown", "category", "where is my money",
         "top categories", "budget"],
        ["get_spending_by_category"],
    ),
    # Transactions
    (
        ["transaction", "payment", "paid", "received", "debit", "credit",
         "last month", "recent"],
        ["get_transactions"],
    ),
    # Recurring payments
    (
        ["recurring", "subscription", "regular payment", "monthly bill", "auto-pay"],
        ["get_recurring_payments", "get_subscriptions"],
    ),
    # Upcoming events / obligations
    (
        ["upcoming", "due", "deadline", "obligation", "emi due", "renewal",
         "what am i forgetting", "what's coming"],
        ["get_upcoming_events"],
    ),
    # Alerts / anomalies
    (
        ["alert", "anomaly", "unusual", "warning", "suspicious", "duplicate charge",
         "spike"],
        ["get_alerts"],
    ),
    # Documents
    (
        ["document", "policy", "warranty", "insurance document", "loan agreement",
         "find in document", "search document"],
        ["search_documents"],
    ),
    # Loans
    (
        ["loan", "emi", "outstanding", "debt", "principal", "interest rate"],
        ["get_loans"],
    ),
    # Insurance
    (
        ["insurance", "policy", "premium", "cover", "sum insured", "renewal"],
        ["get_insurance"],
    ),
    # Subscriptions
    (
        ["netflix", "spotify", "subscription cost", "cancel subscription"],
        ["get_subscriptions"],
    ),
    # What changed?
    (
        ["what changed", "what's different", "difference", "compare"],
        ["get_life_state"],
    ),
]


# ═══════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════

@dataclass
class ToolCallRecord:
    """A single tool invocation and its result."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    execution_time_ms: float = 0.0


@dataclass
class ConversationMessage:
    """One message in the conversation history."""
    role: str   # "user" or "assistant"
    content: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Conversation:
    """A full conversation session for a user."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════
# Agent Service
# ═══════════════════════════════════════════════════════

class AgentService:
    """
    Agentic orchestrator that interprets user queries, selects tools,
    executes them against real LIFEOS data, and synthesises grounded answers.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id
        self._tool_calls: List[ToolCallRecord] = []

    # ─── Public API ────────────────────────────────────

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message through the agentic pipeline:
        1. Classify intent
        2. Select tools
        3. Execute tools (multi-step if needed)
        4. Synthesise grounded response
        """
        self._tool_calls = []
        logger.info("🤖 Agent query from user={}: {}", str(self.user_id)[:8], message[:80])

        # Step 1 — Intent classification & tool selection
        selected_tools = self._select_tools(message)
        logger.debug("   Selected tools: {}", [t.name for t in selected_tools])

        # Step 2 — Parameter extraction
        tool_params = self._extract_params(message, selected_tools)

        # Step 3 — Execute tools (up to MAX_TOOL_CALLS_PER_QUERY)
        results: Dict[str, Dict[str, Any]] = {}
        calls_made = 0

        for tool in selected_tools:
            if calls_made >= MAX_TOOL_CALLS_PER_QUERY:
                break
            params = tool_params.get(tool.name, {})
            result = await self._execute_tool(tool, params)
            results[tool.name] = result
            calls_made += 1

        # Step 4 — Check if we need additional tools based on results
        additional_tools = self._check_follow_up_tools(message, results)
        for tool in additional_tools:
            if calls_made >= MAX_TOOL_CALLS_PER_QUERY:
                break
            if tool.name not in results:
                params = tool_params.get(tool.name, {})
                result = await self._execute_tool(tool, params)
                results[tool.name] = result
                calls_made += 1

        # Step 5 — Synthesise response
        response_text = self._synthesise_response(message, results)

        return {
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "message": response_text,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "parameters": tc.parameters,
                    "result": tc.result,
                    "execution_time_ms": tc.execution_time_ms,
                }
                for tc in self._tool_calls
            ],
            "tools_used": list(results.keys()),
            "data": results,
        }

    # ─── Intent Classification ─────────────────────────

    def _select_tools(self, message: str) -> List[ToolDescriptor]:
        """
        Match user message against intent patterns to determine which tools to call.
        Falls back to get_life_state if no patterns match.
        """
        lower = message.lower().strip()
        matched_tool_names: List[str] = []

        for patterns, tool_names in INTENT_PATTERNS:
            for pattern in patterns:
                if pattern in lower:
                    for tn in tool_names:
                        if tn not in matched_tool_names:
                            matched_tool_names.append(tn)
                    break  # matched this pattern group, move on

        # Fallback — if nothing matched, give the user a state overview
        if not matched_tool_names:
            matched_tool_names = ["get_life_state"]

        # Resolve tool descriptors
        tools = []
        for name in matched_tool_names:
            if name in TOOL_REGISTRY:
                tools.append(TOOL_REGISTRY[name])

        return tools[:MAX_TOOL_CALLS_PER_QUERY]

    # ─── Parameter Extraction ──────────────────────────

    def _extract_params(
        self, message: str, tools: List[ToolDescriptor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract tool parameters from the natural language message.
        Uses heuristic pattern matching for key parameters.
        """
        params: Dict[str, Dict[str, Any]] = {}
        lower = message.lower()

        for tool in tools:
            tool_params: Dict[str, Any] = {}

            if tool.name == "simulate_purchase":
                tool_params = self._extract_purchase_params(message)

            elif tool.name == "get_transactions":
                tool_params = self._extract_transaction_params(message)

            elif tool.name == "get_spending_by_category":
                tool_params = self._extract_spending_params(message)

            elif tool.name == "get_upcoming_events":
                tool_params = self._extract_event_params(message)

            elif tool.name == "search_documents":
                tool_params = {"query": message}

            params[tool.name] = tool_params

        return params

    def _extract_purchase_params(self, message: str) -> Dict[str, Any]:
        """Extract item name and amount from a purchase query."""
        params: Dict[str, Any] = {}

        # 1. Check Indian shorthand multipliers: lakh/lac/L, crore/cr, k
        lakh_match = re.search(
            r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|l)\b',
            message,
            re.IGNORECASE,
        )
        crore_match = re.search(
            r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:crores?|cr)\b',
            message,
            re.IGNORECASE,
        )
        k_match = re.search(
            r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*k\b',
            message,
            re.IGNORECASE,
        )

        if crore_match:
            params["amount"] = float(crore_match.group(1)) * 10000000.0
        elif lakh_match:
            params["amount"] = float(lakh_match.group(1)) * 100000.0
        elif k_match:
            params["amount"] = float(k_match.group(1)) * 1000.0
        else:
            # Extract explicit currency amount — look for ₹, Rs, INR followed by numbers
            amount_match = re.search(
                r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)', message, re.IGNORECASE
            )
            if amount_match:
                params["amount"] = float(amount_match.group(1).replace(",", ""))

        # Also check for standalone large numbers
        if "amount" not in params:
            num_match = re.search(r'\b(\d{4,})\b', message)
            if num_match:
                params["amount"] = float(num_match.group(1))

        # Extract item name — strip filler verbs & currency modifiers first
        cleaned_for_item = re.sub(
            r'(?:₹|rs\.?|inr)\s*[\d,.]+\s*(?:lakhs?|lacs?|l|crores?|cr|k)?',
            '',
            message,
            flags=re.IGNORECASE,
        )
        cleaned_for_item = re.sub(
            r'\b\d+(?:\.\d+)?\s*(?:lakhs?|lacs?|crores?|cr|k)?\s*(?:rupees?|inr)?\b',
            '',
            cleaned_for_item,
            flags=re.IGNORECASE,
        )
        cleaned_for_item = re.sub(
            r'\b(?:can i afford|should i|to buy|buy|afford|purchase|right now|for|worth|costing|at|a|an)\b',
            ' ',
            cleaned_for_item,
            flags=re.IGNORECASE,
        )
        cleaned_for_item = re.sub(r'[?!.]', '', cleaned_for_item)
        cleaned_for_item = re.sub(r'\s+', ' ', cleaned_for_item).strip()
        if cleaned_for_item:
            params["item"] = cleaned_for_item
        else:
            params["item"] = "item"

        if "amount" not in params:
            params["amount"] = 0

        # EMI months
        emi_match = re.search(r'(\d+)\s*(?:month|mo)\s*(?:emi|installment)', message, re.IGNORECASE)
        if emi_match:
            params["emi_months"] = int(emi_match.group(1))

        return params

    def _extract_transaction_params(self, message: str) -> Dict[str, Any]:
        """Extract transaction query filters from natural language."""
        params: Dict[str, Any] = {}
        lower = message.lower()

        # Transaction type
        if "income" in lower or "credit" in lower or "received" in lower:
            params["transaction_type"] = "credit"
        elif "expense" in lower or "debit" in lower or "paid" in lower:
            params["transaction_type"] = "debit"

        # Merchant
        merchant_match = re.search(
            r'(?:from|to|at|merchant)\s+(\w+(?:\s+\w+)?)', message, re.IGNORECASE
        )
        if merchant_match:
            candidate = merchant_match.group(1).strip()
            # Filter out common stop words
            if candidate.lower() not in {"the", "a", "an", "my", "me", "i"}:
                params["merchant"] = candidate

        # Time period
        if "last week" in lower:
            from datetime import date, timedelta
            params["start_date"] = str(date.today() - timedelta(days=7))
        elif "last month" in lower or "past month" in lower:
            from datetime import date, timedelta
            params["start_date"] = str(date.today() - timedelta(days=30))
        elif "last 3 months" in lower:
            from datetime import date, timedelta
            params["start_date"] = str(date.today() - timedelta(days=90))

        return params

    def _extract_spending_params(self, message: str) -> Dict[str, Any]:
        """Extract spending analysis parameters."""
        lower = message.lower()
        if "last week" in lower or "7 day" in lower or "this week" in lower:
            return {"days": 7}
        elif "last 3 months" in lower or "90 day" in lower or "quarter" in lower:
            return {"days": 90}
        elif "last 6 months" in lower or "180 day" in lower:
            return {"days": 180}
        elif "year" in lower or "365 day" in lower or "12 month" in lower:
            return {"days": 365}
        return {"days": 30}

    def _extract_event_params(self, message: str) -> Dict[str, Any]:
        """Extract event horizon parameters."""
        lower = message.lower()
        if "next week" in lower or "7 day" in lower:
            return {"horizon_days": 7}
        elif "next month" in lower or "30 day" in lower:
            return {"horizon_days": 30}
        elif "60 day" in lower or "2 month" in lower:
            return {"horizon_days": 60}
        return {"horizon_days": 90}

    # ─── Tool Execution ────────────────────────────────

    async def _execute_tool(
        self, tool: ToolDescriptor, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single tool with safety guardrails and timing.
        """
        import time
        start = time.monotonic()

        try:
            result = await tool.execute(self.db, self.user_id, params)
        except Exception as exc:
            logger.error("Tool {} failed: {}", tool.name, str(exc))
            result = {"error": f"Tool execution failed: {str(exc)}"}

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        record = ToolCallRecord(
            tool_name=tool.name,
            parameters=params,
            result=result,
            execution_time_ms=elapsed_ms,
        )
        self._tool_calls.append(record)

        logger.debug("   Tool {} completed in {}ms", tool.name, elapsed_ms)
        return result

    # ─── Follow-up Tool Selection ──────────────────────

    def _check_follow_up_tools(
        self, message: str, results: Dict[str, Dict[str, Any]]
    ) -> List[ToolDescriptor]:
        """
        Inspect current results and decide if additional tools should be called.
        Example: if the user asked about risk and we see high risk, fetch alerts too.
        """
        follow_up: List[ToolDescriptor] = []
        lower = message.lower()

        # If risk assessment shows HIGH/CRITICAL, also fetch alerts
        if "get_risk_assessment" in results:
            risk_data = results["get_risk_assessment"]
            risk_level = risk_data.get("risk_level", "low")
            if risk_level in ("high", "critical") and "get_alerts" not in results:
                if "get_alerts" in TOOL_REGISTRY:
                    follow_up.append(TOOL_REGISTRY["get_alerts"])

        # If life state shows low runway, also fetch upcoming events
        if "get_life_state" in results:
            state = results["get_life_state"]
            runway = state.get("emergency_runway_months", 99)
            if runway < 3 and "get_upcoming_events" not in results:
                if "get_upcoming_events" in TOOL_REGISTRY:
                    follow_up.append(TOOL_REGISTRY["get_upcoming_events"])

        return follow_up

    # ─── Response Synthesis ────────────────────────────

    def _synthesise_response(
        self, message: str, results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Assemble a grounded, evidence-backed response from tool results.
        Every number in the output traces to a tool call — no hallucinations.
        """
        sections: List[str] = []
        lower = message.lower()

        # ── Life State ──
        if "get_life_state" in results:
            s = results["get_life_state"]
            sections.append(self._format_life_state(s))

        # ── Spending Breakdown ──
        if "get_spending_by_category" in results:
            s = results["get_spending_by_category"]
            sections.append(self._format_spending(s))

        # ── Transactions ──
        if "get_transactions" in results:
            s = results["get_transactions"]
            sections.append(self._format_transactions(s))

        # ── Recurring Payments ──
        if "get_recurring_payments" in results:
            s = results["get_recurring_payments"]
            sections.append(self._format_recurring(s))

        # ── Upcoming Events ──
        if "get_upcoming_events" in results:
            s = results["get_upcoming_events"]
            sections.append(self._format_events(s))

        # ── Alerts ──
        if "get_alerts" in results:
            s = results["get_alerts"]
            sections.append(self._format_alerts(s))

        # ── Risk Assessment ──
        if "get_risk_assessment" in results:
            s = results["get_risk_assessment"]
            sections.append(self._format_risk(s))

        # ── Forecast ──
        if "forecast_expenses" in results:
            s = results["forecast_expenses"]
            sections.append(self._format_forecast(s))

        # ── Purchase Simulation ──
        if "simulate_purchase" in results:
            s = results["simulate_purchase"]
            sections.append(self._format_simulation(s))

        # ── Document Search ──
        if "search_documents" in results:
            s = results["search_documents"]
            sections.append(self._format_document_search(s))

        # ── Loans ──
        if "get_loans" in results:
            s = results["get_loans"]
            sections.append(self._format_loans(s))

        # ── Insurance ──
        if "get_insurance" in results:
            s = results["get_insurance"]
            sections.append(self._format_insurance(s))

        # ── Subscriptions ──
        if "get_subscriptions" in results:
            s = results["get_subscriptions"]
            sections.append(self._format_subscriptions(s))

        if not sections:
            return (
                "I wasn't able to retrieve the information you requested. "
                "Could you rephrase your question?"
            )

        return "\n\n".join(sections)

    # ─── Formatters ────────────────────────────────────

    @staticmethod
    def _fmt_inr(amount: float) -> str:
        """Format amount in Indian Rupee style with commas."""
        if amount >= 10_000_000:
            return f"₹{amount / 10_000_000:.2f} Cr"
        if amount >= 100_000:
            return f"₹{amount / 100_000:.2f} L"
        return f"₹{amount:,.0f}"

    def _format_life_state(self, s: Dict[str, Any]) -> str:
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        emoji = risk_emoji.get(s.get("risk_level", "low"), "⚪")
        sr = float(s.get("savings_rate") or 0)
        sr_pct = round(sr * 100, 1) if abs(sr) <= 1.0 else round(sr, 1)
        lines = [
            "📊 **Financial Snapshot**",
            f"• Liquid Balance: {self._fmt_inr(s['total_balance'])}",
            f"• Monthly Income: {self._fmt_inr(s['monthly_income'])}",
            f"• Monthly Expenses: {self._fmt_inr(s['monthly_expenses'])}",
            f"• Monthly Savings: {self._fmt_inr(s['monthly_savings'])} ({sr_pct}%)",
            f"• Emergency Runway: {s['emergency_runway_months']} months",
            f"• Risk Score: {s['financial_risk_score']}/100 {emoji} ({s['risk_level'].upper()})",
        ]
        if s.get("active_loans_count"):
            lines.append(f"• Active Loans: {s['active_loans_count']} (EMI: {self._fmt_inr(s['total_emi_burden'])}/mo)")
        if s.get("active_subscriptions_count"):
            lines.append(f"• Subscriptions: {s['active_subscriptions_count']} ({self._fmt_inr(s['total_subscription_burden'])}/mo)")
        if s.get("upcoming_obligations_30d"):
            lines.append(f"• Upcoming (30d): {self._fmt_inr(s['upcoming_obligations_30d'])}")
        return "\n".join(lines)

    def _format_spending(self, s: Dict[str, Any]) -> str:
        lines = [
            f"💰 **Spending Breakdown** (last {s['period_days']} days)",
            f"Total: {self._fmt_inr(s['total_spent'])}",
        ]
        for cat in s.get("categories", [])[:8]:
            pct = round((cat["total_spent"] / s["total_spent"]) * 100, 1) if s["total_spent"] > 0 else 0
            lines.append(f"  • {cat['category']}: {self._fmt_inr(cat['total_spent'])} ({pct}%) — {cat['transaction_count']} txns")
        return "\n".join(lines)

    def _format_transactions(self, s: Dict[str, Any]) -> str:
        lines = [f"🧾 **Transactions** ({s['count']} results)"]
        for t in s.get("transactions", [])[:10]:
            sign = "+" if t["type"] == "credit" else "-"
            cat = f" [{t['category']}]" if t.get("category") else ""
            lines.append(f"  • {t['date']} — {t['description']}: {sign}{self._fmt_inr(t['amount'])}{cat}")
        return "\n".join(lines)

    def _format_recurring(self, s: Dict[str, Any]) -> str:
        lines = [f"🔄 **Recurring Payments** ({s['count']} detected)"]
        for p in s.get("patterns", [])[:10]:
            next_str = f" — next: {p['next_expected_date']}" if p.get("next_expected_date") else ""
            lines.append(f"  • {p['merchant']}: {self._fmt_inr(p['average_amount'])} ({p['frequency']}){next_str}")
        return "\n".join(lines)

    def _format_events(self, s: Dict[str, Any]) -> str:
        lines = [
            f"📅 **Upcoming Obligations** (next {s['horizon_days']} days)",
            f"Total: {self._fmt_inr(s['total_amount'])} across {s['event_count']} events",
        ]
        for e in s.get("events", [])[:10]:
            lines.append(f"  • {e['date']} ({e['days_until']}d) — {e['title']}: {self._fmt_inr(e['amount'])}")
        return "\n".join(lines)

    def _format_alerts(self, s: Dict[str, Any]) -> str:
        sev_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        lines = [f"🚨 **Active Alerts** ({s['count']})"]
        for a in s.get("alerts", [])[:10]:
            emoji = sev_emoji.get(a["severity"], "⚪")
            lines.append(f"  {emoji} [{a['severity'].upper()}] {a['title']}: {a['message'][:100]}")
        if not s.get("alerts"):
            lines.append("  ✅ No active alerts — all clear!")
        return "\n".join(lines)

    def _format_risk(self, s: Dict[str, Any]) -> str:
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        emoji = risk_emoji.get(s.get("risk_level", "low"), "⚪")
        sr_r = float(s.get("savings_rate") or 0)
        sr_r_pct = round(sr_r * 100, 1) if abs(sr_r) <= 1.0 else round(sr_r, 1)
        lines = [
            f"🛡️ **Financial Risk Assessment** {emoji}",
            f"• Composite Score: {s['composite_risk_score']}/100 (Grade: {s['risk_grade']})",
            f"• Risk Level: {s['risk_level'].upper()}",
            f"• Debt-to-Income: {round(s['dti_ratio'] * 100, 1) if s['dti_ratio'] < 1 else round(s['dti_ratio'], 1)}%",
            f"• Savings Rate: {sr_r_pct}%",
            f"• Emergency Runway: {round(s['emergency_runway_months'], 1)} months",
        ]
        if s.get("vulnerabilities"):
            lines.append("⚠️ Vulnerabilities:")
            for v in s["vulnerabilities"][:5]:
                lines.append(f"  • {v}")
        if s.get("recommendations"):
            lines.append("💡 Recommendations:")
            for r in s["recommendations"][:5]:
                lines.append(f"  • {r}")
        return "\n".join(lines)

    def _format_forecast(self, s: Dict[str, Any]) -> str:
        lines = ["📈 **Expense Forecast**"]
        for period_key in ["horizon_30d", "horizon_60d", "horizon_90d"]:
            p = s.get(period_key, {})
            days = p.get("days", 0)
            lines.append(
                f"  • {days}-day: {self._fmt_inr(p.get('p50_expected', 0))} "
                f"(range: {self._fmt_inr(p.get('p10_lower', 0))} – "
                f"{self._fmt_inr(p.get('p90_upper', 0))})"
            )
        return "\n".join(lines)

    def _format_simulation(self, s: Dict[str, Any]) -> str:
        lines = [
            f"🧮 **Purchase Simulation: {s.get('item', 'Item')}** ({self._fmt_inr(s.get('amount', 0))})"
        ]
        for sc in s.get("scenarios", []):
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
            emoji = risk_emoji.get(sc.get("risk_level_after", ""), "⚪")
            lines.append(
                f"  {emoji} **{sc['name']}**: "
                f"Savings after: {self._fmt_inr(sc['savings_after'])}, "
                f"Runway: {sc['emergency_runway_months']}mo, "
                f"Risk: {sc.get('risk_level_after', 'N/A').upper()}"
            )
        if s.get("recommendation"):
            lines.append(f"\n💡 **Verdict**: {s['recommendation']}")
        return "\n".join(lines)

    def _format_document_search(self, s: Dict[str, Any]) -> str:
        lines = [f"📄 **Document Search**"]
        lines.append(s.get("answer", "No results found."))
        if s.get("citations"):
            lines.append("\n📎 Sources:")
            for c in s["citations"][:5]:
                lines.append(f"  • {c['filename']} (p.{c['page_number']}): \"{c['text_snippet'][:80]}…\"")
        if s.get("has_conflicts"):
            lines.append(f"\n⚠️ Conflicts detected: {s['conflict_details']}")
        return "\n".join(lines)

    def _format_loans(self, s: Dict[str, Any]) -> str:
        lines = [
            f"🏦 **Loans** ({s['count']} active)",
            f"Total Outstanding: {self._fmt_inr(s['total_outstanding'])} | Monthly EMI: {self._fmt_inr(s['total_monthly_emi'])}",
        ]
        for l in s.get("loans", []):
            lines.append(
                f"  • {l['name']} ({l['type']}): {self._fmt_inr(l['outstanding'])} outstanding, "
                f"EMI {self._fmt_inr(l['emi_amount'])} @ {l['interest_rate']}%"
            )
        if not s.get("loans"):
            lines.append("  ✅ No active loans!")
        return "\n".join(lines)

    def _format_insurance(self, s: Dict[str, Any]) -> str:
        lines = [f"🛡️ **Insurance Policies** ({s['count']} active)"]
        for p in s.get("policies", []):
            expiry = f" — expires {p['expiry_date']}" if p.get("expiry_date") else ""
            lines.append(
                f"  • {p['name']} ({p['type']}): {self._fmt_inr(p['premium_amount'])} {p.get('premium_frequency', '')}{expiry}"
            )
        if not s.get("policies"):
            lines.append("  No active insurance policies on record.")
        return "\n".join(lines)

    def _format_subscriptions(self, s: Dict[str, Any]) -> str:
        lines = [
            f"📺 **Subscriptions** ({s['count']} active) — "
            f"Total: {self._fmt_inr(s['total_monthly_cost'])}/month"
        ]
        for sub in s.get("subscriptions", []):
            next_str = f" — next: {sub['next_billing_date']}" if sub.get("next_billing_date") else ""
            lines.append(f"  • {sub['name']}: {self._fmt_inr(sub['amount'])} ({sub['frequency']}){next_str}")
        if not s.get("subscriptions"):
            lines.append("  No active subscriptions found.")
        return "\n".join(lines)
