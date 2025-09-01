import os
import asyncio
from dotenv import load_dotenv

# === Models (LLM1/LLM2/LLM3) ==========================
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

# === Agents & Teaming =================================
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

async def main():
    load_dotenv()

    # ------ LLMs (as in your request) ------
    model_client_openai_1 = OpenAIChatCompletionClient(   # LLM1
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    model_client_openai_2 = OpenAIChatCompletionClient(   # LLM2
        model="gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    model_client_anthropic = AnthropicChatCompletionClient(  # LLM3
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    # ===== WORKERS (mapped to LLM1/LLM2/LLM3) ==========
    ux_researcher = AssistantAgent(
        name="ux_researcher",
        model_client=model_client_openai_1,  # LLM1
        system_message=(
            "Role: UX Researcher. Break down user needs, JTBD, pain points, "
            "and success metrics. Produce concise bullets and cite assumptions. "
            "Say 'APPROVED' when your research notes are ready."
        ),
    )

    tech_lead = AssistantAgent(
        name="tech_lead",
        model_client=model_client_openai_2,  # LLM2
        system_message=(
            "Role: Tech Lead / Engineer. Propose a simple, feasible architecture, "
            "interfaces, and a minimal backlog. Flag risks and unknowns. "
            "Say 'APPROVED' when your plan is ready."
        ),
    )

    qa_compliance = AssistantAgent(
        name="qa_compliance",
        model_client=model_client_anthropic,  # LLM3
        system_message=(
            "Role: QA & Compliance. Review for safety, privacy, and regulatory fit "
            "(ISO/OSHA/IEC basics). List testable acceptance criteria. "
            "Say 'APPROVED' when review is ready."
        ),
    )

    # Inner workers’ loop stops once all hand off.
    workers_termination = TextMentionTermination(text="APPROVED") | MaxMessageTermination(max_messages=8)

    workers_team = RoundRobinGroupChat(
        participants=[ux_researcher, tech_lead, qa_compliance],
        termination_condition=workers_termination,
    )

    # ===== ORCHESTRATOR (Product Manager) ===============
    orchestrator_pm = SocietyOfMindAgent(
        name="orchestrator_pm",
        team=workers_team,
        model_client=model_client_openai_2,  # Uses LLM2 to plan/route
    )

    # ===== SYNTHESIZER (Tech Writer) ====================
    synthesizer = AssistantAgent(
        name="synthesizer",
        model_client=model_client_openai_1,  # Uses LLM1 to compile
        system_message=(
            "Role: Tech Writer (Synthesizer). Combine the orchestrator’s summary and "
            "worker outputs into a crisp deliverable (one page). Include: "
            "Problem, Target Users, Proposed Solution, System Sketch, MVP Scope, "
            "Acceptance Criteria, Risks & Next Steps. End with 'FINALIZED'."
        ),
    )

    # ===== OUTER LOOP (Orchestrator → Synthesizer) ======
    outer_termination = TextMentionTermination(text="FINALIZED") | MaxMessageTermination(max_messages=6)

    final_team = RoundRobinGroupChat(
        participants=[orchestrator_pm, synthesizer],
        termination_condition=outer_termination,
    )

    # Example product task (replace as needed)
    task = (
        "IN: Create a one-page MVP brief for a burnout monitoring software. "
        "The system should track employee well-being signals (working hours, "
        "self-reported stress surveys, keyboard/mouse activity, optional wearable data). "
        "It must provide managers with aggregated, anonymized dashboards, "
        "offer early warning indicators, and suggest preventive actions. "
        "Ensure compliance with privacy and labor regulations."
    )

    stream = final_team.run_stream(task=task)
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())
