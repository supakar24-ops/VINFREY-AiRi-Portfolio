"""VINFREY : AiRi - sequential agents with a bounded Builder feedback loop."""

import sys

from agents import builder, planner, researcher, reviewer
from api import AgentError, create_client

MAX_RETRIES = 3


def show_section(title, content):
    print(f"\n=== {title} ===", flush=True)
    print(content, flush=True)


STARTED_MESSAGES = {
    "Planner": "Creating a plan...",
    "Researcher": "Preparing background...",
    "Reviewer": "Checking the draft...",
}


def print_event(event):
    """Default console output for workflow events, unchanged from the original CLI."""
    kind = event["type"]
    if kind == "agent_started":
        agent = event["agent"]
        message = (f"Draft {event['draft']}..." if agent == "Builder"
                   else STARTED_MESSAGES[agent])
        print(f"\n[{agent}] {message}", flush=True)
    elif kind == "agent_finished":
        title = (f"Builder - draft {event['draft']}" if event["agent"] == "Builder"
                 else event["agent"])
        show_section(title, event["content"])
    elif kind == "revision":
        show_section("Retry",
                     f"FAIL -> Builder revision {event['revision']}/{event['max_revisions']}")
    elif kind == "finished":
        message = ("PASS -> Task completed" if event["review"] == "PASS"
                   else "STOP -> Human review required")
        show_section("Final Status",
                     f"{message}\nBuilder revisions: {event['retries']}/{event['max_retries']}")


def run_workflow(user_input, *, client=None, model=None, max_retries=MAX_RETRIES,
                 on_event=None):
    """Return the final draft, review, feedback, and number of Builder revisions.

    Progress goes to on_event as plain dicts; without a hook it prints as before.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("Please enter a nonempty task.")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")
    user_input = user_input.strip()
    if client is None:
        client = create_client()
    emit = on_event or print_event

    emit({"type": "agent_started", "agent": "Planner"})
    plan = planner(user_input, client=client, model=model)
    emit({"type": "agent_finished", "agent": "Planner", "content": plan})

    emit({"type": "agent_started", "agent": "Researcher"})
    research = researcher(plan, user_input=user_input, client=client, model=model)
    emit({"type": "agent_finished", "agent": "Researcher", "content": research})

    feedback = ""
    result = ""
    retry_count = 0
    while True:
        draft = retry_count + 1
        emit({"type": "agent_started", "agent": "Builder", "draft": draft})
        result = builder(
            research, feedback, user_input=user_input, plan=plan,
            previous_result=result, client=client, model=model,
        )
        emit({"type": "agent_finished", "agent": "Builder", "draft": draft, "content": result})

        emit({"type": "agent_started", "agent": "Reviewer", "draft": draft})
        review, feedback = reviewer(
            result, user_input=user_input, plan=plan, research=research,
            client=client, model=model,
        )
        emit({"type": "agent_finished", "agent": "Reviewer", "draft": draft,
              "content": f"{review}\nFeedback: {feedback}",
              "status": review, "feedback": feedback})
        if review == "PASS" or retry_count >= max_retries:
            break
        retry_count += 1
        emit({"type": "revision", "revision": retry_count, "max_revisions": max_retries})

    emit({"type": "finished", "review": review, "retries": retry_count,
          "max_retries": max_retries})
    return {"result": result, "review": review, "feedback": feedback, "retries": retry_count}


def main():
    # Keep Thai and other model output readable on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        user_input = input("VINFREY | What do you want to do?\n> ")
        outcome = run_workflow(user_input)
    except (AgentError, ValueError) as error:
        show_section("Stopped", str(error))
        return 1
    except (KeyboardInterrupt, EOFError):
        show_section("Stopped", "Cancelled by user.")
        return 130
    return 0 if outcome["review"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
