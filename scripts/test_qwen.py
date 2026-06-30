"""Confirm the planner's own ask_model reaches the live Qwen endpoint.

This tests the EXACT code path the planner uses (a raw POST to QWEN_ENDPOINT), not
the OpenAI client from the DashScope snippet, so a pass here means the planner itself
can talk to the model. Run this BEFORE involving the simulator.

Set these three environment variables first (international DashScope shown):

    export QWEN_ENDPOINT="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    export QWEN_MODEL="QWEN_MODEL=qwen-vl-max"
    export QWEN_API_KEY="sk-ws-H.YMPXXX.KFZG.MEUCIQCV_5-eaytWESlUASSb_e8ZhY1K3TN5LQEDMCqax11ybgIgCBXscNw5jjYao5euN7KGXSx5jN2AO22lVpckUCJz_sQ"

Then:  python test_qwen.py

A pass prints the model's one-word reply. A failure prints which of the three values
is likely wrong, read from the error.
"""

import os
import planner   # uses the same QWEN_ENDPOINT / QWEN_MODEL / QWEN_API_KEY it reads at import


def main():
    # Show what the planner is actually pointed at (key masked).
    key = planner.QWEN_API_KEY
    masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else "(set)"
    print("Endpoint:", planner.QWEN_ENDPOINT)
    print("Model:   ", planner.QWEN_MODEL)
    print("API key: ", masked if key != "EMPTY" else "EMPTY  <-- not set! export QWEN_API_KEY")
    print()

    # A trivial text-only request through the planner's real ask_model.
    messages = [{"role": "user",
                 "content": "Reply with exactly one word: ready"}]
    try:
        reply = planner.ask_model(messages, max_tokens=10)
        print("MODEL REPLY:", repr(reply))
        print("\nSUCCESS - the planner can reach Qwen. You can wire the loop now.")
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e))
        print("\nLikely cause, from the error above:")
        print("  401 / Unauthorized       -> QWEN_API_KEY is wrong or unset")
        print("  404 / model not found    -> QWEN_MODEL string is wrong")
        print("  404 on the URL itself    -> QWEN_ENDPOINT must END WITH /chat/completions")
        print("  connection / timeout     -> wrong region host, or no network egress")


if __name__ == "__main__":
    main()