# Build command

use the **Builder Agent** and implement the current build target.

1. **Read and follow** [.claude/agents/builder.md](.claude/agents/builder.md) in full. That agent definition is your role and source of rules.
2. **Use the spec** at [.claude/worktrees/trusting-bhabha/CLAUDE.md](.claude/worktrees/trusting-bhabha/CLAUDE.md) as the single source of truth for what to build.
3. **Execute** the implementation in the order given in the builder agent (steps 1–5 for the Document Explorer Workflow). Work through the next unfulfilled step; if the user asks for a specific step or range, do that instead.
4. **Verify** after each step: backend and Streamlit should still run; after step 5, the new workflow must be reachable and behave as specified.

Do not add scope beyond what the Document Explorer Workflow section in CLAUDE.md describes. When a step is done, briefly state what was implemented and which files were changed.
