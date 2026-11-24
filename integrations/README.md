# Verstak integration

Verstak already accepts an OpenAI-compatible chat-completions URL. Copy the settings from [verstak-agent.env.example](verstak-agent.env.example) into your local Verstak environment with the desired model alias. If Verstak runs in Docker Desktop and the gateway runs on Windows, use `http://host.docker.internal:8088/v1/chat/completions` and expose the gateway to the Docker host network. Set the same secret as the gateway's `VERIFINE_GATEWAY_KEY` in Verstak's `VERSTAK_LLM_API_KEY` if you enable authentication. Use `VERSTAK_LLM_URL` with the full chat-completions route.

Verstak owns its proposal schema, evidence checks, approvals and action execution. Verifine handles response generation and token accounting. Run the existing Verstak triage eval for this provider and measure its own pipeline separately from the bilingual extraction benchmark.
