"""Basic chat example with pneuma-core."""

import asyncio

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.storage.sqlite import SQLiteStorageBackend


async def main() -> None:
    # Load character from YAML
    character = CharacterSheet.load("aine.character.yaml")

    # Create runtime engine
    engine = RuntimeEngine(
        character=character.to_character(),
        llm=ClaudeAdapter(api_key="your-anthropic-api-key"),
        storage=SQLiteStorageBackend("aine.db"),
    )

    # Chat loop
    print(f"Chatting with {character.name}. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = await engine.chat(user_input)
        print(f"{character.name}: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
